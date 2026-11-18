"""Export a policy of a SuperMimic run for PAL's pal_policy_controller.

  uv run --with onnxruntime python -m supermimic.scripts.export_pal --checkpoint-dir <run> \
    --motion-file <the training motions, or a sample of them> --output <policy dir>/<name>.onnx

A policy is one goal encoder of the run with its decoder, ``--encoder`` (the motion one unless
told otherwise). A run with several encoders gives one ONNX per encoder, kept in the same folder
and named ``<name>_<encoder>.onnx``: PAL's launch file picks the policy by that ending.

pal_policy_controller builds the observation from the robot and a streamed reference, in the
pal_mjlab layout its metadata describes: the terms in ``observation_names`` order, the last one
being the goal of the encoder.

- ``motion_goal``, frame by frame ``[joint_pos, joint_vel, rot6d]``, rot6d being the first two
  columns of the rotation matrix, column after column. The actor reads the same values with its
  goal term by term (all the joint frames, then all the orientations) and its rot6d row after row,
  so the exported graph starts by permuting its input into the actor's layout.
- ``velocity_command``, ``[vx, vy, wz]`` in the base frame, as the actor reads it.
- ``upper_body_goal``, frame by frame ``[base height, base lin vel, base ang vel, gravity,
  joint_pos, joint_vel, CoM to feet distance]`` over the joints that are not of the legs, as the
  actor reads it (see make_upper_body_task()).

Without a reference, the controller holds a still goal. The actor only ever saw goals from the
motions, so that goal must be a stand that they hold: ``goal_hold_joint_pos`` (and the
``goal_hold_base_height`` of an upper-body goal) is the most typical frame of their still, upright
stretches (see stand_pose()), hence the motion file should be the training motions or a sample of
them.

The runner settings must be the ones the run was trained with (e.g. ``--runner.history-length``
and ``--runner.slot-mix``, which says which encoders the run has), and so must
``--runner.goal-frames`` and ``--runner.goal-stride``: loading the checkpoint fails if they size
it otherwise.
``env.yaml`` and ``agent.yaml`` are written next to the ONNX.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, cast

import mjlab.tasks  # noqa: F401
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import BaseAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.utils.os import dump_yaml

import supermimic.tasks  # noqa: F401
from supermimic import representations
from supermimic.motion_api import motion_command, set_motion_source
from supermimic.runners.exporter import _export_onnx, _SingleEncoderOnnx
from supermimic.runners.run_config import check_representation_matches
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.tasks.task_spec import build_actor_obs_td

ROT6D = 6

# The actor's sensor terms, with the names pal_policy_controller gives them: the gravity and the
# angular velocity are those of the root body (base_link), not of the IMU on the torso.
PROPRIO_TERMS = {
  "projected_gravity": "base_projected_gravity",
  "base_ang_vel": "base_ang_vel",
  "joint_pos": "joint_pos",
  "joint_vel": "joint_vel",
  "actions": "actions",
}

# The goal term of each encoder, with the name pal_policy_controller gives it.
GOAL_TERMS = {
  "motion": "motion_goal",
  "velocity": "velocity_command",
  "upper_body": "upper_body_goal",
}

# A stand frame: still joints, root upright and still, for at least STAND_SECONDS in a row.
STAND_JOINT_SPEED = 0.15  # rad/s, norm over the joints.
STAND_TILT = np.radians(4.0)
STAND_ROOT_SPEED = 0.05  # m/s + rad/s.
STAND_SECONDS = 1.0


@dataclass
class ExportConfig:
  checkpoint_dir: str
  """The run folder, with ``actor.pt`` and ``world_model.pt``."""
  motion_file: str
  """Motions of the robot: they build the environment and give the stand held without a
  reference. The training motions, or a sample of them."""
  output: str
  """The ``.onnx`` file to write."""
  encoder: Literal["motion", "velocity", "upper_body"] = "motion"
  """The goal encoder exported with the decoder."""
  task_id: str = "SuperMimic-GenCoord-Kangaroo"
  runner: SuperMimicRunnerCfg = field(
    default_factory=lambda: SuperMimicRunnerCfg(actor_obs="proprio")
  )
  device: str | None = None


def controller_index(proprio_dim: int, num_joints: int) -> torch.Tensor:
  """For each actor input, its position in the pal_policy_controller observation.

  Actor: ``[proprio, frames × (pos, vel), frames × rot6d row-major]``.
  Controller: ``[proprio, frames × (pos, vel, rot6d column-major)]``.
  Both lay the proprio history out term by term, each term from its oldest frame to its newest.
  """
  joints_size = 2 * num_joints
  frame_size = joints_size + ROT6D
  index = list(range(proprio_dim))
  for k in range(representations.NUM_GOAL_FRAMES):
    frame = proprio_dim + k * frame_size
    index += range(frame, frame + joints_size)
  for k in range(representations.NUM_GOAL_FRAMES):
    rot6d = proprio_dim + k * frame_size + joints_size
    index += [rot6d + col * 3 + row for row in range(3) for col in range(2)]
  return torch.tensor(index, dtype=torch.long)


class PalOnnx(nn.Module):
  """One encoder and the decoder, on the pal_policy_controller observation."""

  _index: torch.Tensor

  def __init__(
    self, actor, encoder: str, term_widths: list[int], num_joints: int
  ) -> None:
    super().__init__()
    self.actor = _SingleEncoderOnnx(actor, encoder, term_widths)
    # Only the motion goal is laid out otherwise by the controller.
    index = (
      controller_index(self.actor.hist_size, num_joints)
      if encoder == "motion"
      else torch.arange(self.actor.input_size)
    )
    if len(index) != self.actor.input_size:
      raise ValueError(
        f"Goal layout of {len(index)} values, the actor reads {self.actor.input_size}"
      )
    self.register_buffer("_index", index)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.actor(x.index_select(-1, self._index))


def csv(values) -> str:
  return ",".join(f"{float(v):.9g}" for v in values)


def per_action(value, action_dim: int) -> list[float]:
  if isinstance(value, torch.Tensor):
    return value[0].cpu().tolist()
  return [float(value)] * action_dim


def stand_frame(motion, fps: float) -> tuple[int, str]:
  """The stand frame whose joint positions are closest to the median of all of them, and where
  it is.

  Stand frames are those of the stretches of at least STAND_SECONDS, within one clip, with still
  joints and an upright, still root. A real frame rather than the median itself, so that the pose
  is one the motions hold as a whole.
  """
  q = motion.root_quat_w  # (w, x, y, z)
  tilt = torch.acos((1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1.0, 1.0))
  root_speed = motion.root_lin_vel_w.norm(dim=-1) + motion.root_ang_vel_w.norm(dim=-1)
  still = (
    (motion.joint_vel.norm(dim=-1) < STAND_JOINT_SPEED)
    & (tilt < STAND_TILT)
    & (root_speed < STAND_ROOT_SPEED)
  ).tolist()

  min_frames = int(round(STAND_SECONDS * fps))
  ends = motion.motion_ends.tolist()
  frames = []
  for start, end in zip([0, *ends[:-1]], ends, strict=True):
    run = start
    for frame in range(start, end + 1):
      if frame == end or not still[frame]:
        if frame - run >= min_frames:
          frames += range(run, frame)
        run = frame + 1
  if not frames:
    raise ValueError(
      f"The motions have no still, upright stretch of {STAND_SECONDS} s to stand in"
    )

  frames = torch.tensor(frames, device=motion.joint_pos.device)
  poses = motion.joint_pos[frames]
  best = (poses - poses.median(dim=0).values).norm(dim=-1).argmin()
  frame = int(frames[best])
  clip = int(np.searchsorted(ends, frame, side="right"))
  clip_start = 0 if clip == 0 else ends[clip - 1]
  source = (
    f"{motion.motion_names[clip]} frame {frame - clip_start} "
    f"(of {len(frames)} stand frames in {len(still)})"
  )
  return frame, source


def goal_metadata(env: ManagerBasedRlEnv, cfg: ExportConfig) -> dict[str, str]:
  """The size of the goal term of the exported encoder, and what the controller builds it from."""
  if cfg.encoder == "velocity":
    return {"goal_dim": "3"}

  motion = motion_command(env)
  joints = list(motion.motion.fk.joint_names)
  frames, stride = representations.NUM_GOAL_FRAMES, representations.GOAL_FRAME_STRIDE
  frame_size = 2 * len(joints) + ROT6D
  hold, hold_source = stand_frame(motion.motion, 1.0 / env.step_dt)
  hold_height = {}
  if cfg.encoder == "upper_body":
    task = cfg.runner.tasks.upper_body
    joints = [name for name in joints if task.leg_joint_token not in name]
    frames, stride = task.goal_frames, task.goal_stride
    frame_size = 11 + 2 * len(joints)
    hold_height = {"goal_hold_base_height": csv([motion.motion.root_pos_w[hold, 2]])}
  columns = [motion.motion.fk.joint_names.index(name) for name in joints]
  return {
    "goal_dim": str(frames * frame_size),
    "goal_joint_names": ",".join(joints),
    "goal_num_future_frames": str(frames),
    "goal_frame_stride": str(stride),
    "goal_hold_joint_pos": csv(motion.motion.joint_pos[hold, columns].tolist()),
    **hold_height,
    "goal_hold_source": hold_source,
  }


def pal_metadata(
  env: ManagerBasedRlEnv, actor, cfg: ExportConfig, checkpoint: str
) -> dict[str, str]:
  """The metadata pal_policy_controller parses (pal_mjlab's export format)."""
  robot = env.scene["robot"]
  observed = env.observation_manager.get_term_cfg("actor", "joint_pos").params[
    "asset_cfg"
  ]
  observed_joints = [robot.joint_names[i] for i in observed.joint_ids]
  motion = motion_command(env)
  goal_joints = [robot.joint_names[i] for i in motion.joint_ids.tolist()]
  if goal_joints != observed_joints:
    raise ValueError(
      f"Goal joints {goal_joints} are not the observed ones {observed_joints}"
    )

  action = cast(BaseAction, env.action_manager.get_term("joint_pos"))
  action_joints = list(action._target_names)
  ctrl_of_joint = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
  ctrl = [ctrl_of_joint[j] for j in action_joints]
  model = env.sim.mj_model

  names = list(env.observation_manager.active_terms["actor"])
  if names != list(PROPRIO_TERMS):
    raise ValueError(f"Actor terms {names}, expected {list(PROPRIO_TERMS)}")
  widths = [3, 3, len(observed_joints), len(observed_joints), len(action_joints)]
  goal = goal_metadata(env, cfg)

  return {
    "checkpoint": checkpoint,
    "encoder": cfg.encoder,
    "control_dt": f"{env.step_dt:.9g}",
    "joint_names": ",".join(robot.joint_names),
    "default_joint_pos": csv(robot.data.default_joint_pos[0].cpu().tolist()),
    "observation_joint_names": ",".join(observed_joints),
    "action_joint_names": ",".join(action_joints),
    "action_scale": csv(per_action(action._scale, len(action_joints))),
    "action_offset": csv(per_action(action._offset, len(action_joints))),
    "joint_stiffness": csv(model.actuator_gainprm[ctrl, 0]),
    "joint_damping": csv(-model.actuator_biasprm[ctrl, 2]),
    "observation_names": ",".join(
      [PROPRIO_TERMS[n] for n in names] + [GOAL_TERMS[cfg.encoder]]
    ),
    "observation_terms_dim": csv(widths + [int(goal.pop("goal_dim"))]),
    "observation_terms_history_length": csv([actor.history_length] * len(widths) + [1]),
    "goal_root_body": motion.cfg.anchor_body_name,
    **goal,
  }


def check_against_actor(
  onnx_path: str, actor, encoder: str, term_widths: list[int], num_joints: int
) -> float:
  """Feed the same observation to the ONNX, in the controller layout, and to the actor's own
  forward, as training builds it; return the largest difference of their actions."""
  torch.manual_seed(0)
  batch = 1
  history = torch.randn(
    batch, actor.history_length, sum(term_widths)
  )  # Oldest frame first.
  joints = torch.randn(batch, representations.NUM_GOAL_FRAMES, 2 * num_joints)
  quat = torch.nn.functional.normalize(
    torch.randn(batch, representations.NUM_GOAL_FRAMES, 4), dim=-1
  )
  rot = matrix_from_quat(quat)  # (B, N, 3, 3)
  goal = torch.cat([joints.flatten(1), rot[..., :2].flatten(-2).flatten(1)], dim=-1)
  controller_rot6d = rot[..., :2].transpose(-1, -2).flatten(-2)  # Column after column.
  controller_goal = torch.cat([joints, controller_rot6d], dim=-1).flatten(1)
  if encoder != "motion":  # The controller lays the goal out as the actor does.
    spec = next(spec for spec in actor.task_specs if spec.name == encoder)
    goal = controller_goal = torch.randn(batch, spec.goal_dim)

  # Term by term, each term's frames from oldest to newest.
  controller_history = torch.cat(
    [term.flatten(1) for term in history.split(term_widths, dim=-1)], dim=-1
  )
  controller_in = torch.cat([controller_history, controller_goal], dim=-1)

  actor = actor.cpu()
  actor.set_slot_mix({encoder: 1.0})
  goals = {
    name: torch.zeros(batch, spec.goal_dim)
    for name, spec in zip(actor._task_names, actor.task_specs, strict=True)
  }
  goals[encoder] = goal
  with torch.no_grad():
    expected = actor(build_actor_obs_td(goals, history)).numpy()
  session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
  actual = session.run(None, {"obs": controller_in.numpy()})[0]
  return float(np.abs(actual - expected).max())


def main(cfg: ExportConfig) -> None:
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  check_representation_matches(cfg.task_id, cfg.runner.representation)
  representations.set_goal_window(cfg.runner.goal_frames, cfg.runner.goal_stride)
  env_cfg = load_env_cfg(cfg.task_id, play=True)
  env_cfg.scene.num_envs = 1
  set_motion_source(env_cfg, os.path.expanduser(cfg.motion_file))
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

  checkpoint = Path(os.path.expanduser(cfg.checkpoint_dir))
  wm = build_world_model(cfg.runner, env, device)
  wm.load_bundle(torch.load(checkpoint / "world_model.pt", map_location=device))
  actor = build_actor(cfg.runner, wm, device)
  actor.load_state_dict(torch.load(checkpoint / "actor.pt", map_location=device))
  actor.eval()
  if cfg.encoder not in actor.encoders:
    raise ValueError(f"The run has no '{cfg.encoder}' encoder: {list(actor.encoders)}")
  if int(actor._goal_normalizer(cfg.encoder).count) == 0:
    raise ValueError(f"The '{cfg.encoder}' encoder of the run was never trained")

  num_joints = wm.representation.num_joints
  term_widths = wm.representation.proprio_term_widths(wm.num_actions)
  wrapper = PalOnnx(actor, cfg.encoder, term_widths, num_joints).cpu().eval()

  output = Path(os.path.expanduser(cfg.output))
  output.parent.mkdir(parents=True, exist_ok=True)
  _export_onnx(
    wrapper,
    (torch.zeros(1, wrapper.actor.input_size),),
    ["obs"],
    ["actions"],
    str(output),
  )
  metadata = pal_metadata(env.unwrapped, actor, cfg, str(checkpoint.resolve()))
  model = onnx.load(str(output))
  for key, value in metadata.items():
    model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))
  onnx.save(model, str(output))

  error = check_against_actor(str(output), actor, cfg.encoder, term_widths, num_joints)
  print(f"[INFO] ONNX vs actor, largest action difference: {error:.2e}")
  if error > 1e-4:
    raise RuntimeError("The exported ONNX does not reproduce the actor")

  dump_yaml(output.parent / "env.yaml", asdict(env_cfg))
  dump_yaml(output.parent / "agent.yaml", asdict(cfg.runner))
  for key, value in metadata.items():
    print(f"  {key}: {value[:200]}{'...' if len(value) > 200 else ''}")
  print(f"[INFO] Wrote {output}")
  env.close()


if __name__ == "__main__":
  main(tyro.cli(ExportConfig))
