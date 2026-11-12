"""Export the motion policy of a SuperMimic run for PAL's pal_policy_controller.

  uv run --with onnxruntime python -m supermimic.scripts.export_pal --checkpoint-dir <run> \
    --motion-file <the training motions, or a sample of them> --output <policy dir>/<name>.onnx

pal_policy_controller builds the observation from the robot and a streamed motion reference, in
the pal_mjlab layout its metadata describes: the terms in ``observation_names`` order, and the
``motion_goal`` term frame by frame as ``[joint_pos, joint_vel, rot6d]``, rot6d being the first two
columns of the rotation matrix, column after column. The actor reads the same values with its goal
term by term (all the joint frames, then all the orientations) and its rot6d row after row, so the
exported graph starts by permuting its input into the actor's layout.

Without a reference, the controller holds a still goal. The actor only ever saw goals from the
motions, so that goal must be a stand that they hold: ``goal_hold_joint_pos`` is the most typical
frame of their still, upright stretches (see stand_pose()), hence the motion file should be the
training motions or a sample of them.

The runner settings must be the ones the run was trained with: loading the checkpoint fails if
they size it otherwise. ``env.yaml`` and ``agent.yaml`` are written next to the ONNX.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mjlab.tasks  # noqa: F401
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.utils.os import dump_yaml

import supermimic.tasks  # noqa: F401
from supermimic.representations import GOAL_FRAME_STRIDE, NUM_GOAL_FRAMES
from supermimic.runners.exporter import _export_onnx, _SingleEncoderOnnx
from supermimic.runners.run_config import check_representation_matches
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)

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
  task_id: str = "SuperMimic-GenCoord-Kangaroo"
  runner: SuperMimicRunnerCfg = field(
    default_factory=lambda: SuperMimicRunnerCfg(actor_obs="proprio")
  )
  device: str | None = None


def controller_index(proprio_dim: int, num_joints: int) -> torch.Tensor:
  """For each actor input, its position in the pal_policy_controller observation.

  Actor: ``[proprio, frames × (pos, vel), frames × rot6d row-major]``.
  Controller: ``[proprio, frames × (pos, vel, rot6d column-major)]``.
  """
  joints_size = 2 * num_joints
  frame_size = joints_size + ROT6D
  index = list(range(proprio_dim))
  for k in range(NUM_GOAL_FRAMES):
    frame = proprio_dim + k * frame_size
    index += range(frame, frame + joints_size)
  for k in range(NUM_GOAL_FRAMES):
    rot6d = proprio_dim + k * frame_size + joints_size
    index += [rot6d + col * 3 + row for row in range(3) for col in range(2)]
  return torch.tensor(index, dtype=torch.long)


class PalMotionOnnx(nn.Module):
  """The motion encoder and the decoder, on the pal_policy_controller observation."""

  def __init__(self, actor, term_widths: list[int], num_joints: int) -> None:
    super().__init__()
    if actor.history_length != 1:
      raise ValueError("pal_policy_controller exports only support history_length 1")
    self.actor = _SingleEncoderOnnx(actor, "motion", term_widths)
    index = controller_index(self.actor.hist_size, num_joints)
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


def stand_pose(motion, fps: float) -> tuple[torch.Tensor, str]:
  """The joint positions of the stand frame closest to the median of all of them, and where it is.

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
  return poses[best].cpu(), source


def pal_metadata(env: ManagerBasedRlEnv, actor, checkpoint: str) -> dict[str, str]:
  """The metadata pal_policy_controller parses (pal_mjlab's export format)."""
  robot = env.scene["robot"]
  observed = env.observation_manager.get_term_cfg("actor", "joint_pos").params[
    "asset_cfg"
  ]
  observed_joints = [robot.joint_names[i] for i in observed.joint_ids]
  motion = env.command_manager.get_term("motion")
  goal_joints = [robot.joint_names[i] for i in motion.joint_ids.tolist()]
  if goal_joints != observed_joints:
    raise ValueError(
      f"Goal joints {goal_joints} are not the observed ones {observed_joints}"
    )

  action = env.action_manager.get_term("joint_pos")
  action_joints = list(action._target_names)
  ctrl_of_joint = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
  ctrl = [ctrl_of_joint[j] for j in action_joints]
  model = env.sim.mj_model

  names = list(env.observation_manager.active_terms["actor"])
  if names != list(PROPRIO_TERMS):
    raise ValueError(f"Actor terms {names}, expected {list(PROPRIO_TERMS)}")
  widths = [3, 3, len(observed_joints), len(observed_joints), len(action_joints)]
  goal_size = NUM_GOAL_FRAMES * (2 * len(goal_joints) + ROT6D)
  hold, hold_source = stand_pose(motion.motion, 1.0 / env.step_dt)

  return {
    "checkpoint": checkpoint,
    "control_dt": f"{env.step_dt:.9g}",
    "joint_names": ",".join(robot.joint_names),
    "default_joint_pos": csv(robot.data.default_joint_pos[0].cpu().tolist()),
    "observation_joint_names": ",".join(observed_joints),
    "action_joint_names": ",".join(action_joints),
    "action_scale": csv(per_action(action._scale, len(action_joints))),
    "action_offset": csv(per_action(action._offset, len(action_joints))),
    "joint_stiffness": csv(model.actuator_gainprm[ctrl, 0]),
    "joint_damping": csv(-model.actuator_biasprm[ctrl, 2]),
    "observation_names": ",".join([PROPRIO_TERMS[n] for n in names] + ["motion_goal"]),
    "observation_terms_dim": csv(widths + [goal_size]),
    "observation_terms_history_length": csv([actor.history_length] * len(widths) + [1]),
    "goal_joint_names": ",".join(goal_joints),
    "goal_num_future_frames": str(NUM_GOAL_FRAMES),
    "goal_frame_stride": str(GOAL_FRAME_STRIDE),
    "goal_root_body": motion.cfg.anchor_body_name,
    "goal_hold_joint_pos": csv(hold.tolist()),
    "goal_hold_source": hold_source,
  }


def check_against_actor(
  onnx_path: str, wrapper: PalMotionOnnx, num_joints: int
) -> float:
  """Feed the same observation to the ONNX, in the controller layout, and to the actor, in its
  own layout built as in training; return the largest difference of their actions."""
  torch.manual_seed(0)
  batch, proprio_dim = 1, wrapper.actor.hist_size
  proprio = torch.randn(batch, proprio_dim)
  joints = torch.randn(batch, NUM_GOAL_FRAMES, 2 * num_joints)
  quat = torch.nn.functional.normalize(torch.randn(batch, NUM_GOAL_FRAMES, 4), dim=-1)
  rot = matrix_from_quat(quat)  # (B, N, 3, 3)
  actor_in = torch.cat(
    [proprio, joints.flatten(1), rot[..., :2].flatten(-2).flatten(1)], dim=-1
  )
  controller_rot6d = rot[..., :2].transpose(-1, -2).flatten(-2)  # Column after column.
  controller_in = torch.cat(
    [proprio, torch.cat([joints, controller_rot6d], dim=-1).flatten(1)], dim=-1
  )
  with torch.no_grad():
    expected = wrapper.actor.cpu()(actor_in).numpy()
  session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
  actual = session.run(None, {"obs": controller_in.numpy()})[0]
  return float(np.abs(actual - expected).max())


def main(cfg: ExportConfig) -> None:
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  check_representation_matches(cfg.task_id, cfg.runner.representation)
  env_cfg = load_env_cfg(cfg.task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["motion"].motion_file = os.path.expanduser(cfg.motion_file)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

  checkpoint = Path(os.path.expanduser(cfg.checkpoint_dir))
  wm = build_world_model(cfg.runner, env, device)
  wm.load_bundle(torch.load(checkpoint / "world_model.pt", map_location=device))
  actor = build_actor(cfg.runner, wm, device)
  actor.load_state_dict(torch.load(checkpoint / "actor.pt", map_location=device))
  actor.eval()

  num_joints = wm.representation.num_joints
  term_widths = wm.representation.proprio_term_widths(wm.num_actions)
  wrapper = PalMotionOnnx(actor, term_widths, num_joints).cpu().eval()

  output = Path(os.path.expanduser(cfg.output))
  output.parent.mkdir(parents=True, exist_ok=True)
  _export_onnx(
    wrapper,
    (torch.zeros(1, wrapper.actor.input_size),),
    ["obs"],
    ["actions"],
    str(output),
  )
  metadata = pal_metadata(env.unwrapped, actor, str(checkpoint.resolve()))
  model = onnx.load(str(output))
  for key, value in metadata.items():
    model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))
  onnx.save(model, str(output))

  error = check_against_actor(str(output), wrapper, num_joints)
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
