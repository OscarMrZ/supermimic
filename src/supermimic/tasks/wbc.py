"""pal_mjlab's Kangaroo-Grippers WBC task, trained with SuperMimic.

The env is ``Mjlab-WBC-Flat-Pal-Kangaroo-Grippers`` as pal_mjlab builds it: its
robot, CSV motions (``OnlineMotionCommand``), sampling, rewards, curriculum,
terminations, events and proprioception. SuperMimic adds only what its learning
method needs:

- ``WbcMotionCommand``: the WBC command, plus the reference tables SuperMimic
  reads (root and joint tracks, clip ends and a differentiable forward kinematics
  of the tracked bodies) over the same CSV library.
- The actor group holds the WBC proprio terms; the WBC reference terms are the
  ``upper_body`` goal, built from the motion so the world model can imagine it.
- ``full_state`` and ``goal`` groups for the world model, and a ``motion_end``
  time out: a clip end is an episode end, so no transition jumps between clips.
- ``WbcRepresentation``: the world model's reward is the WBC reward terms with
  their weights and stds (following the curriculum), its terminations are the
  WBC ones, and its synthesized proprio is the WBC actor's (IMU gyro and gravity,
  every joint, encoder bias).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, cast

import mujoco
import torch
from mjlab.managers import ObservationGroupCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.forward_kinematics import ForwardKinematics
from mjlab.utils.lab_api.math import matrix_from_quat
from pal_mjlab.tasks.WBC.mdp.commands import (
  ROBOT_XML_DICT,
  OnlineMotion,
  OnlineMotionCommand,
  OnlineMotionCommandCfg,
)

from supermimic.representations import GenCoordRepresentation
from supermimic.utils.helpers import rotation_error_magnitude, small_matmul
from supermimic.utils.tracking_reward import _reanchor_targets

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
  from mjlab.utils.noise import UniformNoiseCfg

  from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg
  from supermimic.sim_context import SimContext

TASK_ID = "SuperMimic-GenCoord-Kangaroo-Grippers-WBC"

# The WBC actor's terms, in its order: the reference (the upper-body goal), then
# the proprio (the actor's sensor frames).
GOAL_TERMS = (
  "ref_base_height",
  "ref_base_lin_vel_b",
  "ref_base_ang_vel_b",
  "ref_gravity_b",
  "ref_joint_pos",
  "ref_joint_vel",
  "ref_cop_vs_com",
)
PROPRIO_TERMS = (
  "base_ang_vel",
  "imu_projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
)
IMU_SITE = "imu"

RUNNER_DEFAULTS: dict[str, Any] = {
  "representation": "wbc",
  "actor_obs": "proprio",
  "sampling_mode": "adaptive",
  "slot_mix": {"motion": 0.0, "velocity": 0.0, "upper_body": 1.0},
  "latent_bc_weight": 0.0,
}
"""SuperMimic runner settings this task trains with: the WBC actor (proprio and
the upper-body goal only) and pal_mjlab's adaptive reset sampling."""


# ---------------------------------------------------------------------------
# Motion command
# ---------------------------------------------------------------------------


class WbcMotion(OnlineMotion):
  """pal_mjlab's ``OnlineMotion``, as built, plus the tables SuperMimic reads.

  ``joint_pos``/``joint_vel`` (every joint), ``root_pos_w``, ``root_quat_w``
  (wxyz), ``root_lin_vel_w``, ``root_ang_vel_w`` (world frame) are float32 views
  of the CSV library (a float32 copy when it is stored at lower precision);
  ``motion_ends``/``motion_names`` are its clips; ``time_step_total`` is every
  frame (pal_mjlab reads it only as the play scrubber's range); ``fk`` computes
  the tracked bodies, the way the world model does.
  """

  root_pos_w: torch.Tensor
  root_quat_w: torch.Tensor
  root_lin_vel_w: torch.Tensor
  root_ang_vel_w: torch.Tensor
  motion_ends: torch.Tensor
  motion_names: list[str]
  fk: ForwardKinematics

  def __init__(self, base: OnlineMotion, fk: ForwardKinematics) -> None:
    # Take over the built motion as it is; OnlineMotion.__init__ already ran.
    vars(self).update(vars(base))
    lib = self.source.motion_csv
    self.joint_pos = lib.joint_pos.float()
    self.joint_vel = lib.joint_vel.float()
    self.root_pos_w = lib.root_pos.float()
    quat = lib.root_quat.float()[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    self.root_quat_w = quat / quat.norm(dim=-1, keepdim=True)  # undo quantization
    self.root_lin_vel_w = lib.root_lin_vel.float()
    self.root_ang_vel_w = lib.root_ang_vel.float()
    self.motion_ends = lib.segment_end_idx
    self.motion_names = list(lib.segment_names)
    self.time_step_total = int(self.joint_pos.shape[0])
    self.fk = fk

  def body_state(
    self, frames: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """World poses and velocities of the tracked bodies at ``frames``, without
    env origins."""
    return self.fk(
      self.root_pos_w[frames],
      self.root_quat_w[frames],
      self.joint_pos[frames],
      self.root_lin_vel_w[frames],
      self.root_ang_vel_w[frames],
      self.joint_vel[frames],
    )


class WbcMotionCommand(OnlineMotionCommand):
  """pal_mjlab's online CSV motion command, unchanged, with a ``WbcMotion``."""

  motion: WbcMotion
  joint_ids: torch.Tensor
  """The robot joints the motion and the world model's state hold: every one."""

  def _load_motion(self) -> WbcMotion:
    base = super()._load_motion()
    self.joint_ids = torch.arange(self.robot.num_joints, device=self.device)
    robot_type = cast(OnlineMotionCommandCfg, self.cfg).robot_type
    csv_joints = _csv_joint_names(robot_type)
    if list(self.robot.joint_names) != csv_joints:
      raise ValueError(
        f"robot joints {self.robot.joint_names} are not the CSV columns {csv_joints}"
      )
    fk = ForwardKinematics(
      self._env.sim.mj_model,
      self.robot.indexing.body_ids,
      self.robot.indexing.joint_q_adr,
      self.body_indexes,
      device=str(self.device),
    )
    return WbcMotion(base, fk)

  def full_joint_state(
    self, joint_pos: torch.Tensor, joint_vel: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """The CSVs hold every joint, passive ones included."""
    return joint_pos, joint_vel


@dataclass(kw_only=True)
class WbcMotionCommandCfg(OnlineMotionCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> WbcMotionCommand:
    return WbcMotionCommand(self, env)


def motion_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The next step would leave the clip: pal_mjlab's command resamples there,
  which teleports the robot; ending the episode keeps that jump out of the
  world model's transitions."""
  command = cast(WbcMotionCommand, env.command_manager.get_term(command_name))
  ends = command.motion.segment_end_idx[command.rand_motion]
  return (command.time_steps + 1 >= ends) & command.moving_envs


def _csv_joint_names(robot_type: str) -> list[str]:
  """The joints the CSV columns hold after the root: the XML's, in order."""
  spec = mujoco.MjSpec.from_file(str(ROBOT_XML_DICT[robot_type]))
  bodies = {b.name for b in spec.bodies}
  for ex in list(spec.excludes):  # as pal_mjlab's OnlineCsvToNpz does
    if ex.bodyname1 not in bodies or ex.bodyname2 not in bodies:
      spec.delete(ex)
  model = spec.compile()
  free = mujoco.mjtJoint.mjJNT_FREE
  return [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] != free]


# ---------------------------------------------------------------------------
# Env configuration
# ---------------------------------------------------------------------------


def wbc_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """``Mjlab-WBC-Flat-Pal-Kangaroo-Grippers`` with SuperMimic's additions."""
  from pal_mjlab.tasks.WBC.kangaroo.env_cfgs import (
    pal_kangaroo_grippers_flat_wbc_env_cfg,
  )

  from supermimic.tasks.env_cfgs import full_state_group, goal_obs_group

  cfg = pal_kangaroo_grippers_flat_wbc_env_cfg(play=play)

  assert cfg.commands is not None
  wbc_cmd = cfg.commands["motion"]
  assert isinstance(wbc_cmd, OnlineMotionCommandCfg)
  cfg.commands["motion"] = WbcMotionCommandCfg(
    **{f.name: getattr(wbc_cmd, f.name) for f in fields(wbc_cmd)}
  )

  actor = cfg.observations["actor"]
  names = tuple(actor.terms)
  if names != GOAL_TERMS + PROPRIO_TERMS:
    raise ValueError(f"WBC actor terms changed: {names}")
  # The proprio keeps its noise config; SuperMimic draws that noise itself (the
  # same half-widths), on real and imagined frames alike.
  cfg.observations["actor"] = ObservationGroupCfg(
    terms={name: actor.terms[name] for name in PROPRIO_TERMS},
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["full_state"] = full_state_group()
  cfg.observations["goal"] = goal_obs_group()

  cfg.terminations["motion_end"] = TerminationTermCfg(
    func=motion_end, time_out=True, params={"command_name": "motion"}
  )
  return cfg


def wbc_runner_cfg() -> SuperMimicRunnerCfg:
  """SuperMimic's runner cfg for this task: ``RUNNER_DEFAULTS`` and the WBC goal
  noise. Training and play start from it."""
  from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg

  runner = dataclasses.replace(SuperMimicRunnerCfg(), **RUNNER_DEFAULTS)
  upper_body = dataclasses.replace(runner.tasks.upper_body, **wbc_goal_noise())
  runner.tasks = dataclasses.replace(runner.tasks, upper_body=upper_body)
  return runner


def wbc_goal_noise() -> dict[str, Any]:
  """``UpperBodyTaskCfg`` noise fields: the WBC actor's reference-term noise, read
  from pal_mjlab's own config."""
  from pal_mjlab.tasks.WBC.kangaroo.env_cfgs import (
    pal_kangaroo_grippers_flat_wbc_env_cfg,
  )

  terms = pal_kangaroo_grippers_flat_wbc_env_cfg().observations["actor"].terms

  def half(name: str) -> tuple[float, ...]:
    noise = cast("UniformNoiseCfg", terms[name].noise)
    lo, hi = noise.n_min, noise.n_max
    lo_t = lo if isinstance(lo, tuple) else (lo,)
    hi_t = hi if isinstance(hi, tuple) else (hi,)
    return tuple((h - lo_) / 2 for lo_, h in zip(lo_t, hi_t, strict=True))

  return {
    "height_noise": half("ref_base_height")[0],
    "lin_vel_noise": half("ref_base_lin_vel_b"),
    "ang_vel_noise": half("ref_base_ang_vel_b"),
    "gravity_noise": half("ref_gravity_b")[0],
    "joint_pos_noise": half("ref_joint_pos")[0],
    "joint_vel_noise": half("ref_joint_vel")[0],
    "com_feet_noise": half("ref_cop_vs_com")[0],
  }


# ---------------------------------------------------------------------------
# World-model representation
# ---------------------------------------------------------------------------


@dataclass
class _RewardTerm:
  name: str
  kind: str
  std: float = 1.0
  bodies: torch.Tensor | None = None
  joints: torch.Tensor | None = None
  hull: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None


# Reward functions the world model reproduces, by function or class name.
_REWARD_KINDS = {
  "motion_global_anchor_position_error_exp": "anchor_pos",
  "motion_global_anchor_orientation_error_exp": "anchor_ori",
  "motion_relative_body_position_error_exp": "body_pos",
  "motion_relative_body_orientation_error_exp": "body_ori",
  "motion_global_body_linear_velocity_error_exp": "body_lin_vel",
  "motion_global_body_angular_velocity_error_exp": "body_ang_vel",
  "motion_anchor_linear_velocity_body_error_exp": "anchor_lin_vel_b",
  "motion_anchor_angular_velocity_body_error_exp": "anchor_ang_vel_b",
  "motion_joint_position_error_exp": "joint_pos",
  "motion_joint_velocity_error_exp": "joint_vel",
  "action_rate_l2": "action_rate",
  "joint_pos_limits": "joint_limit",
  "joint_limits_convex_hull": "convex_hull",
  "is_alive": "alive",
}
# Read contacts or sensors the world model does not predict: left out of its reward.
_NOT_IMAGINED = {"self_collision_cost", "angular_momentum_penalty"}


def _func_name(func: Any) -> str:
  return getattr(func, "__name__", None) or type(func).__name__


class WbcRepresentation(GenCoordRepresentation):
  """Generalized coordinates over every robot joint, with the WBC reward,
  terminations and actor proprio.

  Proprio layout ``[imu gyro(3), imu gravity(3), joint_pos_rel(N),
  joint_vel(N), actions(A)]``: the WBC actor's order.
  """

  _reward_weights: torch.Tensor
  _proprio_noise_bounds: torch.Tensor

  def proprio_term_widths(self, num_actions: int) -> list[int]:
    return [3, 3, self.num_joints, self.num_joints, num_actions]

  def bind(self, env: ManagerBasedRlEnv, sim_context: SimContext) -> None:
    super().bind(env, sim_context)
    device = self._default_joint_pos.device
    cmd = cast(WbcMotionCommand, env.command_manager.get_term("motion"))
    body_names = list(cmd.cfg.body_names)
    robot = cmd.robot
    self._anchor = cmd.motion_anchor_body_index
    if self._anchor != 0:
      raise ValueError("the WBC anchor must be the first tracked body (the root)")

    # IMU: a site on a tracked body, at a fixed rotation from it.
    model = env.sim.mj_model
    site = next(
      s for s in range(model.nsite) if model.site(s).name.split("/")[-1] == IMU_SITE
    )
    imu_body = model.body(int(model.site_bodyid[site])).name.split("/")[-1]
    self._imu_body = body_names.index(imu_body)
    site_quat = torch.as_tensor(model.site_quat[site], dtype=torch.float32)
    self._imu_rot = matrix_from_quat(site_quat).to(device)

    columns = sim_context.find("encoder_bias")
    self._encoder_bias = None if columns is None else (columns, sim_context)

    # Proprio noise: the WBC actor terms' own half-widths.
    terms = env.cfg.observations["actor"].terms
    widths = self.proprio_term_widths(env.action_manager.total_action_dim)
    halves = []
    for name, width in zip(PROPRIO_TERMS, widths, strict=True):
      noise = cast("UniformNoiseCfg | None", terms[name].noise)
      h = 0.0 if noise is None else (float(noise.n_max) - float(noise.n_min)) / 2  # type: ignore[arg-type]
      halves.append(torch.full((width,), h, device=device))
    self._proprio_noise_bounds = torch.cat(halves)

    self._bind_rewards(env, body_names, robot, device)
    self._bind_terminations(env, body_names)

  def _bind_rewards(self, env, body_names, robot, device) -> None:
    manager = env.reward_manager
    self._reward_terms: list[_RewardTerm] = []
    unknown = []
    for name in manager.active_terms:
      term_cfg = manager.get_term_cfg(name)
      fname = _func_name(term_cfg.func)
      if fname in _NOT_IMAGINED:
        continue
      kind = _REWARD_KINDS.get(fname)
      if kind is None:
        unknown.append(f"{name} ({fname})")
        continue
      params = term_cfg.params
      term = _RewardTerm(name=name, kind=kind, std=float(params.get("std") or 1.0))
      if "body_names" in params and params["body_names"] is not None:
        term.bodies = torch.tensor(
          [i for i, b in enumerate(body_names) if b in params["body_names"]],
          device=device,
        )
      asset_cfg = params.get("asset_cfg")
      if kind in ("joint_pos", "joint_vel", "joint_limit") and asset_cfg is not None:
        ids = asset_cfg.joint_ids
        if not isinstance(ids, slice):
          term.joints = torch.as_tensor(ids, device=device)
      if kind == "convex_hull":
        hull = term_cfg.func
        term.hull = [
          (
            torch.as_tensor(robot.find_joints(group)[0], device=device),
            hull.equation_coeff_A.to(device),
            hull.equation_coeff_b.to(device),
          )
          for group in params["joint_names_group"]
        ]
      self._reward_terms.append(term)
    if unknown:
      raise ValueError(f"WBC reward terms the world model cannot imagine: {unknown}")
    self.register_buffer(
      "_reward_weights",
      torch.zeros(len(self._reward_terms), device=device),
      persistent=False,
    )
    self.sync_env(env)

  def _bind_terminations(self, env, body_names) -> None:
    ee = env.cfg.terminations.get("ee_body_pos")
    self._ee_threshold = 1.0
    self._ee_bodies: torch.Tensor | None = None
    if ee is not None:
      wanted = ee.params.get("body_names")
      # As the sim's term: bodies outside the tracked ones are not checked.
      rows = [i for i, b in enumerate(body_names) if wanted is None or b in wanted]
      if rows:
        self._ee_bodies = torch.tensor(rows, device=self._default_joint_pos.device)
        self._ee_threshold = float(ee.params["threshold"])

  def sync_env(self, env: ManagerBasedRlEnv) -> None:
    """Copy the reward weights the curriculum has set into the world model."""
    manager = env.reward_manager
    weights = [manager.get_term_cfg(t.name).weight for t in self._reward_terms]
    self._reward_weights.copy_(torch.tensor(weights, dtype=torch.float32))

  # -- proprio -------------------------------------------------------------

  def compute_wm_obs(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    joint_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> torch.Tensor:
    _, rot, _, ang_vel = self.world_frame_bodies(aux_state, rep_state)
    imu_rot = small_matmul(rot[..., self._imu_body, :, :], self._imu_rot)
    imu_rot_t = imu_rot.transpose(-1, -2)
    gyro = small_matmul(imu_rot_t, ang_vel[..., self._imu_body, :, None])[..., 0]
    gravity = -imu_rot_t[..., :, 2]  # R^T (0, 0, -1)
    qj, vj = joint_state.chunk(2, dim=-1)
    joint_pos_rel = qj - self._default_joint_pos
    if self._encoder_bias is not None:
      columns, sim_context = self._encoder_bias
      joint_pos_rel = joint_pos_rel + sim_context.denormalize(sim_ctx, columns)
    return torch.cat([gyro, gravity, joint_pos_rel, vj], dim=-1)

  # -- reward --------------------------------------------------------------

  def compute_wm_reward_from_reference(
    self,
    next_rep_state: torch.Tensor,
    next_aux_state: torch.Tensor,
    next_joint_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    target_body_pos_w: torch.Tensor,
    target_body_quat_w: torch.Tensor,
    target_body_lin_vel_w: torch.Tensor,
    target_body_ang_vel_w: torch.Tensor,
    ref_joint_state: torch.Tensor | None = None,
    term_prob: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """The WBC reward: ``dt * sum(weight * term)``, each term as pal_mjlab's."""
    assert ref_joint_state is not None and term_prob is not None
    pos, rot, lin_vel, ang_vel = self.world_frame_bodies(next_aux_state, next_rep_state)
    a = self._anchor
    tgt_rel_pos, tgt_rel_mat = _reanchor_targets(
      target_body_pos_w,
      target_body_quat_w,
      next_aux_state[..., :3],
      next_aux_state[..., 3:7],
    )
    tgt_mat = matrix_from_quat(target_body_quat_w)
    qj, vj = next_joint_state.chunk(2, dim=-1)
    ref_qj, ref_vj = ref_joint_state.chunk(2, dim=-1)

    def rows(x: torch.Tensor, term: _RewardTerm) -> torch.Tensor:
      return x if term.bodies is None else x[..., term.bodies, :]

    def cols(x: torch.Tensor, term: _RewardTerm) -> torch.Tensor:
      return x if term.joints is None else x[..., term.joints]

    values = []
    for term in self._reward_terms:
      k, var = term.kind, term.std**2
      if k == "anchor_pos":
        err = ((target_body_pos_w[..., a, :] - pos[..., a, :]) ** 2).sum(-1)
        v = torch.exp(-err / var)
      elif k == "anchor_ori":
        err = rotation_error_magnitude(rot[..., a, :, :], tgt_mat[..., a, :, :]) ** 2
        v = torch.exp(-err / var)
      elif k == "body_pos":
        err = ((rows(tgt_rel_pos, term) - rows(pos, term)) ** 2).sum(-1).mean(-1)
        v = torch.exp(-err / var)
      elif k == "body_ori":
        b = term.bodies
        tm = tgt_rel_mat if b is None else tgt_rel_mat[..., b, :, :]
        pm = rot if b is None else rot[..., b, :, :]
        err = (rotation_error_magnitude(pm, tm) ** 2).mean(-1)
        v = torch.exp(-err / var)
      elif k == "body_lin_vel":
        err = (rows(target_body_lin_vel_w, term) - rows(lin_vel, term)) ** 2
        v = torch.exp(-err.sum(-1).mean(-1) / var)
      elif k == "body_ang_vel":
        err = (rows(target_body_ang_vel_w, term) - rows(ang_vel, term)) ** 2
        v = torch.exp(-err.sum(-1).mean(-1) / var)
      elif k == "anchor_lin_vel_b":
        # Expressed in the robot anchor frame: a rotation keeps the norm.
        err = ((target_body_lin_vel_w[..., a, :] - lin_vel[..., a, :]) ** 2).sum(-1)
        v = torch.exp(-err / var)
      elif k == "anchor_ang_vel_b":
        err = ((target_body_ang_vel_w[..., a, :] - ang_vel[..., a, :]) ** 2).sum(-1)
        v = torch.exp(-err / var)
      elif k == "joint_pos":
        v = torch.exp(-((cols(ref_qj, term) - cols(qj, term)) ** 2).sum(-1) / var)
      elif k == "joint_vel":
        v = torch.exp(-((cols(ref_vj, term) - cols(vj, term)) ** 2).sum(-1) / var)
      elif k == "action_rate":
        v = ((action - last_action) ** 2).sum(-1)
      elif k == "joint_limit":
        lo, hi = self._soft_joint_pos_limits.unbind(-1)
        q = cols(qj, term)
        v = (cols(lo, term) - q).clamp(min=0) + (q - cols(hi, term)).clamp(min=0)
        v = v.sum(-1)
      elif k == "convex_hull":
        assert term.hull is not None
        v = torch.zeros_like(term_prob)
        for ids, coeff_a, coeff_b in term.hull:
          dist = (qj[..., ids] @ coeff_a.T + coeff_b).clamp(min=0).amax(-1)
          v = v + dist**2
      elif k == "alive":
        v = 1.0 - term_prob
      else:
        raise AssertionError(k)
      values.append(v)
    reward = (torch.stack(values, dim=-1) * self._reward_weights).sum(-1)
    return self.dt * reward

  # -- terminations --------------------------------------------------------

  def termination_errors(
    self,
    aux_state: torch.Tensor,
    rep_state: torch.Tensor,
    target_body_pos_w: torch.Tensor,
    target_body_quat_w: torch.Tensor,
  ) -> torch.Tensor | None:
    """The WBC ``ee_body_pos`` term, normalized by its threshold: the height of
    the tracked end bodies against the re-anchored reference."""
    if self._ee_bodies is None:
      return None
    pos, _, _, _ = self.world_frame_bodies(aux_state, rep_state)
    tgt_pos, _ = _reanchor_targets(
      target_body_pos_w, target_body_quat_w, aux_state[..., :3], aux_state[..., 3:7]
    )
    b = self._ee_bodies
    err = (tgt_pos[..., b, 2] - pos[..., b, 2]).abs().amax(-1)
    return err / self._ee_threshold
