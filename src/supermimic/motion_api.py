"""The motion-command contract SuperMimic reads from an env's ``"motion"`` term.

Two commands satisfy it: mjlab_research's ``MotionCommand`` (npz motions) and
``supermimic.tasks.wbc.WbcMotionCommand`` (pal_mjlab's CSV motions). mjlab types
``command_manager.get_term`` as a plain ``CommandTerm``, so every caller goes
through :func:`motion_command` to read the term as what SuperMimic relies on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
  import torch
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
  from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
  from mjlab.utils.forward_kinematics import ForwardKinematics
  from pal_mjlab.tasks.WBC.mdp.commands import OnlineMotionCommandCfg

  type MotionCommandCfgLike = MotionCommandCfg | OnlineMotionCommandCfg


class MotionLibrary(Protocol):
  """Reference tracks over every frame of the concatenated clips."""

  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  root_pos_w: torch.Tensor
  root_quat_w: torch.Tensor
  """wxyz."""
  root_lin_vel_w: torch.Tensor
  root_ang_vel_w: torch.Tensor
  motion_ends: torch.Tensor
  """Exclusive end frame of each clip, increasing."""
  motion_names: list[str]
  time_step_total: int
  """Frames in the library."""
  fk: ForwardKinematics

  def body_state(
    self, frames: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """World poses (wxyz) and velocities of the tracked bodies at ``frames``."""
    ...


class MotionCommandLike(Protocol):
  """What SuperMimic reads from the ``"motion"`` command term."""

  cfg: Any
  """The command's cfg: ``anchor_body_name``, ``body_names``, ``sampling_mode``
  and its motion source, whose field differs between the two commands."""
  motion: MotionLibrary
  robot: Entity
  device: str
  time_steps: torch.Tensor
  joint_ids: torch.Tensor
  """Robot joints the motion and the world model's state hold."""
  body_indexes: torch.Tensor
  motion_anchor_body_index: int
  bin_count: int

  def full_joint_state(
    self, joint_pos: torch.Tensor, joint_vel: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]: ...

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None: ...

  def set_sampling_distribution(self, probs: torch.Tensor) -> None: ...


def motion_command(env: ManagerBasedRlEnv) -> MotionCommandLike:
  """The env's ``"motion"`` command term."""
  return cast(MotionCommandLike, env.command_manager.get_term("motion"))


def motion_command_cfg(env_cfg: ManagerBasedRlEnvCfg) -> MotionCommandCfgLike:
  """The env cfg's ``"motion"`` command cfg: mjlab_research's or pal_mjlab's."""
  from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
  from pal_mjlab.tasks.WBC.mdp.commands import OnlineMotionCommandCfg

  assert env_cfg.commands is not None
  cfg = env_cfg.commands["motion"]
  if not isinstance(cfg, (MotionCommandCfg, OnlineMotionCommandCfg)):
    raise TypeError(f"unsupported motion command cfg {type(cfg).__name__}")
  return cfg


def set_motion_source(env_cfg: ManagerBasedRlEnvCfg, source: str | list[str]) -> None:
  """Point the motion command at local motions: npz files (one, a list, or a
  glob) for mjlab_research's command, one CSV folder for pal_mjlab's."""
  from pal_mjlab.tasks.WBC.mdp.commands import OnlineMotionCommandCfg

  cfg = motion_command_cfg(env_cfg)
  if isinstance(cfg, OnlineMotionCommandCfg):
    if isinstance(source, list):
      if len(source) != 1:
        raise ValueError(f"pal_mjlab's command reads one CSV folder, got {source}")
      source = source[0]
    cfg.motion_dir = source
  else:
    cfg.motion_file = source
