"""Tests for the upper-body task: pal_mjlab's WBC command read from the reference."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from supermimic.motion_api import motion_command
from supermimic.tasks.task_spec import (
  UpperBodyTaskCfg,
  make_upper_body_task,
  reference_com_feet_distance,
)
from tests.conftest import fake_env, identity_aux

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_XML = """
<mujoco>
  <worldbody>
    <body name="base_link" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" mass="5"/>
      <body name="torso" pos="0 0 .2">
        <joint name="pelvis_1_joint" axis="0 0 1"/>
        <geom type="box" size=".1 .1 .2" pos=".05 0 .1" mass="8"/>
        <body name="arm" pos="0 .2 .2">
          <joint name="arm_left_1_joint" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 .3 0 0" size=".03" mass="2"/>
        </body>
      </body>
      <body name="leg_left_3_link" pos="0 .1 -.1">
        <joint name="leg_left_1_joint" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".04" mass="3"/>
        <body name="leg_left_5_link" pos="0 0 -.4">
          <joint name="leg_left_length_joint" type="slide" axis="0 0 1"/>
          <geom type="box" size=".1 .05 .02" mass="1"/>
        </body>
      </body>
      <body name="leg_right_3_link" pos="0 -.1 -.1">
        <joint name="leg_right_1_joint" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".04" mass="3"/>
        <body name="leg_right_5_link" pos="0 0 -.4">
          <joint name="leg_right_length_joint" type="slide" axis="0 0 1"/>
          <geom type="box" size=".1 .05 .02" mass="1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
_NUM_FRAMES = 6
_NUM_UPPER = 2  # pelvis_1_joint, arm_left_1_joint
_FRAME_DIM = 11 + 2 * _NUM_UPPER


def _make_env(seed: int = 0) -> ManagerBasedRlEnv:
  """A two-legged robot and one random clip of it, as the task reads them."""
  model = mujoco.MjModel.from_xml_string(_XML)
  gen = torch.Generator().manual_seed(seed)
  joint_names = [model.joint(j).name for j in range(1, model.njnt)]
  quat = torch.randn(_NUM_FRAMES, 4, generator=gen)
  motion = SimpleNamespace(
    root_pos_w=torch.randn(_NUM_FRAMES, 3, generator=gen),
    root_quat_w=quat / quat.norm(dim=-1, keepdim=True),
    root_lin_vel_w=torch.randn(_NUM_FRAMES, 3, generator=gen),
    root_ang_vel_w=torch.randn(_NUM_FRAMES, 3, generator=gen),
    joint_pos=0.3 * torch.randn(_NUM_FRAMES, len(joint_names), generator=gen),
    joint_vel=torch.randn(_NUM_FRAMES, len(joint_names), generator=gen),
    time_step_total=_NUM_FRAMES,
    motion_ends=torch.tensor([_NUM_FRAMES]),
    fk=SimpleNamespace(joint_names=joint_names),
  )
  robot = SimpleNamespace(
    indexing=SimpleNamespace(
      body_ids=torch.arange(1, model.nbody),
      joint_q_adr=torch.as_tensor(model.jnt_qposadr[1:]),
    ),
    body_names=[model.body(b).name for b in range(1, model.nbody)],
  )
  motion_cmd = SimpleNamespace(
    robot=robot, motion=motion, full_joint_state=lambda pos, vel: (pos, vel)
  )
  return fake_env(
    command_manager=SimpleNamespace(get_term=lambda name: motion_cmd),
    sim=SimpleNamespace(mj_model=model),
  )


def test_com_feet_distance_matches_mujoco():
  env = _make_env()
  motion_cmd, model = motion_command(env), env.sim.mj_model
  feet = ("leg_left_5_link", "leg_right_5_link")

  distance = reference_com_feet_distance(motion_cmd, model, feet, chunk=4)

  data = mujoco.MjData(model)
  motion = motion_cmd.motion
  for t in range(_NUM_FRAMES):
    data.qpos[:] = np.concatenate(
      [motion.root_pos_w[t], motion.root_quat_w[t], motion.joint_pos[t]]
    )
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    com = data.subtree_com[model.body("base_link").id]
    midpoint = np.mean([data.body(name).xpos for name in feet], axis=0)
    expected = np.linalg.norm(com[:2] - midpoint[:2])
    assert math.isclose(distance[t].item(), expected, abs_tol=1e-5)


def test_command_holds_the_wbc_terms_in_order():
  env = _make_env()
  motion = motion_command(env).motion
  # Frame 2: the reference faces +y, moves along world +x and leans nowhere.
  yaw = math.pi / 2
  motion.root_quat_w[2] = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  motion.root_lin_vel_w[2] = torch.tensor([1.0, 0.0, 0.2])
  motion.root_ang_vel_w[2] = torch.tensor([0.0, 0.0, 0.8])
  task = make_upper_body_task(UpperBodyTaskCfg(), env)

  goal = task.extract_training_goal(motion, torch.tensor([2]), identity_aux(1))[0]

  assert task.name == "upper_body"
  assert task.goal_dim == goal.numel() == _FRAME_DIM
  assert torch.isclose(goal[0], motion.root_pos_w[2, 2])
  assert torch.allclose(goal[1:4], torch.tensor([0.0, -1.0, 0.2]), atol=1e-6)
  assert torch.allclose(goal[4:7], torch.tensor([0.0, 0.0, 0.8]), atol=1e-6)
  assert torch.allclose(goal[7:10], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
  # pelvis_1_joint and arm_left_1_joint; no leg joint.
  assert torch.equal(goal[10:12], motion.joint_pos[2, :2])
  assert torch.equal(goal[12:14], motion.joint_vel[2, :2])
  feet = UpperBodyTaskCfg().feet_body_names
  distance = reference_com_feet_distance(motion_command(env), env.sim.mj_model, feet)
  assert torch.isclose(goal[14], distance[2])


def test_command_does_not_depend_on_the_robot():
  env = _make_env()
  motion = motion_command(env).motion
  task = make_upper_body_task(UpperBodyTaskCfg(), env)
  times = torch.tensor([1, 4])
  turned = identity_aux(2)
  turned[:, :3] = torch.tensor([3.0, -2.0, 0.5])
  turned[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])

  goal = task.extract_training_goal(motion, times, identity_aux(2))

  assert torch.equal(goal, task.extract_training_goal(motion, times, turned))


def test_goal_frames_and_stride_widen_the_command():
  env = _make_env()
  motion = motion_command(env).motion
  single = make_upper_body_task(UpperBodyTaskCfg(), env)
  window = make_upper_body_task(UpperBodyTaskCfg(goal_frames=3, goal_stride=2), env)

  goal = window.extract_training_goal(motion, torch.tensor([3]), identity_aux(1))

  # Frames 3, 5 and 7, the last one held at the clip's final frame, 5.
  assert window.goal_dim == goal.shape[1] == 3 * _FRAME_DIM
  frames = single.extract_training_goal(
    motion, torch.tensor([3, 5, 5]), identity_aux(3)
  )
  assert torch.equal(goal[0], frames.flatten())
  assert window.precompute_replay_goal is not None
  assert torch.equal(window.precompute_replay_goal(motion)[3], goal[0])
