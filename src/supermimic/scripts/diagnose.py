"""Diagnose a trained run: does imagination see what the simulator does?

Rolls the checkpoint's policy in the play environment, then replays 32-step windows
of that record through the world model the way the policy learner does, and prints
where the two disagree. It trains nothing.

  uv run src/supermimic/scripts/diagnose.py SuperMimic-GenCoord-Kangaroo \
    --checkpoint-dir <run folder> --motion-file <packed motions> \
    --runner.actor-obs proprio --runner.history-length 10
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import mjlab
import torch
import tyro
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from tensordict import TensorDict

import supermimic.tasks  # noqa: F401
from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.models import build_mlp
from supermimic.representations import PROPRIO_NOISE_HALF_WIDTHS, set_goal_window
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.gencoord_helpers import decompose_rep_state

H = 32  # the policy learner's imagination horizon
WALK, STILL = 0.4, 0.15  # reference root speed [m/s] of a walking / a static window


@dataclass
class DiagnoseConfig:
  checkpoint_dir: str
  motion_file: str
  runner: SuperMimicRunnerCfg = field(default_factory=SuperMimicRunnerCfg)
  num_envs: int = 256
  steps: int = 1500
  windows: int = 512
  """Most windows replayed per group."""
  device: str | None = None


def row(name: str, *values: float) -> None:
  print(f"  {name:<28}" + "".join(f"{float(v):>11.4f}" for v in values))


@torch.no_grad()
def diagnose(task_id: str, cfg: DiagnoseConfig) -> None:
  device, rc = setup_device(cfg.device), cfg.runner
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.commands["motion"].motion_file = [os.path.expanduser(cfg.motion_file)]
  set_goal_window(rc.goal_frames, rc.goal_stride)
  env = create_env(env_cfg, cfg.num_envs, device)
  base = env.unwrapped

  wm = build_world_model(rc, env, device)
  rep, dyn, proprio = wm.representation, wm.dynamics, rc.actor_obs == "proprio"
  model_dir = Path(os.path.expanduser(cfg.checkpoint_dir))
  wm.load_bundle(torch.load(model_dir / "world_model.pt", map_location=device))
  actor = build_actor(rc, wm, device)
  actor.load_state_dict(torch.load(model_dir / "actor.pt", map_location=device))
  critic = build_mlp(
    rep.rep_state_dim + wm.num_goals + wm.sim_context.output_dim,
    rc.critic_hidden_dims,
    output_dim=1,
  ).to(device)
  critic.load_state_dict(torch.load(model_dir / "critic.pt", map_location=device))
  for module in (dyn, actor, critic):
    module.eval()

  cmd = base.command_manager.get_term("motion")
  motion, goal_of = cmd.motion, actor.task_specs[0].extract_training_goal
  sim_ctx = wm.sim_context.read(base)

  def sensor_of(obs: TensorDict, noisy: bool) -> torch.Tensor:
    return rep.compute_actor_sensor(
      rep.represent(obs["full_state"]),
      sim_ctx,
      sim_actor_obs=obs["actor"] if proprio else None,
      noisy=noisy and proprio,
    )

  # 1. Roll the policy in the simulator, as the runner collects.
  keys = ("state", "sensor", "action", "prev", "frame", "reward", "fell", "ended")
  rec: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
  obs = env.get_observations()
  history = sensor_of(obs, True).unsqueeze(1).repeat(1, rc.history_length, 1)
  for _ in range(cfg.steps):
    state, frame = obs["full_state"].clone(), cmd.time_steps.clone()
    goal = goal_of(motion, frame, rep.extract_aux_state(state))
    action = actor(build_actor_obs_td({"motion": goal}, history))
    step = (state, sensor_of(obs, False), action, base.action_manager.action.clone())
    obs, reward, dones, _ = env.step(action)
    fell = base.termination_manager.terminated.clone()
    for k, v in zip(
      keys, (*step, frame, reward.clone(), fell, dones.bool().clone()), strict=True
    ):
      rec[k].append(v)
    new = sensor_of(obs, True).unsqueeze(1)
    history = torch.where(
      dones.bool().view(-1, 1, 1), new, torch.cat([history[:, 1:], new], dim=1)
    )
  R = {k: torch.stack(v) for k, v in rec.items()}  # (T, N, ...)
  T, N = R["fell"].shape

  # Per recorded step: the episode's first step, the steps left in the clip, the
  # reference root's planar speed and forward velocity, and the realized return.
  first = torch.zeros(T, N, dtype=torch.long, device=device)
  for t in range(1, T):
    first[t] = torch.where(R["ended"][t - 1], t, first[t - 1])
  ends = motion.motion_ends
  clip = torch.searchsorted(ends, R["frame"], right=True).clamp(max=len(ends) - 1)
  left = ends[clip] - 1 - R["frame"]
  ref_vel = motion.root_lin_vel_w[R["frame"]]
  speed = ref_vel[..., :2].norm(dim=-1)
  forward = quat_apply_inverse(motion.root_quat_w[R["frame"]], ref_vel)[..., 0]
  gamma = getattr(rc.policy, "gamma", 0.99)
  ret = R["reward"].clone()
  for t in reversed(range(T - 1)):
    ret[t] += gamma * ret[t + 1] * ~R["ended"][t]

  rep_all = rep.represent(R["state"])
  _, _, _, _, qj_all, vj_all = decompose_rep_state(rep_all)
  aux_all = rep.extract_aux_state(R["state"])
  names = [base.scene["robot"].joint_names[i] for i in cmd.joint_ids.tolist()]
  legs = [i for i, n in enumerate(names) if n.startswith("leg_")]
  action_names = base.action_manager.get_term("joint_pos").target_names
  leg_actions = [action_names.index(names[i]) for i in legs]

  moving, still = speed > WALK, speed < STILL
  print(f"\n== Record: {T} steps x {N} envs, {int(R['ended'].sum())} episodes ended")
  print("  per-step fall rate [%] by steps left in the clip      walking    static")
  for lo, hi in ((0, 16), (16, 45), (45, 100), (100, 200), (200, 10**9)):
    near = (left >= lo) & (left < hi)
    rates = [100 * R["fell"][near & m].float().mean().item() for m in (moving, still)]
    row(f"{lo:>4} to {min(hi, 9999):<5}", *rates)
  ended_walk, ended_still = R["ended"] & moving, R["ended"] & still
  row("episodes ending in a fall [%]", *(
    100 * R["fell"][m].float().mean().item() for m in (ended_walk, ended_still)
  ))  # fmt: skip

  # 2. Windows of H steps inside one episode, by how they end and what the clip does.
  def windows(x: torch.Tensor) -> torch.Tensor:
    return x.float().unfold(0, H, 1)  # (T - H + 1, N, H)

  inside = windows(R["ended"])[..., :-1].sum(-1) == 0
  inside[: rc.history_length - 1] = False
  outcome = {
    "normal": ~R["ended"][H - 1 :],
    "pre-fall": R["fell"][H - 1 :],
    "pre-clip-end": R["ended"][H - 1 :] & ~R["fell"][H - 1 :],
  }
  w_speed, w_forward = windows(speed).mean(-1), windows(forward).mean(-1)
  kind = {
    "static": w_speed < STILL,
    "forward": (w_speed > WALK) & (w_forward > 0),
    "backward": (w_speed > WALK) & (w_forward < 0),
  }

  wm_env = WorldModelEnv(
    dynamics=dyn,
    representation=rep,
    env=env,
    batch_size=cfg.windows,
    replay_buffer=None,
    task_specs=[actor.task_specs[0]],
    history_length=rc.history_length,
    termination_softness=rc.termination_softness,
    device=device,
  )
  v_max = wm_env.max_step_reward / (1 - gamma)
  offsets = torch.arange(1 - rc.history_length, 1, device=device)

  def batch_at(t0: torch.Tensor, e: torch.Tensor) -> tuple[TensorDict, torch.Tensor]:
    """The learner's start batch for the windows starting at ``(t0, e)``."""
    ts = t0[:, None] + offsets
    start = first[t0, e][:, None]
    fields = {
      "state": R["state"][t0, e],
      "prev_action": R["prev"][t0, e],
      "sim_ctx": sim_ctx[e],
      "motion_time": R["frame"][t0, e],
      "sensor_window": R["sensor"][ts.maximum(start), e[:, None]],
      "pad_mask": ts < start,
    }
    return TensorDict(fields, batch_size=[len(t0)]), ts.maximum(start)

  def value(critic_obs: torch.Tensor) -> torch.Tensor:
    return critic(critic_obs).clamp(max=v_max).squeeze(-1)

  def value_at(t: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """The critic on recorded simulator states."""
    wm_goal, _ = rep.compute_wm_goal(motion, R["frame"][t, e], aux_all[t, e])
    return value(torch.cat([rep_all[t, e], wm_goal, sim_ctx[e]], dim=-1))

  def sample(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t, e = mask.nonzero(as_tuple=True)
    pick = torch.randperm(len(t), device=device)[: cfg.windows]
    return t[pick], e[pick]

  def imagine(batch: TensorDict, t0: torch.Tensor, e: torch.Tensor, closed: bool):
    """Roll the world model H steps on the recorded actions, or on the actor's."""
    wm_env.num_envs = len(t0)
    obs, alive, total, states = wm_env.reset(batch), 1.0, 0.0, []
    for k in range(H):
      action = actor(obs) if closed else R["action"][t0 + k, e]
      obs, reward, term_prob, _ = wm_env.step(action)
      total, alive = total + alive * reward, alive * (1 - term_prob)
      states.append((wm_env.rep_state, wm_env.aux_state))
    return alive, total, states

  for kind_name, kind_mask in kind.items():
    for outcome_name, outcome_mask in outcome.items():
      t0, e = (inside & kind_mask & outcome_mask).nonzero(as_tuple=True)
      if len(t0) < 16:
        continue
      pick = torch.randperm(len(t0), device=device)[: cfg.windows]
      t0, e = t0[pick], e[pick]
      batch, ts = batch_at(t0, e)
      real_reward = windows(R["reward"])[t0, e].sum(-1)
      print(f"\n== {kind_name} / {outcome_name}: {len(t0)} windows")

      # A. Open loop: the world model on the recorded actions against the simulator.
      alive, total, states = imagine(batch, t0, e, closed=False)
      print("  A. open loop: |error| after k steps          k=1        k=8       k=31")
      errors: dict[str, list[float]] = {}
      for k in (1, 8, 31):
        pred_rep, pred_aux = states[k - 1]
        true_rep, true_aux = rep_all[t0 + k, e], aux_all[t0 + k, e]
        z, g_b, _, _, qj, vj = decompose_rep_state(pred_rep - true_rep)
        channel = {
          "root height [m]": z.abs(),
          "gravity direction": g_b.norm(dim=-1),
          "root xy [m]": (pred_aux - true_aux)[:, :2].norm(dim=-1),
          "leg joint vel (rms)": vj[:, legs].square().mean(-1).sqrt(),
          **{names[i]: qj[:, i].abs() for i in legs},
        }
        for name, err in channel.items():
          errors.setdefault(name, []).append(err.mean().item())
      for name, values in errors.items():
        row(name, *values)
      print("                                            imagined   simulator")
      row(
        "survival over 32 steps", alive.mean().item(), float(outcome_name != "pre-fall")
      )
      row("reward sum", total.mean().item(), real_reward.mean().item())

      # B. Closed loop: what the learner sees, the actor on synthesized sensors.
      alive, total, _ = imagine(batch, t0, e, closed=True)
      print("  B. closed loop")
      row(
        "survival over 32 steps", alive.mean().item(), float(outcome_name != "pre-fall")
      )
      row("reward sum", total.mean().item(), real_reward.mean().item())

      # C. Critic at the window start against the return the simulator gave.
      print("  C. critic                                   value    realized")
      row("V(s) and return to episode end", value_at(t0, e).mean(), ret[t0, e].mean())
      row("steps left in the clip", left[t0, e].float().mean().item())

      if kind_name == "static":
        continue
      # E. Legs: how much the policy moves them, and how the world model says they
      # respond to one action from a true state.
      tk = (t0[:, None] + torch.arange(H - 1, device=device)).flatten()
      ek = e[:, None].expand(-1, H - 1).flatten()
      _, one_step = dyn(
        rep_all[tk, ek], R["action"][tk, ek], aux_all[tk, ek], sim_ctx[ek]
      )
      dv_pred = decompose_rep_state(one_step)[5] - vj_all[tk, ek]
      dv_real = vj_all[tk + 1, ek] - vj_all[tk, ek]
      acts = windows(R["action"])[t0, e]  # (B, A, H)
      vel = vj_all[tk, ek].view(len(t0), H - 1, -1)
      ref = motion.joint_vel[R["frame"][tk, ek]].view(len(t0), H - 1, -1)
      print("  E. legs                    action std  speed/ref  dv model/sim  dv corr")
      for i, a in zip(legs, leg_actions, strict=True):
        corr = torch.corrcoef(torch.stack([dv_pred[:, i], dv_real[:, i]]))[0, 1]
        row(
          names[i],
          acts[:, a].std(-1).mean().item(),
          (vel[..., i].square().mean() / ref[..., i].square().mean()).sqrt().item(),
          (dv_pred[:, i].square().mean() / dv_real[:, i].square().mean()).sqrt().item(),
          corr.item(),
        )

  # F. The steps before a fall in the simulator: when does anything in the objective
  # notice? The critic, and the termination rule's errors and probability.
  t_fall, e_fall = R["fell"].nonzero(as_tuple=True)
  anchor = wm_env._anchor_index
  print("\n== F. k steps before a simulator fall")
  print("                                 critic V  height err   tilt err  fall prob")
  for k in (31, 16, 8, 4, 2, 0):
    ok = t_fall - k >= first[t_fall, e_fall]
    t, e = t_fall[ok] - k, e_fall[ok]
    ref_pos, ref_quat, _, _ = motion.body_state(R["frame"][t, e])
    aux = aux_all[t, e]
    gravity_z = [
      2 * (q[:, 1:3] ** 2).sum(-1) - 1 for q in (ref_quat[:, anchor], aux[:, 3:7])
    ]
    row(
      f"k = {k:<2} ({len(t)} falls)",
      value_at(t, e).mean(),
      (ref_pos[:, anchor, 2] - aux[:, 2]).abs().mean(),
      (gravity_z[0] - gravity_z[1]).abs().mean(),
      wm_env._compute_term_prob(aux, ref_pos, ref_quat).mean(),
    )
  row("any recorded state", value_at(*sample(~R["ended"])).mean())

  # G. The imagined reward formula on true simulator transitions: the next state
  # against the reference frame before the step, and against the one after it.
  t, e = sample(~R["ended"][:-1])
  next_rep, next_aux = rep_all[t + 1, e], aux_all[t + 1, e]
  print("\n== G. reward formula on true transitions     formula   simulator  mean |d|")
  for shift in (0, 1):
    formula = rep.compute_wm_reward_from_reference(
      next_rep,
      next_aux,
      rep.joint_state(next_rep),
      R["action"][t, e],
      R["prev"][t, e],
      *motion.body_state(R["frame"][t, e] + shift),
    )
    sim = R["reward"][t, e]
    row(
      f"reference frame + {shift}",
      formula.mean(),
      sim.mean(),
      (formula - sim).abs().mean(),
    )

  # H. A long imagined rollout from normal states, clip ends held as in training:
  # what imagination itself says a state is worth, against the critic.
  long = 300
  print(f"\n== H. imagined closed loop, {long} steps")
  print(
    "                                 critic V   imagined    alive@32  alive@100  alive@300  r/max 0-32  r/max 100+"
  )
  for kind_name, kind_mask in kind.items():
    t0, e = sample(inside & kind_mask & outcome["normal"])
    wm_env.num_envs = len(t0)
    obs = wm_env.reset(batch_at(t0, e)[0])
    start_value, alive, total = value(obs["critic"]), 1.0, 0.0
    rewards, alives = [], []
    for k in range(long):
      obs, reward, term_prob, _ = wm_env.step(actor(obs))
      total, alive = total + gamma**k * alive * reward, alive * (1 - term_prob)
      rewards.append(reward.mean() / wm_env.max_step_reward)
      alives.append(alive.mean())
    total = total + gamma**long * alive * value(obs["critic"])
    row(
      kind_name,
      start_value.mean(),
      total.mean(),
      alives[31],
      alives[99],
      alives[-1],
      torch.stack(rewards[:32]).mean(),
      torch.stack(rewards[100:]).mean(),
    )

  if not proprio:
    return  # the actor reads the state itself
  # D. The sensor the simulator gave against the one imagination synthesizes from the
  # same state, and what that difference does to the action.
  synth = rep.compute_actor_sensor(
    rep_all,
    sim_ctx.expand(T, N, -1),
    aux_state=aux_all,
    joint_state=rep.joint_state(rep_all),
    last_action=R["prev"],
  )
  print("\n== D. simulated minus synthesized sensor      mean |d|     max |d|")
  diff = (R["sensor"] - synth).abs()
  widths = rep.proprio_term_widths(wm.num_actions)
  for name, part in zip(
    PROPRIO_NOISE_HALF_WIDTHS, diff.split(widths, dim=-1), strict=True
  ):
    row(name, part.mean().item(), part.max().item())
  t0, e = sample(inside & outcome["normal"])
  _, ts = batch_at(t0, e)
  goal = goal_of(motion, R["frame"][t0, e], aux_all[t0, e])
  on_real = actor(build_actor_obs_td({"motion": goal}, R["sensor"][ts, e[:, None]]))
  on_synth = actor(build_actor_obs_td({"motion": goal}, synth[ts, e[:, None]]))
  gap = (on_real - on_synth).abs()
  print("  action on real vs synthesized history     |gap| legs  |action| legs")
  row(
    "no sensor noise on either",
    gap[:, leg_actions].mean().item(),
    on_real[:, leg_actions].abs().mean().item(),
  )


def main() -> None:
  task, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  diagnose(task, tyro.cli(DiagnoseConfig, args=remaining, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
