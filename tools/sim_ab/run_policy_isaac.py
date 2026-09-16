#!/usr/bin/env python3
"""Close the loop on the same exported policy inside Isaac Lab, for a like-for-like check.

Two things are measured at once:

1. Does the policy walk here?  Same command, same skill, same metrics as
   run_policy_mujoco.py, so the two engines can be compared on the policy's behaviour
   rather than on an open-loop trace.

2. Does the observation the deploy side builds by hand match the one the environment
   builds internally?  Every step, the environment's own policy observation is compared
   against one assembled from the robot state with the deploy formulas.  A mismatch here
   would invalidate any MuJoCo result, so it is checked rather than assumed.

Run under env_isaacsim:
    python tools/sim_ab/run_policy_isaac.py --headless --policy exports/go2_secamp.pt --skill 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Go2-SECAMP-Direct-v0")
parser.add_argument("--policy", default=str(
    Path.home() / "Repo/ProjectsTest_Lingheng_Kong/go2_RL_AMP/exports/go2_secamp.pt"))
parser.add_argument("--skill", type=int, default=1)
parser.add_argument("--command", type=float, nargs=3, default=[1.0, 0.0, 0.0])
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--latency", type=int, default=0,
                    help="Whole control periods of action delay, matching run_policy_mujoco.py")
parser.add_argument("--flat", action="store_true",
                    help="Force a flat plane, so the comparison is not confounded by terrain")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _local_source  # noqa: E402, F401 - must precede go2_amp_isaaclab
import protocol as P  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import go2_amp_isaaclab  # noqa: E402, F401
from go2_amp_isaaclab.wrappers import LegacyAmpVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_local_source.report()

SKILL_NAMES = ("pace", "trot", "canter")


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_wxyz
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ])


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.add_observation_noise = False
    cfg.reference_state_initialization = False
    cfg.events = None
    cfg.waypoint_mode = False
    # This script exists to compare against MuJoCo, which has no latency and no IMU bias, and
    # to check that the deploy-side observation formulas still reproduce the environment's.
    # Both comparisons are only meaningful against the nominal plant, so the sim-to-real
    # randomisation is switched off here and opted back in with --latency.
    cfg.action_latency_steps = (args.latency, args.latency)
    cfg.observation_latency_steps = (0, 0)
    cfg.gravity_bias = 0.0
    if args.flat:
        cfg.terrain = cfg.terrain.replace(terrain_type="plane", terrain_generator=None)
    cfg.command_resampling_time_s = 1.0e6
    cfg.episode_length_s = 1.0e6
    cfg.termination_contact_force = 1.0e9

    env = LegacyAmpVecEnvWrapper(gym.make(args.task, cfg=cfg))
    inner = env.env
    robot = inner.robot
    device = inner.device
    policy = torch.jit.load(args.policy, map_location="cpu").eval()

    obs, _ = env.reset()
    command = torch.tensor(args.command, dtype=torch.float32, device=device)
    inner.commands[:] = command
    inner.skill_commands.zero_()
    inner.skill_commands[:, args.skill] = 1.0

    default = inner.default_joint_pos[0].detach().cpu().numpy().astype(np.float64)
    ids = inner._joint_ids
    scale = np.asarray(cfg.command_scale, dtype=np.float64)
    n = int(args.seconds / P.CONTROL_DT)

    log = {k: np.zeros((n, d)) for k, d in
           (("q", 12), ("base_pos", 3), ("base_quat", 4), ("base_vel", 3), ("action", 12))}
    obs_gap = np.zeros(n)
    last_action = np.zeros(12, dtype=np.float32)
    fell_at = None
    origin = inner.terrain.env_origins[0].detach().cpu().numpy()

    for i in range(n):
        inner.commands[:] = command
        inner.skill_commands.zero_()
        inner.skill_commands[:, args.skill] = 1.0

        # The environment's own observation, which is what training used.
        env_obs = obs[0].detach().cpu().numpy().astype(np.float64)

        # The same vector rebuilt with the deploy-side formulas, from the raw robot state.
        q = robot.data.joint_pos[0, ids].detach().cpu().numpy().astype(np.float64)
        dq = robot.data.joint_vel[0, ids].detach().cpu().numpy().astype(np.float64)
        quat = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
        skill_vec = inner.skill_commands[0].detach().cpu().numpy().astype(np.float64)
        hand = np.concatenate((
            projected_gravity(quat),
            np.asarray(args.command, dtype=np.float64) * scale,
            (q - default) * float(cfg.dof_pos_scale),
            dq * float(cfg.dof_vel_scale),
            last_action,
            skill_vec,
        ))
        obs_gap[i] = float(np.abs(hand - env_obs).max()) if hand.shape == env_obs.shape else np.nan

        with torch.inference_mode():
            action = policy(obs.cpu()).squeeze(0).numpy()
        action = np.clip(action[:12], -float(cfg.action_clip), float(cfg.action_clip))
        last_action = action.astype(np.float32)
        obs, _, _, _, _, _, _ = env.step(
            torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0))

        pos = robot.data.root_pos_w[0].detach().cpu().numpy() - origin
        log["q"][i] = robot.data.joint_pos[0, ids].detach().cpu().numpy()
        log["base_pos"][i] = pos
        log["base_quat"][i] = robot.data.root_quat_w[0].detach().cpu().numpy()
        log["base_vel"][i] = robot.data.root_lin_vel_w[0].detach().cpu().numpy()
        log["action"][i] = action
        if fell_at is None and (pos[2] < 0.15
                                or projected_gravity(log["base_quat"][i])[2] > -0.5):
            fell_at = i * P.CONTROL_DT

    travelled = log["base_pos"][-1, :2] - log["base_pos"][0, :2]
    print(f"\nISAAC   skill {SKILL_NAMES[args.skill]}  command {args.command}")
    print(f"  fell            : {'no' if fell_at is None else f'yes at {fell_at:.2f} s'}")
    print(f"  travelled       : {travelled[0]:+.3f} m forward, {travelled[1]:+.3f} m lateral")
    print(f"  mean speed      : {np.linalg.norm(log['base_vel'][:, :2], axis=1).mean():.3f} m/s")
    print(f"  base height     : mean {log['base_pos'][:,2].mean():.3f} m, "
          f"min {log['base_pos'][:,2].min():.3f} m")
    print(f"  |action| mean   : {np.abs(log['action']).mean():.3f}, "
          f"max {np.abs(log['action']).max():.3f}")
    # Step 0 is excluded: the environment carries last_action from before the reset while the
    # hand-built vector starts it at zero, so the two legitimately disagree on the first frame
    # only.  Every later step is a real comparison.
    steady = obs_gap[1:]
    print(f"  hand-built vs env observation: max abs gap {np.nanmax(steady):.6f} "
          f"(mean {np.nanmean(steady):.6f}), step 0 excluded")
    if np.nanmax(steady) > 1.0e-4:
        worst = int(np.nanargmax(steady)) + 1
        print(f"    -> the deploy-side observation does NOT reproduce the training one; "
              f"worst at step {worst}.  Any MuJoCo result is invalid until this is zero.")
    else:
        print("    -> deploy-side observation formulas reproduce the training observation")

    if args.out:
        np.savez(args.out, obs_gap=obs_gap, **log)
        print(f"  wrote {args.out}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # simulation_app.close() tears the process down and reports success, which hides
        # whatever went wrong in main().  Print it before that happens.
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
