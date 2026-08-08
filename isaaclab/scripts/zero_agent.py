#!/usr/bin/env python3
"""Smoke-test a migrated environment with zero actions."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Go2-AMP-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import go2_amp_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.add_observation_noise = False
    cfg.reference_state_initialization = False
    cfg.events = None
    env = gym.make(args.task, cfg=cfg)
    observations, _ = env.reset()
    actions = torch.zeros(args.num_envs, cfg.action_space, device=args.device)
    for _ in range(args.steps):
        observations, rewards, terminated, truncated, extras = env.step(actions)
        assert observations["policy"].shape == (args.num_envs, cfg.observation_space)
        assert torch.isfinite(observations["policy"]).all()
        assert torch.isfinite(rewards).all()
    print(f"PASS: {args.task}, obs={tuple(observations['policy'].shape)}, steps={args.steps}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
