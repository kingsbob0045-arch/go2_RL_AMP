#!/usr/bin/env python3
"""Train a migrated Go2 task in Isaac Lab using the bundled AMP runners."""

import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Go2-AMP-Direct-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--amp_preload_transitions", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import go2_amp_isaaclab  # noqa: E402, F401
from go2_amp_isaaclab.tasks.go2_amp.runner_cfg import (  # noqa: E402
    go2_amp_runner_cfg, go2_rough_runner_cfg, go2_secamp_runner_cfg,
)
from go2_amp_isaaclab.wrappers import LegacyAmpVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import AMPOnPolicyRunner, Go2RoughRunner, Go2SECAMPRunner  # noqa: E402

TASKS = {
    "Isaac-Go2-AMP-Direct-v0": (go2_amp_runner_cfg, AMPOnPolicyRunner),
    "Isaac-Go2-SECAMP-Direct-v0": (go2_secamp_runner_cfg, Go2SECAMPRunner),
    "Isaac-Go2-Rough-Residual-Direct-v0": (go2_rough_runner_cfg, Go2RoughRunner),
}


def main():
    if args.task not in TASKS:
        raise ValueError(f"Unknown task {args.task}; choose one of {tuple(TASKS)}")
    cfg_factory, runner_type = TASKS[args.task]
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    train_cfg = cfg_factory()
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.max_iterations is not None:
        train_cfg["runner"]["max_iterations"] = args.max_iterations
    if args.amp_preload_transitions is not None and "amp_num_preload_transitions" in train_cfg["runner"]:
        train_cfg["runner"]["amp_num_preload_transitions"] = args.amp_preload_transitions
    torch.manual_seed(train_cfg["seed"])
    env = LegacyAmpVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    log_root = Path(__file__).resolve().parents[2] / "logs" / train_cfg["runner"]["experiment_name"]
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = runner_type(env, train_cfg, log_dir=str(log_dir), device=args.device)
    if args.checkpoint:
        runner.load(args.checkpoint)
    runner.learn(train_cfg["runner"]["max_iterations"], init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
