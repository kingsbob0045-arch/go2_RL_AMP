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
parser.add_argument(
    "--amp_dataset",
    choices=("baseline", "mocap_gaits", "kine2go_gaits", "nju_agility", "dogml_gaits"),
    default=None,
    help="SECAMP motion dataset. External datasets must be prepared first.",
)
parser.add_argument("--run_name", type=str, default=None, help="TensorBoard run label.")
parser.add_argument(
    "--disable_reference_state_initialization",
    action="store_true",
    help="Reset from the robot default state instead of sampled motion frames.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import random  # noqa: E402
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AMP_DATASETS = {
    "baseline": "datasets/camp/*",
    "mocap_gaits": "datasets/converted/mocap_gaits/*",
    "kine2go_gaits": "datasets/converted/kine2go_gaits/*.json",
    "nju_agility": "datasets/converted/nju_agility/*.json",
    "dogml_gaits": "datasets/converted/dogml_gaits/*.json",
}


def main():
    if args.task not in TASKS:
        raise ValueError(f"Unknown task {args.task}; choose one of {tuple(TASKS)}")
    cfg_factory, runner_type = TASKS[args.task]
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if args.disable_reference_state_initialization:
        env_cfg.reference_state_initialization = False
    train_cfg = cfg_factory()
    if args.amp_dataset is not None:
        if args.task != "Isaac-Go2-SECAMP-Direct-v0":
            raise ValueError("--amp_dataset is only supported by Isaac-Go2-SECAMP-Direct-v0")
        pattern = AMP_DATASETS[args.amp_dataset]
        motion_files = sorted(str(path) for path in PROJECT_ROOT.glob(pattern))
        if not motion_files:
            raise FileNotFoundError(
                f"No {args.amp_dataset} motions match {PROJECT_ROOT / pattern}. "
                "Run datasets/scripts/prepare_external_datasets.py first."
            )
        env_cfg.motion_glob = pattern
        train_cfg["runner"]["amp_motion_files"] = motion_files
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.max_iterations is not None:
        train_cfg["runner"]["max_iterations"] = args.max_iterations
    if args.amp_preload_transitions is not None and "amp_num_preload_transitions" in train_cfg["runner"]:
        train_cfg["runner"]["amp_num_preload_transitions"] = args.amp_preload_transitions
    random.seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])
    env = LegacyAmpVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    log_root = Path(__file__).resolve().parents[2] / "logs" / train_cfg["runner"]["experiment_name"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name is not None or args.amp_dataset is not None:
        label = args.run_name or f"{args.amp_dataset}_seed{train_cfg['seed']}"
        if "/" in label or "\\" in label or label in ("", ".", ".."):
            raise ValueError("--run_name must be a non-empty directory name")
        log_dir = log_root / "dataset_compare" / f"{label}_{timestamp}"
    else:
        log_dir = log_root / timestamp
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
