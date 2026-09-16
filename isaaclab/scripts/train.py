#!/usr/bin/env python3
"""Train a migrated Go2 task in Isaac Lab using the bundled AMP runners."""

import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Go2-AMP-Direct-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument(
    "--max_iterations", type=int, default=None,
    help="Absolute iteration to stop at.  When resuming with --checkpoint this counts from "
         "the checkpoint's own iteration, so it is a target rather than an extra amount.",
)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--entropy_coef", type=float, default=None,
    help="PPO entropy bonus weight.  The policy std is a raw parameter, so the entropy "
         "gradient w.r.t. it is 1/std and Adam turns that into a near-constant increase of "
         "roughly one learning rate per update.  Once the task reward saturates nothing "
         "opposes it, so long runs need a smaller value than the 0.01 used for short ones.",
)
parser.add_argument("--amp_preload_transitions", type=int, default=None)
parser.add_argument("--action_rate_scale", type=float, default=None)
parser.add_argument(
    "--disc_target_expert", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
    help="Band the discriminator controller holds Disc/d_expert inside.  Pass the same value "
         "twice to pin it, or use --no_disc_adapt to restore the fixed learning rate.",
)
parser.add_argument("--no_disc_adapt", action="store_true",
                    help="Disable the adaptive discriminator controller.")
parser.add_argument("--no_skill_balance", action="store_true",
                    help="Weight the expert preload buffer by clip count instead of giving "
                         "each skill an equal share.")
parser.add_argument(
    "--amp_dataset",
    choices=("baseline", "mocap_gaits", "kine2go_gaits", "nju_agility", "dogml_gaits"),
    default=None,
    help="SECAMP motion dataset. External datasets must be prepared first.",
)
parser.add_argument("--run_name", type=str, default=None, help="TensorBoard run label.")
parser.add_argument(
    "--stage", type=str, default=None, choices=("core", "latency", "dynamics", "terrain"),
    help="Sim-to-real difficulty stage.  The config ships with every group enabled, so this "
         "switches OFF everything above the named stage: 'core' is flat ground with no "
         "latency and no extra randomisation, then latency adds control delay and wider PD "
         "gains, dynamics adds per-link mass/COM, joint friction/armature and IMU bias, and "
         "terrain adds the random grid.  Omit to run everything at once -- which is what "
         "produced a 20-hour run that could not say which change broke it.",
)
parser.add_argument(
    "--early_stop",
    action="store_true",
    help="Stop once every convergence criterion in rsl_rl.utils.convergence holds for 100 "
         "consecutive iterations, and save model_converged.pt.",
)
parser.add_argument(
    "--early_stop_min_iterations", type=int, default=500,
    help="Never declare convergence before this iteration.",
)
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


STAGES = ("core", "latency", "dynamics", "terrain")


def apply_stage(env_cfg, stage: str) -> None:
    """Disable every difficulty group above `stage`.

    Subtractive on purpose: the config carries the full sim-to-real setup, and a stage is a
    prefix of it.  That keeps one source of truth for what "full difficulty" means, instead of
    a second copy of every range living in here and drifting from it.
    """
    level = STAGES.index(stage)
    events = env_cfg.events
    if level < STAGES.index("terrain"):
        # terrain_type="plane" makes TerrainImporter ignore the generator; go2_env's
        # _setup_scene already special-cases it to set env_spacing.
        env_cfg.terrain.terrain_type = "plane"
        env_cfg.terrain.terrain_generator = None
    if level < STAGES.index("dynamics"):
        events.link_mass = None
        events.link_com = None
        events.joint_parameters = None
        env_cfg.gravity_bias = 0.0
    if level < STAGES.index("latency"):
        env_cfg.action_latency_steps = (0, 0)
        env_cfg.observation_latency_steps = (0, 0)
        gains = events.actuator_gains.params
        gains["stiffness_distribution_params"] = (0.9, 1.1)
        gains["damping_distribution_params"] = (0.9, 1.1)
    print(f"Difficulty stage '{stage}': "
          f"terrain={env_cfg.terrain.terrain_type}, "
          f"action_latency={env_cfg.action_latency_steps}, "
          f"gravity_bias={env_cfg.gravity_bias}, "
          f"link_mass={'on' if events.link_mass else 'off'}", flush=True)


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
    if args.entropy_coef is not None:
        train_cfg["algorithm"]["entropy_coef"] = args.entropy_coef
    if args.amp_preload_transitions is not None and "amp_num_preload_transitions" in train_cfg["runner"]:
        train_cfg["runner"]["amp_num_preload_transitions"] = args.amp_preload_transitions
    if args.action_rate_scale is not None:
        env_cfg.action_rate_scale = args.action_rate_scale
    if args.stage is not None:
        apply_stage(env_cfg, args.stage)
    if args.no_disc_adapt:
        train_cfg["algorithm"]["disc_target_expert"] = None
    elif args.disc_target_expert is not None:
        train_cfg["algorithm"]["disc_target_expert"] = tuple(args.disc_target_expert)
    train_cfg["runner"]["amp_balance_skills"] = not args.no_skill_balance
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
    if args.early_stop:
        if not hasattr(runner, "_convergence"):
            raise ValueError(f"--early_stop is not supported by {runner_type.__name__}")
        from rsl_rl.utils.convergence import ConvergenceMonitor
        runner._convergence = ConvergenceMonitor(min_iterations=args.early_stop_min_iterations)
        print(f"Early stopping enabled (minimum {args.early_stop_min_iterations} iterations)", flush=True)
    if args.checkpoint:
        runner.load(args.checkpoint)
    # runner.learn() counts iterations relative to the checkpoint it resumed from, so pass
    # the remaining amount to make --max_iterations the absolute stopping point.
    total_iterations = train_cfg["runner"]["max_iterations"]
    remaining = total_iterations - runner.current_learning_iteration
    if remaining <= 0:
        raise ValueError(
            f"--max_iterations {total_iterations} is not beyond the resumed checkpoint's "
            f"iteration {runner.current_learning_iteration}")
    if runner.current_learning_iteration:
        print(f"Resuming at iteration {runner.current_learning_iteration}; "
              f"running {remaining} more to reach {total_iterations}", flush=True)
    runner.learn(remaining, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
