#!/usr/bin/env python3
"""Evaluate SECAMP convergence and divergence criteria from TensorBoard scalars.

Usage:
    python datasets/scripts/check_convergence.py                       # every dataset_compare run
    python datasets/scripts/check_convergence.py --glob 'dogml*conv*'  # a subset
    python datasets/scripts/check_convergence.py --json summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "logs" / "go2_secamp_isaaclab" / "dataset_compare"

MAX_EPISODE_STEPS = 1000  # episode_length_s 20.0 / step_dt 0.02
AMP_REWARD_COEF = 2.0
STEP_DT = 0.02
MAX_IMI_PER_STEP = AMP_REWARD_COEF * 1.0 * STEP_DT  # discriminator reward ceiling

BLOCK = 100          # iterations per averaging block
MIN_ITERATIONS = 300  # never declare convergence before this much new training
TREND_WINDOW = 300    # iterations used for slope tests

# Convergence thresholds
EPISODE_LENGTH_MIN = 950.0
REWARD_PLATEAU_TOL = 0.02
STD_MAX = 3.0
STD_SLOPE_TOL = 0.01  # see rsl_rl/utils/convergence.py; guards against polyfit noise
DISC_ABS_MAX = 0.80
AMP_LOSS_MIN = 0.02
VALUE_LOSS_TOL = 0.05

# Divergence thresholds
STD_DIVERGED = 8.0
IMI_FRACTION_MIN = 0.05
DISC_WON_ABS = 0.95
DISC_WON_AMP_LOSS = 0.01
EPISODE_LENGTH_COLLAPSE = 200.0


def _scalars(accumulator: EventAccumulator, tag: str) -> np.ndarray:
    if tag not in accumulator.Tags()["scalars"]:
        return np.array([])
    return np.array([event.value for event in accumulator.Scalars(tag)], dtype=float)


def _slope_per_100(values: np.ndarray, window: int) -> float:
    """Least-squares slope of the tail, expressed per 100 iterations."""
    tail = values[-window:]
    if len(tail) < 10:
        return 0.0
    x = np.arange(len(tail), dtype=float)
    return float(np.polyfit(x, tail, 1)[0] * 100.0)


def _block_means(values: np.ndarray, blocks: int = 3) -> list[float]:
    return [float(values[-(i + 1) * BLOCK: len(values) - i * BLOCK].mean())
            for i in range(blocks) if len(values) >= (i + 1) * BLOCK]


def evaluate(run_dir: Path) -> dict:
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0}).Reload()
    reward = _scalars(accumulator, "Train/mean_reward")
    if len(reward) == 0:
        return {"run": run_dir.name, "status": "no_data"}

    episode_length = _scalars(accumulator, "Train/mean_episode_length")
    imitation = _scalars(accumulator, "Train/mean_imi_reward")
    noise_std = _scalars(accumulator, "Policy/mean_noise_std")
    amp_loss = _scalars(accumulator, "Loss/AMP")
    d_expert = _scalars(accumulator, "Disc/d_expert")
    d_policy = _scalars(accumulator, "Disc/d_policy")
    value_loss = _scalars(accumulator, "Loss/value_function")

    first_step = accumulator.Scalars("Train/mean_reward")[0].step
    last_step = accumulator.Scalars("Train/mean_reward")[-1].step
    new_iterations = len(reward)

    episode_length_mean = float(episode_length[-BLOCK:].mean())
    imitation_mean = float(imitation[-BLOCK:].mean())
    imitation_fraction = imitation_mean / (MAX_IMI_PER_STEP * max(episode_length_mean, 1.0))
    std_now = float(noise_std[-1])
    std_slope = _slope_per_100(noise_std, TREND_WINDOW)
    episode_slope = _slope_per_100(episode_length, TREND_WINDOW)
    amp_loss_mean = float(amp_loss[-BLOCK:].mean())
    d_expert_mean = float(d_expert[-BLOCK:].mean())
    d_policy_mean = float(d_policy[-BLOCK:].mean())
    reward_blocks = _block_means(reward)
    value_blocks = _block_means(value_loss, blocks=2)

    reward_plateau = False
    if len(reward_blocks) >= 3:
        deltas = [abs(reward_blocks[i] - reward_blocks[i + 1]) / max(abs(reward_blocks[i + 1]), 1e-6)
                  for i in range(len(reward_blocks) - 1)]
        reward_plateau = all(delta < REWARD_PLATEAU_TOL for delta in deltas)
    value_plateau = (len(value_blocks) >= 2 and
                     abs(value_blocks[0] - value_blocks[1]) / max(abs(value_blocks[1]), 1e-6) < VALUE_LOSS_TOL)

    diverged = []
    if std_now > STD_DIVERGED:
        diverged.append(f"noise_std {std_now:.2f} > {STD_DIVERGED}")
    if imitation_fraction < IMI_FRACTION_MIN:
        diverged.append(f"imitation {imitation_fraction:.1%} < {IMI_FRACTION_MIN:.0%} of ceiling")
    if abs(d_expert_mean) >= DISC_WON_ABS and amp_loss_mean < DISC_WON_AMP_LOSS:
        diverged.append(f"discriminator won (d_expert {d_expert_mean:.2f}, AMP loss {amp_loss_mean:.4f})")
    if episode_length_mean < EPISODE_LENGTH_COLLAPSE:
        diverged.append(f"episode length {episode_length_mean:.0f} < {EPISODE_LENGTH_COLLAPSE:.0f}")
    if not np.isfinite(reward[-1]):
        diverged.append("reward is not finite")

    checks = {
        "episode_length_high": episode_length_mean >= EPISODE_LENGTH_MIN,
        "episode_length_flat": abs(episode_slope) < 0.005 * MAX_EPISODE_STEPS,
        "reward_plateau": reward_plateau,
        "std_not_growing": std_slope <= STD_SLOPE_TOL and std_now < STD_MAX,
        "discriminator_balanced": (abs(d_expert_mean) <= DISC_ABS_MAX
                                   and abs(d_policy_mean) <= DISC_ABS_MAX
                                   and amp_loss_mean > AMP_LOSS_MIN),
        "value_loss_stable": value_plateau,
    }
    enough_training = new_iterations >= MIN_ITERATIONS

    if diverged:
        status = "DIVERGED"
    elif enough_training and all(checks.values()):
        status = "CONVERGED"
    else:
        status = "training"

    return {
        "run": run_dir.name,
        "status": status,
        "iterations": f"{first_step}-{last_step}",
        "new_iterations": new_iterations,
        "metrics": {
            "episode_length": round(episode_length_mean, 1),
            "episode_length_slope_per_100": round(episode_slope, 2),
            "reward": round(float(reward[-BLOCK:].mean()), 3),
            "imitation_fraction_of_ceiling": round(imitation_fraction, 4),
            "noise_std": round(std_now, 3),
            "noise_std_slope_per_100": round(std_slope, 4),
            "d_expert": round(d_expert_mean, 3),
            "d_policy": round(d_policy_mean, 3),
            "amp_loss": round(amp_loss_mean, 4),
            "value_loss": round(float(value_loss[-BLOCK:].mean()), 5),
        },
        "checks": checks,
        "divergence_reasons": diverged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glob", default="*", help="Run-directory glob under dataset_compare/")
    parser.add_argument("--json", type=Path, default=None, help="Also write the report as JSON")
    args = parser.parse_args()

    runs = sorted(path for path in RUN_ROOT.glob(args.glob) if path.is_dir())
    if not runs:
        raise SystemExit(f"No runs match {RUN_ROOT / args.glob}")

    reports = [evaluate(run) for run in runs]
    for report in reports:
        if report["status"] == "no_data":
            print(f"{report['run']}: no scalar data yet")
            continue
        metrics = report["metrics"]
        print(f"\n{report['run']}")
        print(f"  status      : {report['status']}  (iterations {report['iterations']})")
        print(f"  ep_length   : {metrics['episode_length']:.1f}/1000 "
              f"(slope {metrics['episode_length_slope_per_100']:+.2f}/100it)")
        print(f"  reward      : {metrics['reward']:.3f}")
        print(f"  imitation   : {metrics['imitation_fraction_of_ceiling']:.1%} of ceiling")
        print(f"  noise_std   : {metrics['noise_std']:.3f} "
              f"(slope {metrics['noise_std_slope_per_100']:+.4f}/100it)")
        print(f"  disc        : d_expert {metrics['d_expert']:+.3f}  "
              f"d_policy {metrics['d_policy']:+.3f}  AMP loss {metrics['amp_loss']:.4f}")
        for name, passed in report["checks"].items():
            print(f"    [{'x' if passed else ' '}] {name}")
        for reason in report["divergence_reasons"]:
            print(f"    !! {reason}")

    if args.json:
        args.json.write_text(json.dumps(reports, indent=2) + "\n")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
