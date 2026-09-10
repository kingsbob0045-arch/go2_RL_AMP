"""Online convergence detection shared by the SECAMP/AMP runners.

Mirrors the offline thresholds in datasets/scripts/check_convergence.py so that a run
stops on exactly the criteria the report will later apply to it.
"""

from collections import deque

import numpy as np

BLOCK = 100                    # iterations per averaging block
MAX_EPISODE_STEPS = 1000       # episode_length_s 20.0 / step_dt 0.02
MAX_IMI_PER_STEP = 2.0 * 0.02  # amp_reward_coef * 1.0 * step_dt

EPISODE_LENGTH_MIN = 950.0
REWARD_PLATEAU_TOL = 0.02
STD_MAX = 3.0
# polyfit on a flat series returns ~1e-17 rather than exactly 0, and a genuinely stable
# run still drifts a little.  0.01/100it is 0.2 over a 2000-iteration run: negligible,
# while the failure we are guarding against ran at 2.2/100it.
STD_SLOPE_TOL = 0.01
DISC_ABS_MAX = 0.80
AMP_LOSS_MIN = 0.02
VALUE_LOSS_TOL = 0.05
TREND_WINDOW = 300


def _slope_per_100(values, window=TREND_WINDOW):
    tail = np.asarray(values[-window:], dtype=float)
    if len(tail) < 10:
        return 0.0
    return float(np.polyfit(np.arange(len(tail), dtype=float), tail, 1)[0] * 100.0)


def _blocks(values, count):
    values = np.asarray(values, dtype=float)
    return [float(values[-(i + 1) * BLOCK: len(values) - i * BLOCK].mean())
            for i in range(count) if len(values) >= (i + 1) * BLOCK]


class ConvergenceMonitor:
    """Track per-iteration scalars and report when the run satisfies every criterion."""

    def __init__(self, min_iterations=500, patience=100):
        self.min_iterations = min_iterations
        self.patience = patience
        self._history = {name: deque(maxlen=2000) for name in
                         ("reward", "episode_length", "noise_std", "amp_loss",
                          "d_expert", "d_policy", "value_loss")}
        self._converged_streak = 0
        self.last_report = {}

    def update(self, *, reward, episode_length, noise_std, amp_loss,
               d_expert, d_policy, value_loss):
        self._history["reward"].append(reward)
        self._history["episode_length"].append(episode_length)
        self._history["noise_std"].append(noise_std)
        self._history["amp_loss"].append(amp_loss)
        self._history["d_expert"].append(d_expert)
        self._history["d_policy"].append(d_policy)
        self._history["value_loss"].append(value_loss)

    def _checks(self):
        history = {name: list(values) for name, values in self._history.items()}
        if len(history["reward"]) < 3 * BLOCK:
            return None

        episode_length = float(np.mean(history["episode_length"][-BLOCK:]))
        reward_blocks = _blocks(history["reward"], 3)
        value_blocks = _blocks(history["value_loss"], 2)
        std_now = history["noise_std"][-1]
        amp_loss = float(np.mean(history["amp_loss"][-BLOCK:]))
        d_expert = float(np.mean(history["d_expert"][-BLOCK:]))
        d_policy = float(np.mean(history["d_policy"][-BLOCK:]))

        reward_deltas = [abs(reward_blocks[i] - reward_blocks[i + 1]) / max(abs(reward_blocks[i + 1]), 1e-6)
                         for i in range(len(reward_blocks) - 1)]
        return {
            "episode_length_high": episode_length >= EPISODE_LENGTH_MIN,
            "episode_length_flat": abs(_slope_per_100(history["episode_length"])) < 0.005 * MAX_EPISODE_STEPS,
            "reward_plateau": len(reward_deltas) >= 2 and all(d < REWARD_PLATEAU_TOL for d in reward_deltas),
            "std_not_growing": _slope_per_100(history["noise_std"]) <= STD_SLOPE_TOL and std_now < STD_MAX,
            "discriminator_balanced": (abs(d_expert) <= DISC_ABS_MAX and abs(d_policy) <= DISC_ABS_MAX
                                       and amp_loss > AMP_LOSS_MIN),
            "value_loss_stable": (len(value_blocks) >= 2 and
                                  abs(value_blocks[0] - value_blocks[1]) / max(abs(value_blocks[1]), 1e-6)
                                  < VALUE_LOSS_TOL),
        }

    def should_stop(self, iteration):
        """True once every criterion has held for `patience` consecutive iterations."""
        checks = self._checks()
        if checks is None:
            return False
        self.last_report = checks
        if iteration < self.min_iterations or not all(checks.values()):
            self._converged_streak = 0
            return False
        self._converged_streak += 1
        return self._converged_streak >= self.patience

    def summary(self):
        if not self.last_report:
            return "convergence monitor: not enough data yet"
        failing = [name for name, passed in self.last_report.items() if not passed]
        if not failing:
            return f"convergence monitor: all criteria met for {self._converged_streak} iterations"
        return f"convergence monitor: still failing {', '.join(failing)}"
