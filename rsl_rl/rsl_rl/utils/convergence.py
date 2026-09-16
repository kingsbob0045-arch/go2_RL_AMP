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
# A saturation ceiling alone lets the opposite failure through unnoticed: a discriminator
# that has collapsed towards d = 0 for every input scores |d_expert| = 0.0 and passes.  That
# is not convergence, it is the discriminator giving up -- and because the AMP reward is
# coef * clamp(1 - (d-1)^2/4, 0) * dt, d = 0 still pays 75% of the maximum imitation reward
# for any motion at all.  Runs have been read as improving on exactly this basis: imi_reward
# rising from 9.8 to 12.9 while d_expert fell from 0.765 to 0.629 and Loss/AMP rose.  So the
# band is two-sided, and the expert/policy separation is checked as well, because d_expert
# can sit inside the band while the discriminator separates nothing.
# Set just below the controller's band low edge (0.70), so the criterion reads as "the
# discriminator balance controller is still holding its setpoint".  At a looser 0.60 the
# dogml v3 run passed with d_expert 0.634 and was declared converged -- the exact run whose
# d_expert had fallen from 0.765 to 0.629 while its imitation reward rose from 9.8 to 12.9.
# Separation 1.0 is likewise well inside what a healthy run shows (v2 and v3 both ran at
# 1.27-1.42) while still ruling out a collapse towards 0.
DISC_EXPERT_MIN = 0.65
DISC_SEPARATION_MIN = 1.00
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
        self.last_values = {}

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
        self.last_values = {"d_expert": d_expert, "d_policy": d_policy, "amp_loss": amp_loss,
                            "episode_length": episode_length, "noise_std": std_now}
        return {
            "episode_length_high": episode_length >= EPISODE_LENGTH_MIN,
            "episode_length_flat": abs(_slope_per_100(history["episode_length"])) < 0.005 * MAX_EPISODE_STEPS,
            "reward_plateau": len(reward_deltas) >= 2 and all(d < REWARD_PLATEAU_TOL for d in reward_deltas),
            "std_not_growing": _slope_per_100(history["noise_std"]) <= STD_SLOPE_TOL and std_now < STD_MAX,
            "discriminator_not_saturated": (abs(d_expert) <= DISC_ABS_MAX
                                            and abs(d_policy) <= DISC_ABS_MAX
                                            and amp_loss > AMP_LOSS_MIN),
            "discriminator_informative": (d_expert >= DISC_EXPERT_MIN
                                          and (d_expert - d_policy) >= DISC_SEPARATION_MIN),
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
        # The discriminator numbers are printed either way.  They are the criterion most often
        # misread from the reward curves alone, so the report states them rather than only
        # saying pass or fail.
        values = self.last_values
        disc = (f"d_expert {values['d_expert']:+.3f}, d_policy {values['d_policy']:+.3f}, "
                f"separation {values['d_expert'] - values['d_policy']:.3f}, "
                f"AMP loss {values['amp_loss']:.4f}") if values else ""
        failing = [name for name, passed in self.last_report.items() if not passed]
        if not failing:
            return (f"convergence monitor: all criteria met for {self._converged_streak} "
                    f"iterations ({disc})")
        return f"convergence monitor: still failing {', '.join(failing)} ({disc})"
