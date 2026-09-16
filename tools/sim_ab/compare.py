#!/usr/bin/env python3
"""Compare two A/B traces and report where the engines disagree.

The phases are diagnostic, not decorative:
  settle     static equilibrium -- gravity sag, contact stiffness, steady-state error
  chirp      FL thigh swinging in free air -- rotor inertia (armature) and joint damping,
             the only phase where they are not masked by ground reaction
  squat      loaded cycling -- torque limits and contact
  calf_step  square waves -- contact stiffness and damping

Run under go2_ros_env:
    python tools/sim_ab/compare.py /tmp/ab_isaac.npz /tmp/ab_mujoco.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P  # noqa: E402

FREE_JOINT = P.FL_THIGH  # the joint the chirp drives


def tracking_error(trace: dict, joint: int, mask: np.ndarray) -> float:
    return float(np.abs(trace["q"][mask, joint] - trace["target"][mask, joint]).mean())


def transfer(trace: dict, joint: int, mask: np.ndarray) -> tuple[float, float]:
    """Amplitude ratio and lag of the measured joint relative to its command."""
    cmd = trace["target"][mask, joint] - trace["target"][mask, joint].mean()
    out = trace["q"][mask, joint] - trace["q"][mask, joint].mean()
    gain = float(np.std(out) / max(np.std(cmd), 1.0e-9))
    corr = np.correlate(out, cmd, mode="full")
    lag = int(np.argmax(corr) - (len(cmd) - 1)) * P.CONTROL_DT
    return gain, lag


def band_gains(trace: dict, joint: int, mask: np.ndarray, bands=((0.5, 2.0), (2.0, 4.0), (4.0, 6.0))):
    """Gain per frequency band across the sweep, by slicing the chirp in time."""
    t = trace["t"][mask]
    span = t[-1] - t[0]
    f0, f1 = 0.5, 6.0
    out = []
    for lo, hi in bands:
        a = t[0] + span * (lo - f0) / (f1 - f0)
        b = t[0] + span * (hi - f0) / (f1 - f0)
        sub = (t >= a) & (t < b)
        idx = np.where(mask)[0][sub]
        cmd = trace["target"][idx, joint]
        pos = trace["q"][idx, joint]
        out.append(float(np.std(pos - pos.mean()) / max(np.std(cmd - cmd.mean()), 1.0e-9)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("isaac")
    ap.add_argument("mujoco")
    ap.add_argument("--plot", default=None, help="Write a PNG comparison here")
    args = ap.parse_args()

    A = {k: v for k, v in np.load(args.isaac).items()}
    B = {k: v for k, v in np.load(args.mujoco).items()}
    if not np.allclose(A["target"], B["target"]):
        raise SystemExit("The two traces did not run the same protocol.")
    t = A["t"]

    print("=" * 78)
    print(f"{'phase':<12}{'joint RMS diff':>16}{'worst joint':>22}{'base height diff':>20}")
    print("=" * 78)
    for name, start, end in P.PHASES:
        m = P.phase_mask(t, name)
        d = A["q"][m] - B["q"][m]
        rms = float(np.sqrt((d**2).mean()))
        worst = int(np.argmax(np.sqrt((d**2).mean(0))))
        worst_rms = float(np.sqrt((d[:, worst] ** 2).mean()))
        dz = float(np.abs(A["base_pos"][m, 2] - B["base_pos"][m, 2]).mean())
        print(f"{name:<12}{rms:>16.4f}{P.JOINT_NAMES[worst] + f' ({worst_rms:.3f})':>22}{dz:>20.4f}")

    print("\n--- static equilibrium (end of settle) ---")
    m = P.phase_mask(t, "settle")
    idx = np.where(m)[0][-1]
    print(f"  base height      isaac {A['base_pos'][idx,2]:.4f}   mujoco {B['base_pos'][idx,2]:.4f}"
          f"   diff {A['base_pos'][idx,2]-B['base_pos'][idx,2]:+.4f} m")
    print(f"  total foot force isaac {A['contact'][idx].sum():8.1f} N  mujoco {B['contact'][idx].sum():8.1f} N")
    sag_a = A["q"][idx] - P.DEFAULT_JOINT_POS
    sag_b = B["q"][idx] - P.DEFAULT_JOINT_POS
    print(f"  gravity sag (q - default), max |diff| {np.abs(sag_a-sag_b).max():.4f} rad "
          f"on {P.JOINT_NAMES[int(np.argmax(np.abs(sag_a-sag_b)))]}")

    print("\n--- free-leg chirp on FL_thigh (armature / damping signature) ---")
    m = P.phase_mask(t, "chirp")
    for label, tr in (("isaac", A), ("mujoco", B)):
        gain, lag = transfer(tr, FREE_JOINT, m)
        bands = band_gains(tr, FREE_JOINT, m)
        print(f"  {label:<7} overall gain {gain:.3f}  lag {lag*1000:+.0f} ms   "
              f"gain by band  0.5-2Hz {bands[0]:.3f}  2-4Hz {bands[1]:.3f}  4-6Hz {bands[2]:.3f}")
    ga, la = transfer(A, FREE_JOINT, m)
    gb, lb = transfer(B, FREE_JOINT, m)
    print(f"  -> gain ratio mujoco/isaac {gb/max(ga,1e-9):.3f}, lag difference {(lb-la)*1000:+.0f} ms")
    print(f"     a heavier rotor (larger armature) lowers high-frequency gain and adds lag;")
    print(f"     more joint damping lowers gain across the board and adds lag.")

    print("\n--- per-joint RMS position difference over the whole run (rad) ---")
    d = np.sqrt(((A["q"] - B["q"]) ** 2).mean(0))
    for i in range(0, 12, 3):
        print("   " + "  ".join(f"{P.JOINT_NAMES[j]:<9}{d[j]:.4f}" for j in range(i, i + 3)))

    print("\n--- torque saturation (fraction of steps at the limit) ---")
    for label, tr in (("isaac", A), ("mujoco", B)):
        sat = (np.abs(tr["tau"]) >= P.EFFORT_LIMIT * 0.999).mean(0)
        print(f"  {label:<7} " + "  ".join(f"{P.JOINT_NAMES[j]}={sat[j]*100:.0f}%"
                                           for j in range(12) if sat[j] > 0.01) or f"  {label:<7} none")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
        ax[0].plot(t, A["target"][:, FREE_JOINT], "k--", lw=0.8, label="command")
        ax[0].plot(t, A["q"][:, FREE_JOINT], label="isaac")
        ax[0].plot(t, B["q"][:, FREE_JOINT], label="mujoco")
        ax[0].set_ylabel("FL_thigh [rad]"); ax[0].legend(loc="upper right", fontsize=8)
        ax[1].plot(t, A["tau"][:, FREE_JOINT], label="isaac")
        ax[1].plot(t, B["tau"][:, FREE_JOINT], label="mujoco")
        ax[1].set_ylabel("FL_thigh torque [Nm]"); ax[1].legend(loc="upper right", fontsize=8)
        ax[2].plot(t, A["base_pos"][:, 2], label="isaac")
        ax[2].plot(t, B["base_pos"][:, 2], label="mujoco")
        ax[2].set_ylabel("base height [m]"); ax[2].legend(loc="upper right", fontsize=8)
        ax[3].plot(t, A["contact"].sum(1), label="isaac")
        ax[3].plot(t, B["contact"].sum(1), label="mujoco")
        ax[3].set_ylabel("total foot force [N]"); ax[3].set_xlabel("time [s]")
        ax[3].legend(loc="upper right", fontsize=8)
        for a in ax:
            for _, s, e in P.PHASES:
                a.axvline(s, color="0.85", lw=0.7, zorder=0)
        fig.tight_layout(); fig.savefig(args.plot, dpi=110)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
