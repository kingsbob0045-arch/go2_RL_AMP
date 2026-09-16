"""Shared open-loop protocol for the Isaac Lab <-> MuJoCo A/B comparison.

Pure numpy so both sides can import it: the Isaac side runs under env_isaacsim, the
MuJoCo side under go2_ros_env, and neither interpreter has the other's dependencies.

Everything is in Isaac joint order (FL, FR, RL, RR) x (hip, thigh, calf) -- the order
GO2_JOINT_NAMES pins in go2_env_cfg.py and the exported policies use.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES = (
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
)

# Go2AmpEnvCfg: sim dt 0.005 with decimation 4, so a target is held for 4 physics steps.
PHYSICS_DT = 0.005
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION

# UNITREE_GO2_CFG.init_state.joint_pos expanded into Isaac order.
DEFAULT_JOINT_POS = np.array(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float64
)

# Go2AmpEnvCfg actuators: IdealPDActuator, tau = Kp (q* - q) - Kd qd, clipped per joint.
STIFFNESS = 50.0
DAMPING = 1.0
EFFORT_LIMIT = np.array([23.7, 23.7, 45.43] * 4, dtype=np.float64)  # Go2 knee is 45.43, not Go1's 35.55

# Puts the feet just on the ground (FK foot z ~ -0.30, foot radius 0.022); the settle
# phase absorbs whatever is left.
BASE_HEIGHT = 0.325

FL_THIGH, FL_CALF = 1, 2
LIFTED_CALF = -2.5  # folds the shank, raising the FL foot ~18 cm clear of the ground

PHASES = (
    ("settle", 0.0, 1.0),
    ("lift", 1.0, 1.8),
    ("chirp", 1.8, 7.8),
    ("recover", 7.8, 8.8),
    ("squat", 8.8, 11.8),
    ("calf_step", 11.8, 14.8),
)


def _chirp(local: np.ndarray, f0: float, f1: float) -> np.ndarray:
    """Linear-frequency sine sweep; phase is the integral of the instantaneous rate."""
    duration = local[-1] - local[0] if len(local) > 1 else 1.0
    rate = (f1 - f0) / max(duration, 1.0e-9)
    return np.sin(2.0 * np.pi * (f0 * local + 0.5 * rate * local**2))


def build_protocol() -> tuple[np.ndarray, np.ndarray]:
    """Return (t, targets[N, 12]) sampled at the control rate.

    settle      hold the nominal stance so each engine reaches its own equilibrium
    lift        fold the FL calf until the foot clears the ground
    chirp       sweep the FL thigh with that leg in the air.  This is the measurement
                that isolates rotor inertia and joint damping from contact, because a
                free-swinging link is the only place they are not masked by ground reaction
    recover     back to nominal, settle again
    squat       all four legs load and unload -- contact plus torque saturation
    calf_step   square-wave the calves -- contact stiffness and damping
    """
    total = PHASES[-1][2]
    n = int(round(total / CONTROL_DT))
    t = np.arange(n) * CONTROL_DT
    targets = np.tile(DEFAULT_JOINT_POS, (n, 1))

    for name, start, end in PHASES:
        mask = (t >= start) & (t < end)
        if not mask.any():
            continue
        local = t[mask] - start
        span = end - start
        smooth = 0.5 - 0.5 * np.cos(np.pi * local / span)  # C1 ramp, no step transient
        if name == "lift":
            targets[mask, FL_CALF] = DEFAULT_JOINT_POS[FL_CALF] + smooth * (
                LIFTED_CALF - DEFAULT_JOINT_POS[FL_CALF])
        elif name == "chirp":
            targets[mask, FL_CALF] = LIFTED_CALF
            targets[mask, FL_THIGH] = DEFAULT_JOINT_POS[FL_THIGH] + 0.40 * _chirp(local, 0.5, 6.0)
        elif name == "recover":
            targets[mask, FL_CALF] = LIFTED_CALF + smooth * (
                DEFAULT_JOINT_POS[FL_CALF] - LIFTED_CALF)
        elif name == "squat":
            wave = np.sin(2.0 * np.pi * 1.0 * local)
            for leg in range(4):
                targets[mask, leg * 3 + 1] = DEFAULT_JOINT_POS[leg * 3 + 1] + 0.35 * wave
                targets[mask, leg * 3 + 2] = DEFAULT_JOINT_POS[leg * 3 + 2] - 0.55 * wave
        elif name == "calf_step":
            square = np.where(((local * 1.5) % 1.0) < 0.5, 1.0, -1.0)
            for leg in range(4):
                targets[mask, leg * 3 + 2] = DEFAULT_JOINT_POS[leg * 3 + 2] + 0.25 * square

    return t, targets


def phase_mask(t: np.ndarray, name: str) -> np.ndarray:
    for label, start, end in PHASES:
        if label == name:
            return (t >= start) & (t < end)
    raise KeyError(name)
