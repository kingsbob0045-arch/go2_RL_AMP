"""Measure the footfall phase of a converted motion clip, and score it against gait templates.

Why this exists
---------------
DogML clips are labelled by a single free-text word in the source annotation ("walk", "trot",
"run"), which the converter maps to the three SECAMP skills.  That label is the annotator's
description of the whole video, not a measurement of the clip that was cut out of it, and it
is wrong often enough to matter: a "walk" clip that is really a trot teaches the pace
discriminator a trot, and the skill conditioning silently stops meaning anything.

What it does *not* do
---------------------
It does not relabel.  Three earlier classifier variants disagreed with each other about where
the trot/walk/amble boundary sits, which is exactly the region most DogML clips live in, so an
automatic relabelling would be trading a noisy human label for a noisy machine one.  Instead
this only supplies evidence for *rejection*: a clip is dropped when the measured footfall
pattern matches some other skill decisively better than its own.  The walk templates are
deliberately grouped under `pace`, because mapping walk to the pace skill is a deliberate
choice that is being kept, not an error to be screened out.

Method
------
Per-leg world foot height is a clean, near-sinusoidal stride signal: it peaks mid-swing and
flattens during stance.  The dominant stride frequency is taken from the summed spectrum of
all four legs, and each leg's phase is read from its own complex FFT coefficient in that same
bin.  Phases are expressed relative to the front-left leg, so the result is invariant to where
in the stride the clip happens to start.

Reading the phase from one shared bin, rather than per-leg peak picking, is what makes the
result stable on short clips: all four legs are forced to describe the same stride, so a leg
whose own spectrum is ambiguous cannot drift into a different cycle.
"""

from __future__ import annotations

import numpy as np

# Column layout of a converted AMP frame (61 columns).
ROOT_POS = slice(0, 3)
FOOT_POS = slice(19, 31)

# Relative footfall phase of each leg with respect to FL, in cycles, for the gaits the three
# SECAMP skills have to cover.  Leg order is Isaac's: FL, FR, RL, RR.
#
# `walk` sits under `pace` on purpose -- see the module docstring.  `bound` sits under
# `canter` because it is the same asymmetric family: both break the left/right alternation
# that defines pace and trot, and neither should ever be screened out in favour of one of
# those two.
GAIT_TEMPLATES: dict[str, tuple[tuple[float, float, float, float], ...]] = {
    "pace": (
        (0.0, 0.50, 0.00, 0.50),   # true pace: lateral pairs together
        (0.0, 0.50, 0.25, 0.75),   # lateral-sequence walk, left lead
        (0.0, 0.50, 0.75, 0.25),   # lateral-sequence walk, right lead
    ),
    "trot": (
        (0.0, 0.50, 0.50, 0.00),   # diagonal pairs together
    ),
    "canter": (
        (0.0, 0.10, 0.50, 0.60),   # transverse gallop
        (0.0, 0.10, 0.60, 0.50),   # rotary gallop
        (0.0, 0.00, 0.50, 0.50),   # bound: front pair together, rear pair together
    ),
}

# Stride frequencies outside this band are not dog gaits at the speeds these clips contain;
# admitting them lets a slow drift or a single-frame glitch masquerade as the stride.
MIN_STRIDE_HZ = 0.5
MAX_STRIDE_HZ = 6.0
# Shortest clip whose spectrum can resolve a stride at MIN_STRIDE_HZ at all.
MIN_FRAMES = 24


def _circular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance between two phases in cycles, wrapping at 1.0; range [0, 0.5]."""
    difference = np.abs(a - b) % 1.0
    return np.minimum(difference, 1.0 - difference)


def measure_phase(frames: np.ndarray, frame_duration: float) -> dict[str, object]:
    """Measure stride frequency, per-leg relative phase, and how periodic the clip is.

    Args:
        frames: converted motion, shape (T, 61).
        frame_duration: seconds per frame.

    Returns:
        dict with `stride_hz`, `phase` (4 floats in [0, 1), FL first and always 0.0),
        `periodicity` in [0, 1], and `measurable` -- False when the clip is too short or has no
        usable stride peak, in which case the phase must not be used as evidence for anything.
    """
    frames = np.asarray(frames, dtype=np.float64)
    num_frames = len(frames)
    unmeasurable = {"stride_hz": 0.0, "phase": [0.0, 0.0, 0.0, 0.0],
                    "periodicity": 0.0, "measurable": False}
    if num_frames < MIN_FRAMES:
        return unmeasurable

    # World foot height: the converter stores foot position relative to the root.
    root_z = frames[:, ROOT_POS][:, 2]
    foot_z = frames[:, FOOT_POS].reshape(num_frames, 4, 3)[:, :, 2] + root_z[:, None]

    # Remove the mean and taper the ends.  Without the window a clip that does not contain a
    # whole number of strides leaks power across neighbouring bins, which moves the phase.
    signal = foot_z - foot_z.mean(axis=0, keepdims=True)
    window = np.hanning(num_frames)
    spectrum = np.fft.rfft(signal * window[:, None], axis=0)          # (F, 4)
    frequencies = np.fft.rfftfreq(num_frames, d=frame_duration)        # (F,)

    power = np.abs(spectrum) ** 2
    total_power = power[1:].sum()                                      # exclude DC
    if total_power <= 0.0:
        return unmeasurable

    band = (frequencies >= MIN_STRIDE_HZ) & (frequencies <= MAX_STRIDE_HZ)
    if not band.any():
        return unmeasurable
    summed = power.sum(axis=1)
    summed_in_band = np.where(band, summed, 0.0)
    peak = int(np.argmax(summed_in_band))
    if summed_in_band[peak] <= 0.0:
        return unmeasurable

    # How much of the clip's motion the single stride bin explains.  A clean periodic gait
    # concentrates its power here; a clip that stumbles, changes gait mid-way or is mostly
    # noise spreads it out, and its phase means nothing.
    periodicity = float(power[peak].sum() / total_power)

    # Phase is reported as a footfall *lag*: how far after FL, in cycles, each leg repeats
    # FL's motion.  A leg lagging by tau contributes exp(-i.2.pi.f.tau) to the transform, so
    # its FFT angle runs the other way and has to be negated to read as a lag.  Templates are
    # written as lags, and two of them (the walks) are chiral, so the sign is load-bearing.
    angles = np.angle(spectrum[peak])                                  # (4,)
    phase = ((angles[0] - angles) / (2.0 * np.pi)) % 1.0
    phase[0] = 0.0
    return {
        "stride_hz": float(frequencies[peak]),
        "phase": [float(value) for value in phase],
        "periodicity": periodicity,
        "measurable": True,
    }


def score_gaits(phase: list[float] | np.ndarray) -> dict[str, float]:
    """Score a measured phase against every skill, in [0, 1]; 1.0 is an exact template match.

    Only FR, RL and RR carry information -- FL is the reference and is 0.0 by construction --
    so the mean is taken over those three.  Distances are normalised by the 0.5 maximum of a
    circular distance, which makes 0.0 the score of a leg in perfect antiphase to its template.
    """
    phase = np.asarray(phase, dtype=np.float64)
    scores = {}
    for skill, templates in GAIT_TEMPLATES.items():
        best = 0.0
        for template in templates:
            distance = _circular_distance(phase[1:], np.asarray(template)[1:]).mean()
            best = max(best, 1.0 - distance / 0.5)
        scores[skill] = float(best)
    return scores


def screen_clip(frames: np.ndarray, frame_duration: float, skill: str,
                *, margin: float, min_periodicity: float) -> dict[str, object]:
    """Decide whether a clip's measured footfall pattern contradicts its assigned skill.

    Returns the measurement, the per-skill scores, and a `verdict` of "accept", or a rejection
    reason.  A clip is only rejected on positive evidence: either the stride could not be
    measured well enough for the label to be checked at all, or some other skill fits the
    measured phase better than the assigned one by more than `margin`.
    """
    measurement = measure_phase(frames, frame_duration)
    result: dict[str, object] = dict(measurement)
    result["skill"] = skill

    if not measurement["measurable"]:
        # Shorter than one stride, so the label cannot be checked either way.  That is an
        # absence of evidence, not evidence of a problem, and this screen only rejects on
        # positive evidence -- so the clip is kept and marked as never having been verified.
        # AMP never looks at more than amp_horizon frames at once anyway, so a sub-stride clip
        # is still usable expert data; only its skill label is unconfirmed.
        result["verdict"] = "accept_unverified"
        result["scores"] = {}
        return result

    # Scores are recorded for every measurable clip, including ones the periodicity gate is
    # about to reject, so the thresholds can be re-chosen later from the saved record without
    # re-running the conversion.
    scores = score_gaits(measurement["phase"])
    own = scores[skill]
    rival, rival_score = max(
        ((name, value) for name, value in scores.items() if name != skill),
        key=lambda item: item[1])
    result["scores"] = scores
    result["own_score"] = own
    result["best_other"] = rival
    result["best_other_score"] = rival_score

    if measurement["periodicity"] < min_periodicity:
        result["verdict"] = "reject_aperiodic"
    elif rival_score - own > margin:
        result["verdict"] = "reject_mislabelled"
    else:
        result["verdict"] = "accept"
    return result
