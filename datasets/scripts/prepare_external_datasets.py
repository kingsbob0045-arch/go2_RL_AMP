#!/usr/bin/env python3
"""Download, convert, and validate external motions for SECAMP."""

from __future__ import annotations

import argparse
import sys
import hashlib
import io
import json
import math
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gait_phase import screen_clip  # noqa: E402

DATASETS_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = DATASETS_ROOT / "external_raw"
OUTPUT_ROOT = DATASETS_ROOT / "converted"
CHECKSUM_FILE = DATASETS_ROOT / "external_checksums.sha256"

KINE2GO_REVISION = "3698e74f8f9e889697cafc872fb32dd5e8285fb7"
MOCAP_GAIT_FILES = (
    "pace0.txt", "pace1.txt", "pace2.txt",
    "trot0.txt", "trot1.txt", "trot2.txt",
    "canter0.txt", "canter1.txt", "canter2.txt",
)
KINE2GO_DOG_CLIPS = (
    "ai4_dog_canter",
    "ai4_dog_pace",
    "ai4_dog_trot_00",
    "ai4_dog_trot_01",
)
# Horse clips from the same release, deliberately excluded: a horse's limb proportions and
# stride timing are not a dog's, and the three SECAMP skills are meant to be dog gaits.
# Kept here so the exclusion is visible rather than implied by an absence.
KINE2GO_HORSE_CLIPS = (
    "vhdc_horse1_s1_trot_01",
    "vhdc_horse1_s1_trot_02",
    "vhdc_horse1_s1_trot_03",
    "vhdc_horse1_s2_trot_01",
    "vhdc_horse1_s2_trot_02",
    "vhdc_horse1_s2_trot_03",
)
KINE2GO_GAIT_CLIPS = KINE2GO_DOG_CLIPS
KINE2GO_FPS_OVERRIDES = {
    "vhdc_horse1_s2_trot_01": 120.0,
    "vhdc_horse1_s2_trot_02": 120.0,
    "vhdc_horse1_s2_trot_03": 120.0,
}
NJU_FILES = (
    "canter_0.json", "canter_1.json", "canter_2.json", "canter_3.json",
    "pace_0.json", "pace_1.json", "pace_2.json",
    "trot_0.json", "trot_1.json", "trot_2.json",
)
DOGML_FILE_ID = "1sb46FX8QBtS8EcgQNFw12AKUgQp9RRIx"
DOGML_FRAME_DURATION = 1.0 / 60.0
DOGML_LABEL_TO_SKILL = {"walk": "pace", "trot": "trot", "run": "canter"}
DOGML_CHAIN_ORDER_FL_FR_RL_RR = (4, 0, 12, 8)
DOGML_MIN_DISTANCE = 0.25
DOGML_MIN_MOVING_FRACTION = 0.60

# Footfall-phase screening of the DogML label (datasets/scripts/gait_phase.py).  The source
# label is one free-text word describing a whole video, so it does not survive being cut into
# clips intact.  Screening rejects; it never relabels, because three earlier classifier
# variants disagreed with each other about exactly the trot/walk boundary most of these clips
# sit on.  A clip is only dropped when another skill fits its measured phase better than its
# own by more than DOGML_PHASE_MARGIN, or when the stride is too weak for the label to be
# checked at all.  0.15 of the 0-1 score is roughly 0.075 cycles of mean footfall error per
# leg: far beyond what measurement noise produces on a real gait, so an accepted clip that is
# genuinely mislabelled has to be mislabelled subtly.
#
# Both thresholds were chosen from the measured distribution over all 1214 clips rather than
# picked a priori (datasets/converted/dogml_phase_screen.json holds the raw measurements):
#   * margin 0.20 rejects 37 clips, the tail of a distribution whose median clip fits its own
#     label 0.23 better than any other.  29 of the 37 are DogML "trot" clips that measure as
#     clean lateral-sequence walks -- e.g. phase [0.00, 0.38, 0.24, 0.75] at periodicity 0.73,
#     which is a walk by any reading.  They come from a handful of source videos, so the
#     annotator labelled the whole video and the walking segments inherited it.
#   * periodicity 0.15 rejects the bottom 14% of measurable clips, whose foot-height motion is
#     not dominated by any single stride: turns, gait transitions and stumbles.  Their phase
#     is not trustworthy enough to screen on, and they are not clean gait exemplars either.
DOGML_PHASE_MARGIN = 0.20
DOGML_MIN_PERIODICITY = 0.15

GO2_HIP_OFFSETS = np.asarray((
    (0.214512, 0.0465, -0.005366),
    (0.214512, -0.0465, -0.005366),
    (-0.172288, 0.0465, -0.005366),
    (-0.172288, -0.0465, -0.005366),
))
GO2_HIP_LINK_LENGTH = 0.0955
GO2_LEG_LINK_LENGTH = 0.213
# Official Go2 joint limits, in Isaac leg order FL, FR, RL, RR.  Front and rear thighs
# differ, which a uniform tile cannot express -- the values these replaced were uniform and
# matched no joint of the Go2 at all.  They were also narrower, so _go2_inverse_kinematics
# was clipping retargeted poses the robot can actually reach: the thigh was held at
# -0.686 rad where the Go2 reaches -1.5708, cutting most of the forward swing range.
# Cross-checked against go2_Deploy/resources/go2/go2.xml, which carries the same numbers.
GO2_JOINT_LOWER = np.array(
    [-1.0472, -1.5708, -2.7227] * 2 + [-1.0472, -0.5236, -2.7227] * 2, dtype=np.float64
)
GO2_JOINT_UPPER = np.array(
    [1.0472, 3.4907, -0.83776] * 2 + [1.0472, 4.5379, -0.83776] * 2, dtype=np.float64
)

# Source layouts, established by brute-forcing every (block layout x 24 leg permutations x
# abduction sign) combination against Go2 forward kinematics and keeping the one whose feet
# agree with its own joint angles.  The winner is decisive in every case -- it lands at the
# 0.022 m residual of the analytic FK model itself, while the runner-up is 2-10x worse.
# tools/dataset_layout_check.py re-runs the search; validate_motion() enforces the result.
#
# Isaac target order is FL, FR, RL, RR.
LEG_REORDER_FR_FL_RR_RL = (1, 0, 3, 2)   # source FR,FL,RR,RL -> Isaac
LEG_REORDER_FL_RL_FR_RR = (0, 2, 1, 3)   # source FL,RL,FR,RR -> Isaac
LEG_REORDER_IDENTITY = (0, 1, 2, 3)      # source already FL,FR,RL,RR

# Kine2Go stores joints joint-major -- [4 abduction][4 thigh][4 calf] -- not leg-major.
# Reading it as leg-major scrambles every joint: it put the FK residual at 0.28 m and drove
# 100% of frames outside the Go2 joint limits.  The column means read joint-major are
# abduction ~0, thigh 0.70/0.73 front and 0.94/1.03 rear, calf ~-1.5, i.e. the Go2 stance.
KINE2GO_JOINTS_ARE_JOINT_MAJOR = True
KINE2GO_JOINT_LEGS = LEG_REORDER_IDENTITY
KINE2GO_FOOT_LEGS = LEG_REORDER_FL_RL_FR_RR

# NJU stores joints leg-major and already in Isaac order, but its feet are in the PyBullet
# FR,FL,RR,RL order -- confirmed independently by the sign of each foot's mean x and y.
# The previous code applied the foot permutation to the joints as well and additionally
# negated every abduction angle; both were wrong and together cost 0.163 m of FK residual.
NJU_JOINT_LEGS = LEG_REORDER_IDENTITY
NJU_FOOT_LEGS = LEG_REORDER_FR_FL_RR_RL

# Acceptance gate for a converted clip.  The analytic FK used by go2_env.py differs from the
# true Go2 kinematics by ~0.022 m on its own, so anything past 0.05 m is a layout bug, not
# model error.
MAX_FK_RESIDUAL = 0.05


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        output.write(response.read())


def _verify_raw_files(paths: list[Path]) -> None:
    checksums = {}
    for line in CHECKSUM_FILE.read_text().splitlines():
        digest, relative_path = line.split(maxsplit=1)
        checksums[relative_path] = digest
    for path in paths:
        relative_path = path.relative_to(RAW_ROOT).as_posix()
        expected = checksums.get(relative_path)
        if expected is None:
            raise ValueError(f"No checksum recorded for {relative_path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {relative_path}: expected {expected}, got {actual}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_kine2go() -> None:
    root = RAW_ROOT / "kine2go_gaits"
    base = (
        "https://huggingface.co/datasets/kine2go-review/kine2go/resolve/"
        f"{KINE2GO_REVISION}/data"
    )
    for clip in KINE2GO_GAIT_CLIPS:
        _download(f"{base}/{clip}/motion.npy", root / f"{clip}_motion.npy")
        _download(f"{base}/{clip}/clip.json", root / f"{clip}_clip.json")


def download_nju() -> None:
    root = RAW_ROOT / "nju_agility"
    base = (
        "https://raw.githubusercontent.com/NJU-RLC/quadrupedal-agility/"
        "15d16ea99bc23e5d401b0b0e3b88a8edcc28f0ed/"
        "bbc/mocap_data/mocap_all_lb"
    )
    for filename in NJU_FILES:
        _download(f"{base}/{filename}", root / filename)


def download_dogml() -> None:
    destination = RAW_ROOT / "dogml" / "dataset.zip"
    if destination.exists():
        return
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError(
            "DogML is hosted on Google Drive. Install gdown or place dataset.zip in "
            f"{destination.parent}."
        ) from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=DOGML_FILE_ID, output=str(destination), quiet=False)
    if result is None or not destination.exists():
        raise RuntimeError(f"DogML download did not create {destination}")


def _reorder_legs(values: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    """Reorder a leg-major block: [leg0 xyz | leg1 xyz | ...] -> the requested leg order."""
    return values.reshape(len(values), 4, 3)[:, order, :].reshape(len(values), 12)


def _joint_major_to_leg_major(values: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    """Transpose a joint-major block into the leg-major layout the AMP format expects.

    Input  [abduction x4 | thigh x4 | calf x4]
    Output [leg0 (abduction, thigh, calf) | leg1 ... ] in the requested leg order.
    """
    per_joint = values.reshape(len(values), 3, 4)[:, :, order]
    return np.transpose(per_joint, (0, 2, 1)).reshape(len(values), 12)


def _quat_rotate_inverse_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = -quaternion[:, :3]
    w = quaternion[:, 3:4]
    cross = np.cross(xyz, vector)
    return vector + 2.0 * (w * cross + np.cross(xyz, cross))


def _quat_wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
    return quaternion[:, (1, 2, 3, 0)]


def _quat_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[:, 1:]
    cross = np.cross(xyz, vector)
    return vector + 2.0 * (quaternion[:, :1] * cross + np.cross(xyz, cross))


def _finite_difference(values: np.ndarray, frame_duration: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, frame_duration, axis=0, edge_order=1)


def _write_motion(path: Path, frames: np.ndarray, frame_duration: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "LoopMode": "Wrap",
        "FrameDuration": frame_duration,
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": False,
        "MotionWeight": 1.0,
        "Frames": frames.tolist(),
    }
    with path.open("w") as output:
        json.dump(payload, output, separators=(",", ":"))


def prepare_mocap_gaits() -> list[Path]:
    source_root = DATASETS_ROOT / "mocap_motions_go2"
    output_root = OUTPUT_ROOT / "mocap_gaits"
    if output_root.exists():
        for stale_path in output_root.iterdir():
            stale_path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for filename in MOCAP_GAIT_FILES:
        source = source_root / filename
        destination = output_root / filename
        shutil.copy2(source, destination)
        output_paths.append(destination)
    return output_paths


def convert_kine2go() -> list[Path]:
    raw_root = RAW_ROOT / "kine2go_gaits"
    output_root = OUTPUT_ROOT / "kine2go_gaits"
    if output_root.exists():
        for stale_path in output_root.glob("*.json"):
            stale_path.unlink()
    output_paths: list[Path] = []
    skill_counts = {"pace": 0, "trot": 0, "canter": 0}

    for clip in KINE2GO_GAIT_CLIPS:
        metadata = json.loads((raw_root / f"{clip}_clip.json").read_text())
        source = np.load(raw_root / f"{clip}_motion.npy")
        if source.ndim != 2 or source.shape[1] != 61:
            raise ValueError(f"{clip}: expected motion.npy shape (T, 61), got {source.shape}")

        tags = set(metadata["tags"])
        skill = next((name for name in skill_counts if name in tags), "turn")
        if skill == "turn":
            raise ValueError(f"{clip}: expected a pace/trot/canter tag, got {metadata['tags']}")
        skill_counts[skill] += 1
        destination = output_root / f"{skill}_{clip}.json"

        root_pos = source[:, 48:51]
        root_quat = _quat_wxyz_to_xyzw(source[:, 51:55])
        joint_pos = _joint_major_to_leg_major(source[:, 6:18], KINE2GO_JOINT_LEGS)
        joint_vel = _joint_major_to_leg_major(source[:, 24:36], KINE2GO_JOINT_LEGS)
        foot_world = _reorder_legs(source[:, 36:48], KINE2GO_FOOT_LEGS)
        foot_local = _quat_rotate_inverse_xyzw(
            np.repeat(root_quat, 4, axis=0),
            (foot_world.reshape(-1, 3) - np.repeat(root_pos, 4, axis=0)),
        ).reshape(len(source), 12)
        lin_vel = _quat_rotate_inverse_xyzw(root_quat, source[:, 55:58])
        ang_vel = _quat_rotate_inverse_xyzw(root_quat, source[:, 58:61])
        frame_duration = 1.0 / KINE2GO_FPS_OVERRIDES.get(
            clip, float(metadata["fps"])
        )
        foot_vel = _finite_difference(foot_local, frame_duration)
        frames = np.hstack((
            root_pos, root_quat, joint_pos, foot_local,
            lin_vel, ang_vel, joint_vel, foot_vel,
        ))
        _write_motion(destination, frames, frame_duration)
        output_paths.append(destination)

    print(f"Kine2Go skill coverage: {skill_counts}")
    return output_paths


def convert_nju() -> list[Path]:
    raw_root = RAW_ROOT / "nju_agility"
    output_root = OUTPUT_ROOT / "nju_agility"
    if output_root.exists():
        for stale_path in output_root.glob("*.json"):
            stale_path.unlink()
    output_paths: list[Path] = []

    for filename in NJU_FILES:
        payload = json.loads((raw_root / filename).read_text())
        source = np.asarray(payload["Frames"], dtype=np.float64)
        if source.ndim != 2 or source.shape[1] != 61:
            raise ValueError(f"{filename}: expected Frames shape (T, 61), got {source.shape}")

        root_pos = source[:, 0:3].copy()
        root_quat = source[:, 3:7]
        # Joints are already in Isaac order; only the feet carry the PyBullet leg order.
        # The abduction angles are in Isaac's sign convention too -- negating them, as this
        # converter used to, is what put the feet 0.16 m away from where the joints place them.
        joint_pos = _reorder_legs(source[:, 7:19], NJU_JOINT_LEGS)
        joint_vel = _reorder_legs(source[:, 37:49], NJU_JOINT_LEGS)
        foot_world = _reorder_legs(source[:, 19:31], NJU_FOOT_LEGS)

        foot_height = foot_world[:, 2::3]
        ground_offset = float(np.min(foot_height))
        foot_height -= ground_offset
        root_pos[:, 2] -= ground_offset

        foot_pos = _quat_rotate_inverse_xyzw(
            np.repeat(root_quat, 4, axis=0),
            (foot_world.reshape(-1, 3) - np.repeat(root_pos, 4, axis=0)),
        ).reshape(len(source), 12)
        lin_vel = _quat_rotate_inverse_xyzw(root_quat, source[:, 31:34])
        ang_vel = _quat_rotate_inverse_xyzw(root_quat, source[:, 34:37])
        foot_vel = _finite_difference(foot_pos, float(payload["FrameDuration"]))
        frames = np.hstack((
            root_pos, root_quat, joint_pos, foot_pos,
            lin_vel, ang_vel, joint_vel, foot_vel,
        ))
        destination = output_root / filename
        _write_motion(destination, frames, float(payload["FrameDuration"]))
        output_paths.append(destination)
    return output_paths


def _recover_dogml_root(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation_angle = np.zeros(len(source), dtype=np.float64)
    rotation_angle[1:] = source[:-1, 0]
    rotation_angle = np.cumsum(rotation_angle)

    source_quat = np.zeros((len(source), 4), dtype=np.float64)
    source_quat[:, 0] = np.cos(rotation_angle)
    source_quat[:, 2] = -np.sin(rotation_angle)
    source_position = np.zeros((len(source), 3), dtype=np.float64)
    source_position[1:, (0, 2)] = source[:-1, 1:3]
    source_position = np.cumsum(
        _quat_rotate_wxyz(source_quat, source_position), axis=0
    )
    source_position[:, 1] = source[:, 3]

    root_position = source_position[:, (2, 0, 1)]
    root_quat = np.zeros((len(source), 4), dtype=np.float64)
    root_quat[:, 2] = -np.sin(rotation_angle)
    root_quat[:, 3] = np.cos(rotation_angle)
    return root_position, root_quat, -2.0 * rotation_angle


def _dogml_target_feet(source: np.ndarray) -> np.ndarray:
    root = np.column_stack((np.zeros(len(source)), source[:, 3], np.zeros(len(source))))
    keypoints = np.concatenate(
        (root[:, None, :], source[:, 4:49].reshape(len(source), 15, 3)), axis=1
    )
    feet = np.empty((len(source), 4, 3), dtype=np.float64)
    target_lengths = (GO2_HIP_LINK_LENGTH, GO2_LEG_LINK_LENGTH, GO2_LEG_LINK_LENGTH)
    for leg, chain_start in enumerate(DOGML_CHAIN_ORDER_FL_FR_RL_RR):
        target = GO2_HIP_OFFSETS[leg].copy()
        for segment, target_length in enumerate(target_lengths):
            vector = keypoints[:, chain_start + segment + 1] - keypoints[:, chain_start + segment]
            vector = vector[:, (2, 0, 1)]
            length = np.linalg.norm(vector, axis=1, keepdims=True)
            if np.any(length < 1.0e-6):
                raise ValueError("DogML motion contains a zero-length leg segment")
            target = target + vector * (target_length / length)
        feet[:, leg] = target
    return feet


def _go2_inverse_kinematics(foot_positions: np.ndarray) -> np.ndarray:
    joint_positions = np.empty_like(foot_positions)
    for leg in range(4):
        relative = foot_positions[:, leg] - GO2_HIP_OFFSETS[leg]
        hip_y = GO2_HIP_LINK_LENGTH * (1.0 if leg % 2 == 0 else -1.0)
        z_hip = -np.sqrt(np.maximum(
            np.square(relative[:, 1]) + np.square(relative[:, 2]) - hip_y**2,
            1.0e-8,
        ))
        abduction = (
            np.arctan2(relative[:, 2], relative[:, 1])
            - np.arctan2(z_hip, np.full(len(relative), hip_y))
        )
        abduction = (abduction + np.pi) % (2.0 * np.pi) - np.pi
        leg_length = np.sqrt(np.square(relative[:, 0]) + np.square(z_hip))
        knee = -2.0 * np.arccos(np.clip(
            leg_length / (2.0 * GO2_LEG_LINK_LENGTH), 0.0, 1.0
        ))
        swing = np.arctan2(-relative[:, 0], -z_hip)
        hip = swing - knee / 2.0
        joint_positions[:, leg] = np.column_stack((abduction, hip, knee))
    flattened = joint_positions.reshape(len(foot_positions), 12)
    return np.clip(flattened, GO2_JOINT_LOWER, GO2_JOINT_UPPER)


def _go2_forward_kinematics(joint_positions: np.ndarray) -> np.ndarray:
    joint_positions = joint_positions.reshape(len(joint_positions), 4, 3)
    feet = np.empty_like(joint_positions)
    for leg in range(4):
        abduction, hip, knee = joint_positions[:, leg].T
        length = np.sqrt(
            GO2_LEG_LINK_LENGTH**2 * (2.0 + 2.0 * np.cos(knee))
        )
        swing = hip + knee / 2.0
        x = -length * np.sin(swing)
        z_hip = -length * np.cos(swing)
        hip_y = GO2_HIP_LINK_LENGTH * (1.0 if leg % 2 == 0 else -1.0)
        y = np.cos(abduction) * hip_y - np.sin(abduction) * z_hip
        z = np.sin(abduction) * hip_y + np.cos(abduction) * z_hip
        feet[:, leg] = np.column_stack((x, y, z)) + GO2_HIP_OFFSETS[leg]
    return feet.reshape(len(joint_positions), 12)


def _convert_dogml_motion(source: np.ndarray) -> tuple[np.ndarray, float]:
    target_feet = _dogml_target_feet(source)
    joint_pos = _go2_inverse_kinematics(target_feet)
    foot_pos = _go2_forward_kinematics(joint_pos)
    error = np.linalg.norm(
        foot_pos.reshape(len(source), 4, 3) - target_feet, axis=2
    )

    root_pos, root_quat, root_yaw = _recover_dogml_root(source)
    lowest_foot_height = root_pos[:, 2] + np.min(foot_pos[:, 2::3], axis=1)
    # Apply one clip-wide ground offset so genuine airborne phases remain intact.
    root_pos[:, 2] += 0.02 - np.min(lowest_foot_height)
    world_lin_vel = _finite_difference(root_pos, DOGML_FRAME_DURATION)
    lin_vel = _quat_rotate_inverse_xyzw(root_quat, world_lin_vel)
    ang_vel = np.zeros((len(source), 3), dtype=np.float64)
    ang_vel[:, 2] = _finite_difference(root_yaw, DOGML_FRAME_DURATION)
    joint_vel = _finite_difference(joint_pos, DOGML_FRAME_DURATION)
    foot_vel = _finite_difference(foot_pos, DOGML_FRAME_DURATION)
    frames = np.hstack((
        root_pos, root_quat, joint_pos, foot_pos,
        lin_vel, ang_vel, joint_vel, foot_vel,
    ))
    return frames, float(np.max(error))


def convert_dogml(*, enforce_phase_screen: bool = True) -> list[Path]:
    archive = RAW_ROOT / "dogml" / "dataset.zip"
    output_root = OUTPUT_ROOT / "dogml_gaits"
    if output_root.exists():
        for stale_path in output_root.glob("*.json"):
            stale_path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    manifest: list[dict[str, object]] = []
    skill_counts = {"pace": 0, "trot": 0, "canter": 0}
    excluded_counts = {
        "non_gait_label": 0,
        "duplicate": 0,
        "conflicting_label": 0,
        "stationary": 0,
        "reject_mislabelled": 0,
        "reject_aperiodic": 0,
    }
    # Kept, but with an unverifiable label -- counted separately so the number is visible.
    unverified_count = 0
    screen_records: list[dict[str, object]] = []
    with zipfile.ZipFile(archive) as dataset:
        text_paths = sorted(
            path for path in dataset.namelist()
            if path.startswith("dataset/text/") and path.endswith(".txt")
        )
        candidates: dict[str, list[tuple[str, str, str]]] = {}
        for text_path in text_paths:
            annotation = dataset.read(text_path).decode("utf-8", errors="strict")
            label = annotation.splitlines()[0].lstrip("#").strip().lower()
            if label not in DOGML_LABEL_TO_SKILL:
                excluded_counts["non_gait_label"] += 1
                continue

            stem = Path(text_path).stem
            motion_path = f"dataset/motion/robot/{stem}.npy"
            raw_motion = dataset.read(motion_path)
            digest = hashlib.sha256(raw_motion).hexdigest()
            candidates.setdefault(digest, []).append((label, stem, motion_path))

        for digest, records in sorted(candidates.items()):
            labels = {record[0] for record in records}
            if len(labels) != 1:
                excluded_counts["conflicting_label"] += len(records)
                continue
            excluded_counts["duplicate"] += len(records) - 1
            label, stem, motion_path = min(records, key=lambda record: record[1])
            raw_motion = dataset.read(motion_path)
            source = np.load(io.BytesIO(raw_motion), allow_pickle=False)
            if source.ndim != 2 or source.shape[1] != 191 or len(source) < 2:
                raise ValueError(
                    f"{motion_path}: expected motion shape (T, 191), got {source.shape}"
                )
            if not np.isfinite(source).all():
                raise ValueError(f"{motion_path}: contains NaN or infinity")

            step_distance = np.linalg.norm(source[:, 1:3], axis=1)
            distance = float(np.sum(step_distance))
            moving_fraction = float(np.mean(step_distance > 1.0e-3))
            if distance < DOGML_MIN_DISTANCE or moving_fraction < DOGML_MIN_MOVING_FRACTION:
                excluded_counts["stationary"] += 1
                continue

            skill = DOGML_LABEL_TO_SKILL[label]
            frames, max_retarget_error = _convert_dogml_motion(source)
            if max_retarget_error > 0.075:
                raise ValueError(
                    f"{motion_path}: Go2 retargeting error {max_retarget_error:.4f} m "
                    "exceeds 0.075 m"
                )
            screen = screen_clip(
                frames, DOGML_FRAME_DURATION, skill,
                margin=DOGML_PHASE_MARGIN, min_periodicity=DOGML_MIN_PERIODICITY)
            screen_records.append({"stem": stem, "label": label, **screen})
            if screen["verdict"] == "accept_unverified":
                unverified_count += 1
            elif screen["verdict"] != "accept":
                excluded_counts[screen["verdict"]] += 1
                if enforce_phase_screen:
                    continue

            destination = output_root / f"{skill}_{stem}.json"
            _write_motion(destination, frames, DOGML_FRAME_DURATION)
            output_paths.append(destination)
            skill_counts[skill] += 1
            manifest.append({
                "output": destination.name,
                "source": motion_path,
                "source_sha256": digest,
                "source_label": label,
                "skill": skill,
                "frames": len(source),
                "distance": distance,
                "moving_fraction": moving_fraction,
                "max_retarget_error_m": max_retarget_error,
                "stride_hz": screen["stride_hz"],
                "phase": screen["phase"],
                "periodicity": screen["periodicity"],
                "phase_scores": screen["scores"],
                "phase_verdict": screen["verdict"],
            })

    manifest_path = OUTPUT_ROOT / "dogml_gaits_manifest.json"
    manifest_path.write_text(json.dumps({
        "selection": {
            "accepted_labels": DOGML_LABEL_TO_SKILL,
            "minimum_distance": DOGML_MIN_DISTANCE,
            "minimum_moving_fraction": DOGML_MIN_MOVING_FRACTION,
            "excluded_labels": "all mixed labels and all labels other than walk/trot/run",
            "label_conflicts": "reject every copy when one motion digest has multiple gait labels",
            "phase_screen": {
                "enforced": enforce_phase_screen,
                "margin": DOGML_PHASE_MARGIN,
                "min_periodicity": DOGML_MIN_PERIODICITY,
                "note": "rejects only; never relabels -- see datasets/scripts/gait_phase.py",
                "kept_unverified": unverified_count,
            },
        },
        "skill_counts": skill_counts,
        "excluded_counts": excluded_counts,
        "motions": manifest,
    }, indent=2))
    screen_path = OUTPUT_ROOT / "dogml_phase_screen.json"
    screen_path.write_text(json.dumps(screen_records, indent=2))
    print(f"DogML skill coverage: {skill_counts}; exclusions: {excluded_counts}")
    print(f"DogML clips kept with an unverifiable label (shorter than one stride): "
          f"{unverified_count}")
    print(f"DogML selection manifest: {manifest_path}")
    print(f"DogML phase screening record: {screen_path} "
          f"({'enforced' if enforce_phase_screen else 'REPORT ONLY'})")
    return output_paths


def validate_motion(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    frames = np.asarray(payload["Frames"], dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != 61 or len(frames) < 2:
        raise ValueError(f"{path}: expected at least two 61-value frames, got {frames.shape}")
    if not np.isfinite(frames).all():
        raise ValueError(f"{path}: contains NaN or infinity")
    frame_duration = float(payload["FrameDuration"])
    if not math.isfinite(frame_duration) or frame_duration <= 0:
        raise ValueError(f"{path}: invalid FrameDuration {frame_duration}")
    norms = np.linalg.norm(frames[:, 3:7], axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 1.0e-3:
        raise ValueError(f"{path}: quaternion norm error exceeds 1e-3")

    # Kinematic gate.  The AMP discriminator is fed joint angles alongside foot positions,
    # but on the policy side the feet are computed from the joints, so the two are identical
    # by construction.  A reference clip where they disagree hands the discriminator a
    # feature no policy can ever reproduce, and imitation silently degenerates.  Both the
    # Kine2Go joint-major bug and the NJU abduction flip were invisible for a full training
    # sweep; this check turns either of them into an immediate, named failure.
    joint_pos = frames[:, 7:19]
    stored_feet = frames[:, 19:31].reshape(len(frames), 4, 3)
    residual = np.linalg.norm(
        _go2_forward_kinematics(joint_pos).reshape(len(frames), 4, 3) - stored_feet, axis=2
    )
    fk_residual = float(residual.mean())
    if fk_residual > MAX_FK_RESIDUAL:
        raise ValueError(
            f"{path}: foot positions disagree with joint angles by {fk_residual:.3f} m "
            f"(limit {MAX_FK_RESIDUAL} m). The source block layout is being misread.")

    below = np.maximum(GO2_JOINT_LOWER - joint_pos, 0.0)
    above = np.maximum(joint_pos - GO2_JOINT_UPPER, 0.0)
    violation = float(np.max(below + above))
    if violation > 1.0e-6:
        worst = int(np.argmax((below + above).max(axis=0)))
        raise ValueError(
            f"{path}: joint {worst} exceeds the Go2 limit by {violation:.3f} rad. "
            f"The Go2 cannot reach this pose, so it cannot imitate it.")

    return {
        "dataset": path.parent.name,
        "file": path.name,
        "frames": len(frames),
        "fps": 1.0 / frame_duration,
        "fk_residual_m": round(fk_residual, 5),
        "max_joint_limit_violation_rad": round(violation, 6),
        "sha256": _sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("all", "mocap", "kine2go", "nju", "dogml"), default="all",
        help="Dataset to prepare.",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Convert existing files without downloading missing raw files.",
    )
    parser.add_argument(
        "--no-phase-screen", action="store_true",
        help="Measure DogML footfall phase and record it, but keep every clip.  Use this to "
             "inspect the score distribution before choosing rejection thresholds.",
    )
    args = parser.parse_args()

    output_paths: list[Path] = []
    if args.dataset in ("all", "mocap"):
        output_paths.extend(prepare_mocap_gaits())
    if args.dataset in ("all", "kine2go"):
        if not args.no_download:
            download_kine2go()
        kine2go_paths = [
            RAW_ROOT / "kine2go_gaits" / f"{clip}_{filename}"
            for clip in KINE2GO_GAIT_CLIPS
            for filename in ("motion.npy", "clip.json")
        ]
        _verify_raw_files(kine2go_paths)
        output_paths.extend(convert_kine2go())
    if args.dataset in ("all", "nju"):
        if not args.no_download:
            download_nju()
        _verify_raw_files([RAW_ROOT / "nju_agility" / filename for filename in NJU_FILES])
        output_paths.extend(convert_nju())
    if args.dataset in ("all", "dogml"):
        if not args.no_download:
            download_dogml()
        archive = RAW_ROOT / "dogml" / "dataset.zip"
        _verify_raw_files([archive])
        output_paths.extend(convert_dogml(enforce_phase_screen=not args.no_phase_screen))

    provenance_paths = []
    for dataset_name in ("mocap_gaits", "kine2go_gaits", "nju_agility", "dogml_gaits"):
        dataset_root = OUTPUT_ROOT / dataset_name
        if dataset_root.exists():
            provenance_paths.extend(
                path for path in sorted(dataset_root.iterdir())
                if path.is_file() and path.suffix in (".json", ".txt")
            )
    summaries = [validate_motion(path) for path in provenance_paths]
    provenance = OUTPUT_ROOT / "provenance.json"
    summaries.sort(key=lambda item: (item["dataset"], item["file"]))
    provenance.write_text(json.dumps(summaries, indent=2))
    print(
        f"Prepared {len(output_paths)} motions; provenance covers "
        f"{len(summaries)} motions: {provenance}"
    )


if __name__ == "__main__":
    main()
