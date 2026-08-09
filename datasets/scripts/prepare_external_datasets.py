#!/usr/bin/env python3
"""Download, convert, and validate external motions for SECAMP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path

import numpy as np

DATASETS_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = DATASETS_ROOT / "external_raw"
OUTPUT_ROOT = DATASETS_ROOT / "converted"
CHECKSUM_FILE = DATASETS_ROOT / "external_checksums.sha256"

KINE2GO_REVISION = "3698e74f8f9e889697cafc872fb32dd5e8285fb7"
KINE2GO_CLIPS = (
    "ai4_dog_canter",
    "ai4_dog_right_turn",
    "ai4_dog_synth_circle_walk",
    "ai4_dog_synth_half_flip_jump",
    "solo8_crawl_fast",
    "solo8_jump_forward_a",
    "vhdc_horse1_s1_trot_01",
    "vhdc_horse1_s1_walk_01",
)
NJU_FILES = (
    "canter_0.json", "canter_1.json", "canter_2.json", "canter_3.json",
    "pace_0.json", "pace_1.json", "pace_2.json",
    "trot_0.json", "trot_1.json", "trot_2.json",
)

LEG_REORDER_FR_FL_RR_RL = (1, 0, 3, 2)
FOOT_REORDER_FL_RL_FR_RR = (0, 2, 1, 3)


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
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {relative_path}: expected {expected}, got {actual}"
            )


def download_kine2go() -> None:
    root = RAW_ROOT / "kine2go_review"
    base = (
        "https://huggingface.co/datasets/kine2go-review/kine2go/resolve/"
        f"{KINE2GO_REVISION}/review_sample/data"
    )
    for clip in KINE2GO_CLIPS:
        _download(f"{base}/{clip}/motion.npy", root / f"{clip}.npy")
        _download(f"{base}/{clip}/clip.json", root / f"{clip}.json")


def download_nju() -> None:
    root = RAW_ROOT / "nju_agility"
    base = (
        "https://raw.githubusercontent.com/NJU-RLC/quadrupedal-agility/"
        "15d16ea99bc23e5d401b0b0e3b88a8edcc28f0ed/"
        "bbc/mocap_data/mocap_all_lb"
    )
    for filename in NJU_FILES:
        _download(f"{base}/{filename}", root / filename)


def _reorder_legs(values: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    return values.reshape(len(values), 4, 3)[:, order, :].reshape(len(values), 12)


def _quat_rotate_inverse_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = -quaternion[:, :3]
    w = quaternion[:, 3:4]
    cross = np.cross(xyz, vector)
    return vector + 2.0 * (w * cross + np.cross(xyz, cross))


def _quat_wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
    return quaternion[:, (1, 2, 3, 0)]


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


def convert_kine2go() -> list[Path]:
    raw_root = RAW_ROOT / "kine2go_review"
    output_root = OUTPUT_ROOT / "kine2go_review"
    if output_root.exists():
        for stale_path in output_root.glob("*.json"):
            stale_path.unlink()
    output_paths: list[Path] = []
    skill_counts = {"pace": 0, "trot": 0, "canter": 0}

    for clip in KINE2GO_CLIPS:
        metadata = json.loads((raw_root / f"{clip}.json").read_text())
        source = np.load(raw_root / f"{clip}.npy")
        if source.ndim != 2 or source.shape[1] != 61:
            raise ValueError(f"{clip}: expected motion.npy shape (T, 61), got {source.shape}")

        tags = set(metadata["tags"])
        skill = next((name for name in skill_counts if name in tags), "turn")
        if skill == "turn":
            print(f"Skipping {clip}: reviewer clip has no SECAMP pace/trot/canter label")
            continue
        skill_counts[skill] += 1
        destination = output_root / f"{skill}_{clip}.json"

        root_pos = source[:, 48:51]
        root_quat = _quat_wxyz_to_xyzw(source[:, 51:55])
        joint_pos = _reorder_legs(source[:, 6:18], LEG_REORDER_FR_FL_RR_RL)
        joint_vel = _reorder_legs(source[:, 24:36], LEG_REORDER_FR_FL_RR_RL)
        foot_world = _reorder_legs(source[:, 36:48], FOOT_REORDER_FL_RL_FR_RR)
        foot_local = _quat_rotate_inverse_xyzw(
            np.repeat(root_quat, 4, axis=0),
            (foot_world.reshape(-1, 3) - np.repeat(root_pos, 4, axis=0)),
        ).reshape(len(source), 12)
        lin_vel = _quat_rotate_inverse_xyzw(root_quat, source[:, 55:58])
        ang_vel = _quat_rotate_inverse_xyzw(root_quat, source[:, 58:61])
        frame_duration = 1.0 / float(metadata["fps"])
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
        joint_pos = _reorder_legs(source[:, 7:19], LEG_REORDER_FR_FL_RR_RL)
        joint_vel = _reorder_legs(source[:, 37:49], LEG_REORDER_FR_FL_RR_RL)
        foot_world = _reorder_legs(source[:, 19:31], LEG_REORDER_FR_FL_RR_RL)

        # NJU's PyBullet abduction convention is opposite Isaac's for all hips.
        joint_pos[:, (0, 3, 6, 9)] *= -1.0
        joint_vel[:, (0, 3, 6, 9)] *= -1.0
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
    return {
        "file": path.name,
        "frames": len(frames),
        "fps": 1.0 / frame_duration,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("all", "kine2go", "nju"), default="all",
        help="Dataset to prepare.",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Convert existing files without downloading missing raw files.",
    )
    args = parser.parse_args()

    output_paths: list[Path] = []
    if args.dataset in ("all", "kine2go"):
        if not args.no_download:
            download_kine2go()
        kine2go_paths = [
            RAW_ROOT / "kine2go_review" / f"{clip}.{extension}"
            for clip in KINE2GO_CLIPS
            for extension in ("npy", "json")
        ]
        _verify_raw_files(kine2go_paths)
        output_paths.extend(convert_kine2go())
    if args.dataset in ("all", "nju"):
        if not args.no_download:
            download_nju()
        _verify_raw_files([RAW_ROOT / "nju_agility" / filename for filename in NJU_FILES])
        output_paths.extend(convert_nju())

    summaries = [validate_motion(path) for path in output_paths]
    provenance = OUTPUT_ROOT / "provenance.json"
    provenance.write_text(json.dumps(summaries, indent=2))
    print(f"Validated {len(summaries)} converted motions; provenance: {provenance}")


if __name__ == "__main__":
    main()
