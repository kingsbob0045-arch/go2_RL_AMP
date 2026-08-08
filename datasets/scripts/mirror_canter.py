"""
Mirror canter motion files left-right (XZ-plane reflection) for data augmentation.

Frame layout (61 columns):
  [0:3]   root_pos  (x, y, z)
  [3:7]   root_rot  quaternion (x, y, z, w)
  [7:19]  joint_pos (FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
                     RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf)
  [19:31] toe_pos_local  (same leg order, xyz per foot)
  [31:34] linear_vel (vx, vy, vz)
  [34:37] angular_vel (wx, wy, wz)
  [37:49] joint_vel  (same leg order)
  [49:61] toe_vel_local  (same leg order, xyz per foot)

Mirror rules (reflection about XZ plane, y → -y):
  root_pos:    negate y
  root_rot:    (x,y,z,w) → (-x, y, -z, w)
  joint_pos/vel: swap FL↔FR, RL↔RR; negate hip joints
  toe_pos/vel_local: swap FL↔FR, RL↔RR; negate y of each foot
  linear_vel:  negate vy
  angular_vel: negate wx and wz  (pseudovector under XZ reflection)
"""
import json
import copy
import numpy as np
import glob
import os

# Joint permutation: swap FL(0-2)↔FR(3-5) and RL(6-8)↔RR(9-11)
JOINT_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
# After permutation, hip joints sit at indices 0, 3, 6, 9 — negate them
HIP_SIGN = np.array([-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1], dtype=float)

# Foot permutation: swap FL(0-2)↔FR(3-5) and RL(6-8)↔RR(9-11)
FOOT_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
# Negate y of each foot (indices 1, 4, 7, 10 within the 12-element block)
FOOT_Y_SIGN = np.array([1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1], dtype=float)


def mirror_frame(frame: np.ndarray) -> np.ndarray:
    f = frame.copy()

    # root_pos: negate y
    f[1] *= -1

    # root_rot (x,y,z,w): negate x and z
    f[3] *= -1
    f[5] *= -1

    # joint_pos
    jp = frame[7:19][JOINT_PERM] * HIP_SIGN
    f[7:19] = jp

    # toe_pos_local
    tp = frame[19:31][FOOT_PERM] * FOOT_Y_SIGN
    f[19:31] = tp

    # linear_vel: negate vy
    f[32] *= -1

    # angular_vel: negate wx and wz
    f[34] *= -1
    f[36] *= -1

    # joint_vel
    jv = frame[37:49][JOINT_PERM] * HIP_SIGN
    f[37:49] = jv

    # toe_vel_local
    tv = frame[49:61][FOOT_PERM] * FOOT_Y_SIGN
    f[49:61] = tv

    return f


def mirror_motion_file(src_path: str, dst_path: str):
    with open(src_path) as fh:
        data = json.load(fh)

    frames = np.array(data["Frames"])
    mirrored = np.stack([mirror_frame(frames[i]) for i in range(len(frames))])

    out = copy.deepcopy(data)
    out["Frames"] = mirrored.tolist()

    with open(dst_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  {src_path} → {dst_path}")


if __name__ == "__main__":
    camp_dir = os.path.dirname(os.path.abspath(__file__))
    src_files = sorted(glob.glob(os.path.join(camp_dir, "canter[0-9].txt")))

    # Find next available index for augmented files
    existing = glob.glob(os.path.join(camp_dir, "canter[0-9].txt"))
    used_indices = {int(os.path.basename(p)[len("canter"):-len(".txt")])
                    for p in existing}
    next_idx = max(used_indices) + 1

    print(f"Mirroring {len(src_files)} canter files starting at index {next_idx}:")
    for i, src in enumerate(src_files):
        dst = os.path.join(camp_dir, f"canter{next_idx + i}.txt")
        mirror_motion_file(src, dst)

    print("Done.")
