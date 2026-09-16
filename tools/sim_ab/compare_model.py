#!/usr/bin/env python3
"""Compare the Isaac Lab plant dump against the MuJoCo model, term by term.

Run under go2_ros_env:
    python tools/sim_ab/compare_model.py /tmp/isaac_model.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_MJCF = Path.home() / "Repo/ProjectsTest_Lingheng_Kong/go2_Deploy/resources/go2/go2.xml"
# Isaac calls the trunk "base"; the MJCF calls it "base_link".
BODY_ALIAS = {"base": "base_link"}
ISAAC_MUJOCO = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("isaac_json")
    ap.add_argument("--mjcf", default=str(DEFAULT_MJCF))
    args = ap.parse_args()

    I = json.loads(Path(args.isaac_json).read_text())
    m = mujoco.MjModel.from_xml_path(args.mjcf)
    mj_body = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(m.nbody)}
    mj_joint = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)}

    print("=" * 92)
    print("LINK MASS AND CENTRE OF MASS")
    print("=" * 92)
    print(f"{'body':<14}{'isaac kg':>10}{'mujoco kg':>11}{'d kg':>9}   "
          f"{'isaac COM (m)':<28}{'mujoco COM (m)':<28}{'|dCOM| mm':>10}")
    mass_gap = 0.0
    com_gaps = []
    for k, name in enumerate(I["body_names"]):
        mn = BODY_ALIAS.get(name, name)
        if mn not in mj_body:
            print(f"{name:<14}{I['masses'][k]:>10.4f}{'--':>11}   (no MuJoCo body)")
            continue
        j = mj_body[mn]
        ic = np.array(I["coms"][k][:3])
        mc = np.array(m.body_ipos[j])
        dm = I["masses"][k] - m.body_mass[j]
        dc = np.linalg.norm(ic - mc) * 1000.0
        mass_gap += abs(dm)
        com_gaps.append(dc)
        print(f"{name:<14}{I['masses'][k]:>10.4f}{m.body_mass[j]:>11.4f}{dm:>+9.4f}   "
              f"{str(np.round(ic, 5)):<28}{str(np.round(mc, 5)):<28}{dc:>10.2f}")
    mj_total = float(m.body_mass.sum())
    print(f"\n  total mass   isaac {I['total_mass']:.4f} kg   mujoco {mj_total:.4f} kg   "
          f"diff {I['total_mass'] - mj_total:+.4f} kg")
    print(f"  summed |mass difference| {mass_gap:.4f} kg;  worst COM offset {max(com_gaps):.2f} mm")

    print("\n" + "=" * 92)
    print("JOINT PROPERTIES  (Isaac order FL, FR, RL, RR)")
    print("=" * 92)
    rows = [
        ("armature", "joint_armature", m.dof_armature[6:18]),
        ("damping", "joint_damping", m.dof_damping[6:18]),
        ("dry friction", "joint_friction", m.dof_frictionloss[6:18]),
    ]
    for label, key, mj_vals in rows:
        if I.get(key) is None:
            print(f"{label:<14} isaac: unavailable")
            continue
        iso = np.array(I[key])
        mj = np.array(mj_vals)[ISAAC_MUJOCO]
        flag = "OK" if np.allclose(iso, mj, atol=1e-6) else "MISMATCH"
        print(f"{label:<14} isaac {np.unique(np.round(iso,6))}   mujoco {np.unique(np.round(mj,6))}   {flag}")

    if I.get("joint_pos_limits") is not None:
        print("\nposition limits (rad):")
        lim = np.array(I["joint_pos_limits"])
        mj_lo = m.jnt_range[1:13, 0][ISAAC_MUJOCO]
        mj_hi = m.jnt_range[1:13, 1][ISAAC_MUJOCO]
        names = [n for n in I["joint_names"]]
        # Isaac joint order may differ from the protocol order; report by Isaac's own names.
        for i, n in enumerate(names):
            j = mj_joint.get(n)
            if j is None:
                continue
            mlo, mhi = m.jnt_range[j]
            same = abs(lim[i][0] - mlo) < 1e-4 and abs(lim[i][1] - mhi) < 1e-4
            if not same:
                print(f"  {n:<16} isaac [{lim[i][0]:+.4f}, {lim[i][1]:+.4f}]   "
                      f"mujoco [{mlo:+.4f}, {mhi:+.4f}]   MISMATCH")
        print("  (only mismatches listed)")

    print("\n" + "=" * 92)
    print("CONTACT AND SOLVER")
    print("=" * 92)
    foot = [i for i in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "") in ("FL", "FR", "RL", "RR")]
    print(f"  ground friction   isaac static {I['ground_static_friction']:.3f} / dynamic "
          f"{I['ground_dynamic_friction']:.3f}   mujoco foot geom {m.geom_friction[foot[0], 0]:.3f} "
          f"(priority {m.geom_priority[foot[0]]}, so it overrides the floor)")
    print(f"  restitution       isaac {I['ground_restitution']:.3f}")
    print(f"  solver iterations isaac position {I['solver_position_iterations']} / "
          f"velocity {I['solver_velocity_iterations']}")
    print(f"  timestep          isaac {I['sim_dt']} x decimation {I['decimation']}")

    if I.get("foot_pos_rel_base_at_reset") is not None:
        f = np.array(I["foot_pos_rel_base_at_reset"])
        print(f"\n  Isaac foot heights at the default pose (m): {np.round(f[:, 2], 5)}")
        print(f"    spread {1000*(f[:,2].max()-f[:,2].min()):.2f} mm -- a non-flat contact set "
              f"makes a standing robot rock on one diagonal")


if __name__ == "__main__":
    main()
