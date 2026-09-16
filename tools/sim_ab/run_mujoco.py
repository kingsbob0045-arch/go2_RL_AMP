#!/usr/bin/env python3
"""Run the shared A/B protocol in MuJoCo and save the trajectory.

Mirrors Go2AmpEnv's control loop exactly: the joint target is held for DECIMATION
physics steps and the PD torque is recomputed every physics step from the current
state, then clipped per joint -- which is what IdealPDActuator does inside Isaac.

Run under go2_ros_env (the interpreter that has mujoco):
    python tools/sim_ab/run_mujoco.py --out /tmp/ab_mujoco.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P  # noqa: E402

# MuJoCo qpos/qvel run FR, FL, RR, RL; the protocol is Isaac FL, FR, RL, RR.
# The permutation is self-inverse, so the same array converts both ways.
ISAAC_MUJOCO = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(
        Path.home() / "Repo/ProjectsTest_Lingheng_Kong/go2_Deploy/resources/go2/scene_terrain.xml"))
    parser.add_argument("--out", default="/tmp/ab_mujoco.npz")
    parser.add_argument("--armature", type=float, default=None, help="Override every joint's armature")
    parser.add_argument("--damping", type=float, default=None, help="Override every joint's damping")
    parser.add_argument("--frictionloss", type=float, default=None, help="Override every joint's dry friction")
    parser.add_argument("--foot-friction", type=float, default=None, help="Override the foot sliding friction")
    parser.add_argument("--settle-steps", type=int, default=400,
                        help="Physics steps of nominal hold before logging, to reach static equilibrium")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    model.opt.timestep = P.PHYSICS_DT

    leg_joints = np.arange(1, 13)  # joint 0 is the free root
    if args.armature is not None:
        model.dof_armature[6:18] = args.armature
    if args.damping is not None:
        model.dof_damping[6:18] = args.damping
    if args.frictionloss is not None:
        model.dof_frictionloss[6:18] = args.frictionloss
    if args.foot_friction is not None:
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if name in ("FL", "FR", "RL", "RR"):
                model.geom_friction[i, 0] = args.foot_friction

    data = mujoco.MjData(model)
    t, targets = P.build_protocol()

    data.qpos[:] = 0.0
    data.qpos[2] = P.BASE_HEIGHT
    data.qpos[3] = 1.0
    data.qpos[7:19] = P.DEFAULT_JOINT_POS[ISAAC_MUJOCO]
    mujoco.mj_forward(model, data)

    effort_mj = P.EFFORT_LIMIT[ISAAC_MUJOCO]
    nominal_mj = P.DEFAULT_JOINT_POS[ISAAC_MUJOCO]

    def drive(target_mj: np.ndarray) -> np.ndarray:
        tau = P.STIFFNESS * (target_mj - data.qpos[7:19]) - P.DAMPING * data.qvel[6:18]
        return np.clip(tau, -effort_mj, effort_mj)

    for _ in range(args.settle_steps):
        data.ctrl[:] = drive(nominal_mj)
        mujoco.mj_step(model, data)

    n = len(t)
    log = {k: np.zeros((n, d)) for k, d in
           (("q", 12), ("qd", 12), ("tau", 12), ("base_pos", 3), ("base_quat", 4))}
    contact = np.zeros((n, 4))
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("FL", "FR", "RL", "RR")]

    for i in range(n):
        target_mj = targets[i][ISAAC_MUJOCO]
        for _ in range(P.DECIMATION):
            tau = drive(target_mj)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
        log["q"][i] = data.qpos[7:19][ISAAC_MUJOCO]
        log["qd"][i] = data.qvel[6:18][ISAAC_MUJOCO]
        log["tau"][i] = tau[ISAAC_MUJOCO]
        log["base_pos"][i] = data.qpos[0:3]
        log["base_quat"][i] = data.qpos[3:7]
        forces = np.zeros(4)
        for c in range(data.ncon):
            con = data.contact[c]
            for leg, gid in enumerate(foot_ids):
                if gid in (con.geom1, con.geom2):
                    buf = np.zeros(6)
                    mujoco.mj_contactForce(model, data, c, buf)
                    forces[leg] += abs(buf[0])
        contact[i] = forces

    np.savez(args.out, t=t, target=targets, contact=contact,
             engine="mujoco", **log)
    print(f"wrote {args.out}  ({n} control steps, {t[-1]:.2f} s)")
    print(f"  armature={model.dof_armature[6]:.4f} damping={model.dof_damping[6]:.4f} "
          f"frictionloss={model.dof_frictionloss[6]:.4f} foot_friction={model.geom_friction[foot_ids[0], 0]:.3f}")
    print(f"  final base height {data.qpos[2]:.4f} m")


if __name__ == "__main__":
    main()
