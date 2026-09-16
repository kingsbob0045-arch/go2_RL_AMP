#!/usr/bin/env python3
"""Close the loop on an exported SECAMP policy inside MuJoCo, with controllable latency.

The A/B harness showed the two plants agree once armature, damping and foot friction are
matched, which moves suspicion off the model and onto how the deployed stack runs the
controller.  mujoco_simulator.py recomputes the PD torque on a 1000 Hz ROS timer, steps
physics on a separate Python thread, and publishes state on a third timer, none of them
synchronised; Isaac holds the target for exactly 4 physics steps and recomputes the torque
every step.  This script runs the Isaac timing exactly, then adds whole control periods of
delay on demand, so the cost of that looseness can be measured instead of argued about.

Observation layout matches deploy_isaaclab.py / the exported sidecar (45 for secamp):
    projected_gravity(3) | command * command_scale(3) | (q - default) * dof_pos_scale(12)
    | qd * dof_vel_scale(12) | last_action(12) | skill one-hot(3)

Run under go2_ros_env:
    python tools/sim_ab/run_policy_mujoco.py --policy exports/go2_secamp.pt --skill 1 --latency 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P  # noqa: E402

ISAAC_MUJOCO = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
SKILL_NAMES = ("pace", "trot", "canter")


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_wxyz
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default=str(
        Path.home() / "Repo/ProjectsTest_Lingheng_Kong/go2_RL_AMP/exports/go2_secamp.pt"))
    ap.add_argument("--model", default=str(
        Path.home() / "Repo/ProjectsTest_Lingheng_Kong/go2_Deploy/resources/go2/scene_terrain.xml"))
    ap.add_argument("--skill", type=int, default=1, help="0 pace, 1 trot, 2 canter")
    ap.add_argument("--command", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                    help="Raw vx, vy, wz before command_scale")
    ap.add_argument("--command-scale", type=float, nargs=3, default=[0.5, 0.5, 1.0])
    ap.add_argument("--dof-pos-scale", type=float, default=1.0)
    ap.add_argument("--dof-vel-scale", type=float, default=0.05)
    ap.add_argument("--action-scale", type=float, default=0.25)
    ap.add_argument("--latency", type=int, default=0,
                    help="Whole control periods of delay between reading state and applying the action")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--armature", type=float, default=0.0)
    ap.add_argument("--damping", type=float, default=0.0)
    ap.add_argument("--foot-friction", type=float, default=1.0)
    ap.add_argument("--start-height", type=float, default=P.BASE_HEIGHT,
                    help="Base height the episode starts from; Isaac's env spawns at 0.38, "
                         "which drops the robot ~0.08 m onto its feet before the policy acts")
    ap.add_argument("--settle-steps", type=int, default=400,
                    help="Physics steps holding the nominal pose before the policy takes over; "
                         "0 reproduces Isaac, which hands over immediately after the spawn")
    ap.add_argument("--calf-mass", type=float, default=None,
                    help="Override each calf link mass (Isaac carries 0.154 + a 0.04 foot; "
                         "the MJCF lumps 0.2414 into the calf)")
    ap.add_argument("--impratio", type=float, default=None,
                    help="MuJoCo friction stiffness ratio; the model ships 100")
    ap.add_argument("--cone", choices=("elliptic", "pyramidal"), default=None)
    ap.add_argument("--video", default=None, help="Write an MP4 here")
    ap.add_argument("--out", default=None, help="Write the trajectory npz here")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    model.opt.timestep = P.PHYSICS_DT
    model.dof_armature[6:18] = args.armature
    model.dof_damping[6:18] = args.damping
    for i in range(model.ngeom):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "") in ("FL", "FR", "RL", "RR"):
            model.geom_friction[i, 0] = args.foot_friction
    if args.calf_mass is not None:
        for leg in ("FR", "FL", "RR", "RL"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
            model.body_mass[bid] = args.calf_mass
    if args.impratio is not None:
        model.opt.impratio = args.impratio
    if args.cone is not None:
        model.opt.cone = (mujoco.mjtCone.mjCONE_ELLIPTIC if args.cone == "elliptic"
                          else mujoco.mjtCone.mjCONE_PYRAMIDAL)
    data = mujoco.MjData(model)

    policy = torch.jit.load(args.policy, map_location="cpu").eval()
    sidecar = Path(args.policy).with_suffix(".json")
    meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    expected_obs = int(meta.get("observation_dim", 45))

    nominal_mj = P.DEFAULT_JOINT_POS[ISAAC_MUJOCO]
    effort_mj = P.EFFORT_LIMIT[ISAAC_MUJOCO]
    data.qpos[:] = 0.0
    data.qpos[2] = args.start_height
    data.qpos[3] = 1.0
    data.qpos[7:19] = nominal_mj
    mujoco.mj_forward(model, data)

    def drive(target_mj: np.ndarray) -> np.ndarray:
        tau = P.STIFFNESS * (target_mj - data.qpos[7:19]) - P.DAMPING * data.qvel[6:18]
        return np.clip(tau, -effort_mj, effort_mj)

    for _ in range(args.settle_steps):  # settle on the ground before handing over
        data.ctrl[:] = drive(nominal_mj)
        mujoco.mj_step(model, data)

    skill = np.zeros(3, dtype=np.float32)
    skill[args.skill] = 1.0
    command = np.asarray(args.command, dtype=np.float64) * np.asarray(args.command_scale)
    last_action = np.zeros(12, dtype=np.float32)
    # A queue of `latency` periods reproduces a stack that applies an action computed from
    # state it read some periods ago -- the defect a threaded, unsynchronised loop produces.
    pending = deque([P.DEFAULT_JOINT_POS.copy() for _ in range(args.latency + 1)],
                    maxlen=args.latency + 1)

    n = int(args.seconds / P.CONTROL_DT)
    log = {k: np.zeros((n, d)) for k, d in
           (("q", 12), ("base_pos", 3), ("base_quat", 4), ("base_vel", 3), ("action", 12))}
    renderer = frames = None
    if args.video:
        renderer = mujoco.Renderer(model, height=480, width=640)

    fell_at = None
    for i in range(n):
        q = data.qpos[7:19][ISAAC_MUJOCO]
        dq = data.qvel[6:18][ISAAC_MUJOCO]
        obs = np.concatenate((
            projected_gravity(data.qpos[3:7]),
            command,
            (q - P.DEFAULT_JOINT_POS) * args.dof_pos_scale,
            dq * args.dof_vel_scale,
            last_action,
            skill,
        )).astype(np.float32)
        if obs.shape[0] != expected_obs:
            raise SystemExit(f"observation is {obs.shape[0]} long, the export wants {expected_obs}")
        with torch.inference_mode():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        action = np.clip(action[:12], -100.0, 100.0)
        last_action = action.astype(np.float32)
        pending.append(P.DEFAULT_JOINT_POS + args.action_scale * action)

        target_mj = pending[0][ISAAC_MUJOCO]
        for _ in range(P.DECIMATION):
            data.ctrl[:] = drive(target_mj)
            mujoco.mj_step(model, data)

        log["q"][i] = data.qpos[7:19][ISAAC_MUJOCO]
        log["base_pos"][i] = data.qpos[0:3]
        log["base_quat"][i] = data.qpos[3:7]
        log["base_vel"][i] = data.qvel[0:3]
        log["action"][i] = action
        # projected gravity z goes toward 0 as the body tips past horizontal
        if fell_at is None and (data.qpos[2] < 0.15 or projected_gravity(data.qpos[3:7])[2] > -0.5):
            fell_at = i * P.CONTROL_DT
        if renderer is not None and i % 2 == 0:
            renderer.update_scene(data)
            if frames is None:
                frames = []
            frames.append(renderer.render().copy())

    travelled = log["base_pos"][-1, :2] - log["base_pos"][0, :2]
    mean_speed = float(np.linalg.norm(log["base_vel"][:, :2], axis=1).mean())
    print(f"latency {args.latency} period(s)  skill {SKILL_NAMES[args.skill]}  "
          f"command {args.command}")
    print(f"  fell            : {'no' if fell_at is None else f'yes at {fell_at:.2f} s'}")
    print(f"  travelled       : {travelled[0]:+.3f} m forward, {travelled[1]:+.3f} m lateral")
    print(f"  mean speed      : {mean_speed:.3f} m/s")
    print(f"  base height     : mean {log['base_pos'][:,2].mean():.3f} m, "
          f"min {log['base_pos'][:,2].min():.3f} m")
    print(f"  |action| mean   : {np.abs(log['action']).mean():.3f}, "
          f"max {np.abs(log['action']).max():.3f}")

    if args.out:
        np.savez(args.out, latency=args.latency, **log)
        print(f"  wrote {args.out}")
    if frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=25)
        print(f"  wrote {args.video}")


if __name__ == "__main__":
    main()
