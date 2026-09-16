#!/usr/bin/env python3
"""Run the shared A/B protocol in Isaac Lab and save the trajectory.

Drives the real training environment rather than a bare articulation, so the actuator
model, decimation, solver settings and contact material are exactly the ones the policy
was trained against.  Targets are realised through the normal action path:
    _joint_targets = default_joint_pos + action_scale * action
so the requested action is (target - default) / action_scale.

Run under env_isaacsim:
    python tools/sim_ab/run_isaac.py --headless --out /tmp/ab_isaac.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Go2-AMP-Direct-v0")
parser.add_argument("--out", default="/tmp/ab_isaac.npz")
parser.add_argument("--settle-steps", type=int, default=100,
                    help="Control steps of nominal hold before logging, to reach static equilibrium")
parser.add_argument("--velocity-iterations", type=int, default=None,
                    help="Override solver_velocity_iteration_count (UNITREE_GO2_CFG ships 0, which "
                         "PhysX warns produces noisy velocities)")
parser.add_argument("--position-iterations", type=int, default=None,
                    help="Override solver_position_iteration_count (ships 4)")
parser.add_argument("--friction", type=float, default=None,
                    help="Override the ground static/dynamic friction (cfg ships 1.0)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import go2_amp_isaaclab  # noqa: E402, F401
from go2_amp_isaaclab.wrappers import LegacyAmpVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    # Strip everything stochastic: this run measures the nominal plant, not the
    # randomised one, because that is what the MuJoCo model represents.
    cfg.add_observation_noise = False
    cfg.reference_state_initialization = False
    cfg.events = None
    # Physics is unaffected by termination, but a reset mid-protocol would ruin the trace.
    cfg.termination_contact_force = 1.0e9
    cfg.episode_length_s = 10_000.0
    if args.velocity_iterations is not None:
        cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iterations
    if args.position_iterations is not None:
        cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = args.position_iterations
    if args.friction is not None:
        cfg.sim.physics_material.static_friction = args.friction
        cfg.sim.physics_material.dynamic_friction = args.friction
        cfg.terrain.physics_material = cfg.sim.physics_material

    env = LegacyAmpVecEnvWrapper(gym.make(args.task, cfg=cfg))
    inner = env.env
    robot = inner.robot
    device = inner.device

    t, targets = P.build_protocol()
    default = inner.default_joint_pos[0].detach().cpu().numpy().astype(np.float64)
    if not np.allclose(default, P.DEFAULT_JOINT_POS, atol=1.0e-6):
        raise RuntimeError(
            f"Isaac default joint pos {default} disagrees with protocol.DEFAULT_JOINT_POS "
            f"{P.DEFAULT_JOINT_POS}; the A/B comparison would be meaningless.")

    scale = float(inner.cfg.action_scale)
    ids = inner._joint_ids
    foot_ids, _ = inner.contact_sensor.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"])

    def action_for(target: np.ndarray) -> torch.Tensor:
        a = (target - P.DEFAULT_JOINT_POS) / scale
        return torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)

    joint_pos = torch.tensor(P.DEFAULT_JOINT_POS, dtype=torch.float32, device=device).unsqueeze(0)

    def pin_root(height: float) -> None:
        """_reset_idx seeds the root twist with sample_uniform(-0.5, 0.5); left in place
        that spins the robot several degrees before friction absorbs it, which is enough
        to ruin a static comparison.  Re-pin the root explicitly instead."""
        root = robot.data.default_root_state.clone()
        root[:, 0:3] = torch.tensor([0.0, 0.0, height], device=device)
        root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        root[:, 7:13] = 0.0
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))

    def report(tag: str) -> None:
        lin = robot.data.root_lin_vel_w[0].detach().cpu().numpy()
        ang = robot.data.root_ang_vel_w[0].detach().cpu().numpy()
        pos = robot.data.root_pos_w[0].detach().cpu().numpy()
        quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        yaw = np.degrees(np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                                    1 - 2 * (quat[2] ** 2 + quat[3] ** 2)))
        print(f"  [{tag:<18}] pos {np.round(pos, 4)} yaw {yaw:+7.2f} deg "
              f"|v| {np.linalg.norm(lin):.4f} |w| {np.linalg.norm(ang):.4f}")

    env.reset()
    report("after reset")
    hold = action_for(P.DEFAULT_JOINT_POS)

    # Two-stage settle: pin, let contact establish, pin again to clear whatever the first
    # contact transient injected, then settle undisturbed.
    for stage in range(2):
        pin_root(P.BASE_HEIGHT)
        report(f"pinned (stage {stage})")
        for _ in range(args.settle_steps // 2):
            env.step(hold)
        report(f"settled (stage {stage})")

    n = len(t)
    log = {k: np.zeros((n, d)) for k, d in
           (("q", 12), ("qd", 12), ("tau", 12), ("base_pos", 3), ("base_quat", 4))}
    contact = np.zeros((n, 4))

    for i in range(n):
        env.step(action_for(targets[i]))
        log["q"][i] = robot.data.joint_pos[0, ids].detach().cpu().numpy()
        log["qd"][i] = robot.data.joint_vel[0, ids].detach().cpu().numpy()
        log["tau"][i] = robot.data.applied_torque[0, ids].detach().cpu().numpy()
        log["base_pos"][i] = robot.data.root_pos_w[0].detach().cpu().numpy()
        log["base_quat"][i] = robot.data.root_quat_w[0].detach().cpu().numpy()
        forces = inner.contact_sensor.data.net_forces_w[0, foot_ids]
        contact[i] = torch.norm(forces, dim=-1).detach().cpu().numpy()

    # The scene origin is not the world origin when env_spacing is applied.
    log["base_pos"] -= inner.terrain.env_origins[0].detach().cpu().numpy()

    np.savez(args.out, t=t, target=targets, contact=contact, engine="isaac", **log)
    print(f"wrote {args.out}  ({n} control steps, {t[-1]:.2f} s)")
    print(f"  final base height {log['base_pos'][-1, 2]:.4f} m")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
