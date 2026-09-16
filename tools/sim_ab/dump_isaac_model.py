#!/usr/bin/env python3
"""Dump the Isaac Lab Go2 plant parameters so MuJoCo can be aligned against numbers.

Writes a JSON with per-body mass / centre of mass / inertia and per-joint armature,
damping, friction, limits and actuator gains, all in the articulation's own body and
joint order plus names, so the MuJoCo side can match term by term.

Run under env_isaacsim:
    python tools/sim_ab/dump_isaac_model.py --headless --out /tmp/isaac_model.json
"""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Go2-AMP-Direct-v0")
parser.add_argument("--out", default="/tmp/isaac_model.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import go2_amp_isaaclab  # noqa: E402, F401
from go2_amp_isaaclab.wrappers import LegacyAmpVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def listify(x):
    if hasattr(x, "detach"):  # torch tensors live on the GPU here
        x = x.detach().cpu()
    return np.asarray(x).astype(float).tolist()


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.events = None
    # Without this, _reset_idx seeds a pose sampled from the reference motions, and the
    # "foot geometry at the default pose" measurement below silently reports that instead.
    cfg.reference_state_initialization = False
    cfg.add_observation_noise = False
    env = LegacyAmpVecEnvWrapper(gym.make(args.task, cfg=cfg))
    robot = env.env.robot
    out: dict = {}

    def grab(key: str, fn) -> None:
        """Collect one field, surviving fields this Isaac Lab build does not expose.

        The PhysX tensor views can abort the process rather than raise, so print before
        each call: if the run dies the last line names the field that killed it.
        """
        print(f"  reading {key} ...", flush=True)
        try:
            out[key] = listify(fn())
        except BaseException as exc:  # noqa: BLE001 - a missing field must not lose the rest
            out[key] = None
            print(f"    unavailable: {type(exc).__name__}: {exc}", flush=True)

    out["body_names"] = list(robot.body_names)
    out["joint_names"] = list(robot.joint_names)
    grab("masses", lambda: robot.root_physx_view.get_masses()[0])
    grab("coms", lambda: robot.root_physx_view.get_coms()[0])
    grab("inertias", lambda: robot.root_physx_view.get_inertias()[0])
    grab("joint_armature", lambda: robot.data.joint_armature[0])
    grab("joint_damping", lambda: robot.data.joint_damping[0])
    grab("joint_stiffness", lambda: robot.data.joint_stiffness[0])
    grab("joint_friction", lambda: robot.data.joint_friction[0])
    grab("joint_pos_limits", lambda: robot.data.joint_pos_limits[0])
    grab("soft_joint_pos_limits", lambda: robot.data.soft_joint_pos_limits[0])
    grab("joint_vel_limits", lambda: robot.data.joint_vel_limits[0])
    grab("joint_effort_limits", lambda: robot.data.joint_effort_limits[0])
    grab("default_joint_pos", lambda: robot.data.default_joint_pos[0])
    if out.get("masses"):
        out["total_mass"] = float(np.asarray(out["masses"]).sum())

    out.update({
        "sim_dt": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "ground_static_friction": float(cfg.sim.physics_material.static_friction),
        "ground_dynamic_friction": float(cfg.sim.physics_material.dynamic_friction),
        "ground_restitution": float(cfg.sim.physics_material.restitution),
        "solver_position_iterations": int(cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count),
        "solver_velocity_iterations": int(cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count),
    })

    # Write what we have before touching anything else, so a later crash still leaves a file.
    def flush_out() -> None:
        with open(args.out, "w") as handle:
            json.dump(out, handle, indent=2)

    flush_out()
    print(f"wrote {args.out}", flush=True)

    # Foot positions relative to the base at the default pose.  If the four contact points
    # are not coplanar the robot rocks on a diagonal, which is what makes a standing robot
    # creep and yaw.
    print("  reading foot geometry ...", flush=True)
    try:
        import torch

        env.reset()
        # Pin the articulation to its declared default pose; reset alone does not guarantee it.
        default = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(default, torch.zeros_like(default))
        env.env.sim.forward()
        foot_ids, foot_names = robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
        base_id = robot.find_bodies(["base"])[0][0]
        pos = robot.data.body_pos_w[0]
        out["foot_names"] = list(foot_names)
        out["foot_pos_rel_base_at_reset"] = listify(pos[foot_ids] - pos[base_id])
        flush_out()
    except BaseException as exc:  # noqa: BLE001
        print(f"    unavailable: {type(exc).__name__}: {exc}", flush=True)

    if out.get("total_mass"):
        print(f"  total mass {out['total_mass']:.4f} kg over {len(out['body_names'])} bodies")
    if out.get("coms") and "base" in out["body_names"]:
        print(f"  base COM   {np.round(out['coms'][out['body_names'].index('base')], 5)}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
