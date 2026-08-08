"""Visualize a single motion file on the Go2 robot.

Usage (run from repo root):
  python legged_gym/scripts/visualize_motion.py \
      --motion_file datasets/mocap_motions_go2/trot0.txt \
      --num_loops 5 --slow 1.0
"""

import argparse
import json
import time
import os

import isaacgym  # must be imported before torch
from isaacgym import gymapi, gymtorch
import numpy as np
import torch

LEGGED_GYM_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
URDF_REL = "resources/robots/go2/urdf/go2.urdf"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file",
                        default="datasets/mocap_motions_go2/trot0.txt")
    parser.add_argument("--num_loops", type=int, default=5)
    parser.add_argument("--slow", type=float, default=1.0,
                        help="Sleep multiplier on frame_duration (>1 = slower)")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load motion frames
    # ------------------------------------------------------------------ #
    with open(args.motion_file) as f:
        data = json.load(f)
    frames = np.array(data["Frames"], dtype=np.float32)
    frame_duration = float(data["FrameDuration"])
    num_frames = frames.shape[0]
    print(f"Loaded {num_frames} frames @ {frame_duration:.4f}s each "
          f"from {args.motion_file}")

    # Frame layout: [root_pos(3), root_quat_xyzw(4), joints(12), ...]
    root_pos   = frames[:, 0:3]
    root_quat  = frames[:, 3:7]   # xyzw
    raw_joints = frames[:, 7:19]

    joints = raw_joints

    # ------------------------------------------------------------------ #
    # IsaacGym setup
    # ------------------------------------------------------------------ #
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = frame_duration
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    urdf_abs = os.path.join(LEGGED_GYM_ROOT_DIR, URDF_REL)
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset_options.collapse_fixed_joints = True
    asset_options.replace_cylinder_with_capsule = True
    asset_options.flip_visual_attachments = True
    asset_options.fix_base_link = False
    asset_options.disable_gravity = False

    robot_asset = gym.load_asset(
        sim, os.path.dirname(urdf_abs), os.path.basename(urdf_abs), asset_options)
    num_dofs = gym.get_asset_dof_count(robot_asset)
    print(f"Asset loaded: {num_dofs} DOFs")

    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    gym.create_actor(env, robot_asset, pose, "robot", 0, 1, 0)

    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)
    gym.viewer_camera_look_at(
        viewer, None,
        gymapi.Vec3(2.0, -3.0, 1.2),
        gymapi.Vec3(0.0,  0.0, 0.5))

    gym.prepare_sim(sim)

    dof_state_tensor  = gym.acquire_dof_state_tensor(sim)
    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)

    dof_states  = gymtorch.wrap_tensor(dof_state_tensor)   # (12, 2)
    root_states = gymtorch.wrap_tensor(root_state_tensor)  # (1, 13)

    actor_ids = torch.tensor([0], dtype=torch.int32)

    print(f"Playing {args.num_loops} loops. Press Escape to quit.")


    # Camera offset relative to robot (side-rear view)
    cam_offset = np.array([2.0, -3.0, 1.0])   # dx, dy, dz from robot

    sleep_dt = frame_duration * args.slow

    for _ in range(args.num_loops):
        for fi in range(num_frames):
            if gym.query_viewer_has_closed(viewer):
                break

            robot_x, robot_y = float(root_pos[fi, 0]), float(root_pos[fi, 1])
            root_states[0, 0] = robot_x
            root_states[0, 1] = robot_y
            root_states[0, 2] = float(root_pos[fi, 2])
            root_states[0, 3] = float(root_quat[fi, 0])  # x
            root_states[0, 4] = float(root_quat[fi, 1])  # y
            root_states[0, 5] = float(root_quat[fi, 2])  # z
            root_states[0, 6] = float(root_quat[fi, 3])  # w
            root_states[0, 7:13] = 0.0

            gym.set_actor_root_state_tensor_indexed(
                sim,
                gymtorch.unwrap_tensor(root_states),
                gymtorch.unwrap_tensor(actor_ids), 1)

            for dof_i in range(num_dofs):
                dof_states[dof_i, 0] = float(joints[fi, dof_i])
                dof_states[dof_i, 1] = 0.0

            gym.set_dof_state_tensor_indexed(
                sim,
                gymtorch.unwrap_tensor(dof_states),
                gymtorch.unwrap_tensor(actor_ids), 1)

            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)

            # Camera follow
            robot_pos = np.array([robot_x, robot_y, float(root_pos[fi, 2])])
            cam_pos    = robot_pos + cam_offset
            cam_target = robot_pos + np.array([0.0, 0.0, 0.3])
            gym.viewer_camera_look_at(
                viewer, None,
                gymapi.Vec3(*cam_pos),
                gymapi.Vec3(*cam_target))

            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
            time.sleep(sleep_dt)

        if gym.query_viewer_has_closed(viewer):
            break

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Done.")


if __name__ == "__main__":
    main()
