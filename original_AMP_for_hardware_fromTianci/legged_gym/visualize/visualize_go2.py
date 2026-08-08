# Visualize the GO2 robot URDF in Isaac Gym with a default standing pose.
# No RL environment is needed — this script uses the Isaac Gym API directly.
#
# Usage:
#   python legged_gym/scripts/viz/visualize_go2.py
#   ESC to quit

import sys
import numpy as np

import isaacgym  # must be imported before torch
from isaacgym import gymapi, gymutil, gymtorch
import torch

from rich import print

sys.path.insert(0, __import__('os').path.join(
    __import__('os').path.dirname(__file__), '..', '..', '..'))
from legged_gym import LEGGED_GYM_ROOT_DIR


# Default standing pose — derived from GO2 URDF joint limits.
# Hip: ±0.1 rad splay; thigh: ~0.8–1.0 rad; calf: -1.5 rad.
DEFAULT_JOINT_ANGLES = {
    'FL_hip_joint':   0.1,   'FR_hip_joint':  -0.1,
    'RL_hip_joint':   0.1,   'RR_hip_joint':  -0.1,
    'FL_thigh_joint': 0.8,   'FR_thigh_joint': 0.8,
    'RL_thigh_joint': 1.0,   'RR_thigh_joint': 1.0,
    'FL_calf_joint': -1.5,   'FR_calf_joint': -1.5,
    'RL_calf_joint': -1.5,   'RR_calf_joint': -1.5,
}

STANDING_HEIGHT = 0.38   # metres, approximate GO2 standing base height


def create_sim(gym):
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0., 0., -9.81)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation.")
    return sim


def load_go2_asset(gym, sim):
    asset_root = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf"
    asset_file = "go2.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_options.collapse_fixed_joints = True
    asset_options.replace_cylinder_with_capsule = True
    asset_options.flip_visual_attachments = True  # .dae meshes are y-up
    asset_options.fix_base_link = False
    asset_options.disable_gravity = True          # kinematic hold — no falling
    asset_options.density = 0.001
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.armature = 0.0
    asset_options.thickness = 0.01

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if asset is None:
        raise RuntimeError(f"Failed to load GO2 URDF from {asset_root}/{asset_file}")
    return asset


def main():
    gym = gymapi.acquire_gym()
    sim = create_sim(gym)

    # Ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0., 0., 1.)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0
    gym.add_ground(sim, plane_params)

    # Load asset and inspect DOFs
    robot_asset = load_go2_asset(gym, sim)
    num_dof = gym.get_asset_dof_count(robot_asset)
    dof_names = gym.get_asset_dof_names(robot_asset)

    print(f"\n=== GO2 URDF Visualizer ===")
    print(f"Asset: {LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf")
    print(f"DOF count: {num_dof}")
    print("DOF name → default angle (rad):")
    default_pos = np.zeros(num_dof, dtype=np.float32)
    for i, name in enumerate(dof_names):
        default_pos[i] = DEFAULT_JOINT_ANGLES.get(name, 0.0)
        print(f"  [{i:2d}] {name:25s} = {default_pos[i]:+.3f}")

    # Create environment and actor
    env = gym.create_env(sim,
                         gymapi.Vec3(-1., -1., 0.),
                         gymapi.Vec3( 1.,  1., 2.), 1)
    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(0., 0., STANDING_HEIGHT)
    start_pose.r = gymapi.Quat(0., 0., 0., 1.)
    actor = gym.create_actor(env, robot_asset, start_pose, "go2", 0, 1)

    gym.prepare_sim(sim)

    # Acquire state tensors
    actor_root_tensor = gym.acquire_actor_root_state_tensor(sim)
    dof_state_tensor  = gym.acquire_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    root_states = gymtorch.wrap_tensor(actor_root_tensor)  # [1, 13]
    dof_state   = gymtorch.wrap_tensor(dof_state_tensor)   # [num_dof, 2]
    actor_idx   = torch.tensor([0], dtype=torch.int32)

    default_pos_t = torch.from_numpy(default_pos)

    def set_standing_pose():
        """Force the robot into the default standing pose every frame."""
        root_states[0, 0] = 0.; root_states[0, 1] = 0.; root_states[0, 2] = STANDING_HEIGHT
        root_states[0, 3] = 0.; root_states[0, 4] = 0.
        root_states[0, 5] = 0.; root_states[0, 6] = 1.
        root_states[0, 7:] = 0.
        dof_state[:, 0] = default_pos_t
        dof_state[:, 1] = 0.
        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(actor_idx), 1)
        gym.set_dof_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(dof_state),
            gymtorch.unwrap_tensor(actor_idx), 1)

    # Create viewer
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)
    if viewer is None:
        raise RuntimeError("Failed to create viewer. Is a display available?")

    gym.viewer_camera_look_at(
        viewer, None,
        gymapi.Vec3(2.0, 1.5, 1.0),   # camera position
        gymapi.Vec3(0.0, 0.0, 0.3))    # look-at target

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")

    print("\nVisualization running — press ESC to quit.\n")

    while not gym.query_viewer_has_closed(viewer):
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "QUIT" and evt.value > 0:
                gym.destroy_viewer(viewer)
                gym.destroy_sim(sim)
                return

        set_standing_pose()

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
