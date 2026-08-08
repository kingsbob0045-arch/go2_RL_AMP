"""Isaac Lab configurations preserving the legacy sim-to-real contract."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfPyramidSlopedTerrainCfg
from isaaclab.terrains.trimesh import MeshRandomGridTerrainCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


GO2_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)


def _go2_cfg() -> ArticulationCfg:
    cfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cfg.init_state.pos = (0.0, 0.0, 0.38)
    cfg.actuators["base_legs"].stiffness = 50.0
    cfg.actuators["base_legs"].damping = 1.0
    return cfg


@configclass
class DomainRandomizationCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.25, 1.75),
            "dynamic_friction_range": (0.25, 1.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(15.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class Go2AmpEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_space = 12
    observation_space = 42
    state_space = 48
    action_scale = 0.25
    observation_clip = 100.0

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True, clone_in_fabric=False,
    )
    robot_cfg: ArticulationCfg = _go2_cfg()
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True,
    )
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        num_envs=scene.num_envs, env_spacing=scene.env_spacing,
        physics_material=sim.physics_material,
    )
    events: DomainRandomizationCfg = DomainRandomizationCfg()

    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    command_scale = (2.0, 2.0, 0.25)
    command_ranges = ((0.0, 3.5), (-0.3, 0.3), (-1.57, 1.57))
    command_resampling_time_s = 10.0
    tracking_lin_vel_scale = 1.5
    tracking_ang_vel_scale = 0.5
    tracking_sigma = 0.25
    termination_contact_force = 1.0

    add_observation_noise = True
    gravity_noise = 0.05
    dof_pos_noise = 0.03
    dof_vel_noise = 1.5
    motion_glob = "datasets/mocap_motions_go2/*"
    reference_state_initialization = True
    reference_state_initialization_prob = 0.85
    amp_horizon = 1
    skill_dim = 0
    waypoint_mode = False
    history_steps = 1
    residual_policy = False
    motion_prior = ""
    residual_action_scale = 0.1


@configclass
class Go2SecampEnvCfg(Go2AmpEnvCfg):
    observation_space = 45
    state_space = 51
    motion_glob = "datasets/camp/*"
    amp_horizon = 2
    skill_dim = 3
    waypoint_mode = True
    command_scale = (0.5, 0.5, 1.0)
    tracking_lin_vel_scale = 0.0
    tracking_ang_vel_scale = 0.0
    tracking_position_scale = 2.0
    skill_speed_limits = (1.0, 1.5, 3.0)
    waypoint_distance_range = (5.0, 15.0)
    waypoint_arrival_threshold = 0.1
    waypoint_max_turn_angle = 1.5708


@configclass
class Go2RoughResidualEnvCfg(Go2AmpEnvCfg):
    action_space = 15
    observation_space = 210
    state_space = 48
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", collision_group=-1,
        num_envs=4096, max_init_terrain_level=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20,
            horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
            use_cache=False, curriculum=True,
            sub_terrains={
                "smooth_slope": HfPyramidSlopedTerrainCfg(
                    proportion=0.5, slope_range=(0.0, 0.35), platform_width=2.0,
                ),
                "rough_grid": MeshRandomGridTerrainCfg(
                    proportion=0.5, grid_width=0.45, grid_height_range=(0.0, 0.12),
                ),
            },
        ),
    )
    reference_state_initialization = False
    history_steps = 5
    residual_policy = True
    motion_prior = "pretrained/[pos]+[lsgan]+[h2]+[project].pt"
    command_ranges = ((0.0, 1.0), (-0.5, 0.5), (-1.57, 1.57))
    tracking_lin_vel_scale = 1.0
