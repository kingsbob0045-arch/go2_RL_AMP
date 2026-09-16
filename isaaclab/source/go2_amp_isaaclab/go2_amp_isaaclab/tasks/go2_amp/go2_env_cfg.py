"""Isaac Lab configurations preserving the legacy sim-to-real contract."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
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
    # Isaac Lab ships the Go2 with a DC-motor model whose available torque decays linearly
    # with joint speed (tau_max = 23.5 * (1 - qd/30)), so at the ~15 rad/s reached during a
    # 3 m/s canter the robot only has half its torque.  The legged_gym reference this project
    # reproduces applies a plain +/- URDF-effort clip with no speed droop, and gives the calf
    # its true 45.43 Nm instead of a blanket 23.5 Nm.  IdealPDActuator is the exact analogue:
    # tau = Kp (q* - q) - Kd qd, clipped to the effort limit.
    cfg.actuators = {
        "hip_thigh": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=23.7, stiffness=50.0, damping=1.0, friction=0.0,
        ),
        "calf": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            # 45.43 Nm is the Go2's knee peak torque.  35.55 Nm, which this line carried
            # until now, is the *Go1* figure, and measuring saturation against it overstated
            # the calf's saturation badly (91% "saturated" against a ceiling 22% too low).
            effort_limit=45.43, stiffness=50.0, damping=1.0, friction=0.0,
        ),
    }
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
            # Widened from +/-10%: a real Go2's motor constant and effective controller gain
            # vary by more than that across units and with temperature, and the policy has to
            # hold across the spread rather than only at the nominal point.
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
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

    # --- added for sim-to-real ---------------------------------------------------------
    # The terms above vary friction, the trunk mass, the actuator gains and an occasional
    # shove.  Everything below is a gap the hardware will have and that the policy would
    # otherwise meet for the first time on the robot.

    link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # Every link, not just the trunk.  The legs are what swing, so an error in shank
            # mass changes the dynamics far more than the same fraction on the body: this
            # project's own MuJoCo model carries a calf 24% heavier than the USD does, which
            # is the size of discrepancy this range has to cover.
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    link_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # +/-1 cm covers the payload and cabling a real unit carries, and the 26 mm
            # calf-COM disagreement measured between this project's own two models.
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )
    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            # "abs", not "scale": the USD ships joint friction and armature at exactly 0, so
            # a multiplicative range would randomise nothing at all.  Verified with
            # tools/sim_ab/dump_isaac_model.py.
            "friction_distribution_params": (0.0, 0.05),
            "armature_distribution_params": (0.0, 0.012),
            "operation": "abs",
            "distribution": "uniform",
        },
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
    # legged_gym clips actions at +/-100, not +/-1.  Clipping at the action range makes the
    # environment blind to any sampled action beyond it, which removes the surrogate loss's
    # only downward pressure on the policy std and lets the entropy bonus grow it without
    # bound (observed: std 1 -> 38 over 2000 iterations).
    action_clip = 100.0

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        # Defaults overflow the contact-patch stream past ~8k environments, which silently
        # drops contacts.  These ceilings cover 16384 environments with headroom and are
        # numerically inert whenever the default buffers would already have sufficed.
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=2**19,
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
            gpu_collision_stack_size=2**27,
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
    # legged_gym's LeggedRobotCfg.rewards.only_positive_rewards, which the reference SECAMP
    # config inherits: "avoids early termination problems" caused by a negative task reward
    # making suicide the best available action early in training.
    only_positive_rewards = True

    # Regularisation.  legged_gym's four terms at legged_gym's coefficients.  The SECAMP
    # reference zeroes all four on the theory that the discriminator alone keeps the motion
    # natural; that theory holds for character animation, where there is no motor to burn and
    # no hardware to cross over to.  These are published, widely reproduced values, which
    # outranks anything derived from first principles here.
    #
    # _get_rewards returns (reward - penalty) * step_dt, the same convention as legged_gym's
    # _prepare_reward_function, so the coefficients transfer across directly with no rescaling.
    torque_scale = 2.0e-4
    dof_acc_scale = 2.5e-7
    action_rate_scale = 1.0e-2
    dof_pos_limit_scale = 10.0    # applied to soft_joint_pos_limits, as legged_gym does

    add_observation_noise = True
    gravity_noise = 0.05
    dof_pos_noise = 0.03
    dof_vel_noise = 1.5

    # Control latency, in whole control periods (one period = decimation * sim.dt = 20 ms),
    # resampled per environment on reset.  (0, 0) disables.
    #
    # The dominant term is the control period itself: an action computed from the state at t
    # is held over [t, t+20ms].  The board-local hops on top of that -- DDS /lowstate to
    # /lowcmd, plus network inference -- are a few milliseconds each.  So end to end is one
    # control period, and published Unitree sim-to-real work randomises on that order.
    #
    # Observation delay and action delay are the same physical loop delay, so opening both
    # double-counts it: the previous (2, 4) and (0, 1) put the worst case at five periods,
    # 100 ms, which is not a sim-to-real margin but a harder task.  Keep the budget in one
    # place and leave the observation path undelayed.
    action_latency_steps = (0, 1)
    observation_latency_steps = (0, 0)

    # Persistent per-episode IMU bias on the projected-gravity observation, on top of the
    # zero-mean gravity_noise above.  A real IMU's bias does not resample every step: it is a
    # constant tilt for the whole run, which is a different and harder disturbance than noise.
    gravity_bias = 0.03
    motion_glob = "datasets/mocap_motions_go2/*"
    reference_state_initialization = True
    # At 1.0 -- what this was -- every episode starts mid-clip, with the clip's joint and base
    # velocities.  The standing default pose at rest is then the one initial state the policy
    # never sees, and it is exactly the state the hardware hands it: low_level_ctrl ramps the
    # robot to default_angles and only then gives the policy control.  A policy that has never
    # seen that state saturates its output there, all four calves fold to the same extreme, and
    # Kp = 50 turns the resulting error into a launch.
    #
    # Mixing the two initial distributions is what makes the Isaac episode length mean anything
    # about the robot.  (The earlier note here said a default-pose reset "mostly produced instant
    # terminations" -- that is the symptom, not a reason to remove the case.)
    reference_state_initialization_prob = 0.5
    # Spread on the default-pose half, so it is a family of standing poses rather than one point.
    default_pose_joint_noise = 0.1
    # Point the first waypoint along the direction the spawned clip is actually travelling,
    # instead of a uniformly random bearing.  See _resample_waypoints in go2_env.py.
    waypoint_align_to_reference = True
    # Draw the reference clip from the pool matching the episode's sampled skill, instead of
    # from every clip regardless of skill.  See _reset_idx.
    reference_matches_skill = True
    amp_horizon = 1
    skill_dim = 0
    # Skill order, matching _SKILL_MAP_3 in rsl_rl/datasets/secamp_motion_loader.py and the
    # "0=pace, 1=trot, 2=canter" note in go2_Deploy's go2_secamp.yaml.  Also the prefix every
    # dataset's filenames carry, which is how _build_skill_traj_pools groups the clips.
    skill_names = ("pace", "trot", "canter")
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
    # Switch the observed target to the next waypoint before actually arriving, so the command
    # magnitude does not shrink on approach.  Without it every waypoint carries a deceleration
    # incentive that fights the canter speed target.
    waypoint_lookahead_threshold = 0.3
    waypoint_max_turn_angle = 1.5708

    # A perfectly flat floor is the one surface the robot will never actually walk on, and a
    # policy trained only on it learns foot timing that assumes the ground is exactly where
    # it predicted.  This is deliberately gentle -- a 0-4 cm random grid, no slopes and no
    # curriculum -- because the goal is to stop the policy depending on a flat world, not to
    # teach it rough-terrain locomotion, which is what Go2RoughResidualEnvCfg is for.
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        # 4096 matches Go2AmpEnvCfg.scene.num_envs; configclass turns `scene` into a dataclass
        # field, so it cannot be read off the parent class here.  train.py overrides both.
        prim_path="/World/ground", terrain_type="generator", collision_group=-1,
        num_envs=4096, max_init_terrain_level=0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20,
            horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
            use_cache=False, curriculum=False,
            sub_terrains={
                "rough_grid": MeshRandomGridTerrainCfg(
                    proportion=1.0, grid_width=0.45, grid_height_range=(0.0, 0.04),
                ),
            },
        ),
    )

    def __post_init__(self):
        super().__post_init__()

        self.viewer.eye = (2.5,2.0,1)
        self.viewer.lookat = (0,0,-0.1)
        self.viewer.env_index = 0
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


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
