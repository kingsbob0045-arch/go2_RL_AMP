import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

MOTION_FILES = glob.glob('datasets/camp/*')

SKILL_DIM = 3 # pace=0, trot=1, canter=2
# [1,0,0]=pace, [0,1,0]=trot, [0,0,1]=canter

class Go2SECAMPCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        # ----- cat settings --------------------------------------------
        enable_skill = True
        skill_cmd_dim = SKILL_DIM   # pace trot canter

        # ----- env settings --------------------------------------------
        num_envs = 5480
        include_history_steps = None
        num_observations    = 42 + SKILL_DIM   # 45
        num_privileged_obs  = 48 + SKILL_DIM   # 51

        # ----- rsi settings --------------------------------------------
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85

        # ----- amp settings --------------------------------------------
        amp_horizon = 2
        # joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos
        # 12 + 12 + 3 + 3 + 12 + 1 = 43
        amp_num_obs = 43
        amp_motion_files = MOTION_FILES
        amp_motion_reorder = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.38] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 50.}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = [
            "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
            "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand:
        randomize_friction = True
        friction_range = [0.25, 1.75]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        # Per-skill desired speed cap for waypoint-direction velocity tracking reward (m/s).
        # Index matches skill one-hot order: [pace=0, trot=1, canter=2]
        # r_tracking = min(<v_world, d_w>, v_max[skill])
        tracking_vel_max = [1.0, 1.5, 3.0]  # pace, trot, canter
        class scales( LeggedRobotCfg.rewards.scales ):
            termination      = 0.0
            tracking_pos     = 2.0   
            tracking_lin_vel = 0.0   # disabled
            tracking_ang_vel = 0.0   # disabled
            lin_vel_z        = 0.0
            ang_vel_xy       = 0.0
            orientation      = 0.0
            torques          = 0.0
            dof_vel          = 0.0
            dof_acc          = 0.0
            base_height      = 0.0
            feet_air_time    = 0.0
            collision        = 0.0
            feet_stumble     = 0.0
            action_rate      = 0.0
            stand_still      = 0.0
            dof_pos_limits   = 0.0

    class commands:
        curriculum = False
        max_curriculum = 1.
        # commands[:, 0:2] = body-frame XY to waypoint (set in compute_observations)
        # commands[:, 2]   = 0 (unused)
        num_commands = 3
        resampling_time = 20.   # backup resample interval; normally auto-resampled on last waypoint
        heading_command = False

        # Waypoint sequence
        num_waypoints = 2                        # K waypoints sampled per episode
        pos_target_dist_range = [5.0, 15.0]      # [m] XY polar distance per waypoint
        arrival_threshold = 0.1                  # [m] XY distance for waypoint index advance
        lookahead_threshold = 0.3                # [m] XY distance to switch obs target to next wp

        # Max direction change between consecutive waypoint legs (prevents U-turns)
        max_turn_angle = 1.5708  # [rad] π/2 = 90°
        normalize = False
        clamp = True

        # Required by parent _parse_cfg; not used for actual command sampling
        # (we override _resample_commands with waypoint-based logic)
        class ranges:
            lin_vel_x   = [0.0, 0.0]
            lin_vel_y   = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading     = [0.0, 0.0]

    class curriculum:
        enabled = False


class Go2SECAMPCfgPPO( LeggedRobotCfgPPO ):
    seed = 77
    runner_class_name = 'Go2SECAMPRunner'

    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [1024, 512, 256]
        critic_hidden_dims = [1024, 512, 256]

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4
        disc_learning_rate = 1e-3   # discriminator LR; policy LR stays at learning_rate (1e-3)

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = '[pos]+[lsgan]+[h2]+[project]+[clamp]'
        experiment_name = 'go2_secamp'
        algorithm_class_name = 'SECAMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 50000 # number of policy updates
        save_interval = 200
        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_motion_reorder = False  # Go2 mocap data is already in Isaac order
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 4

        wandb_enable = True
        lerp_schedule_enabled = False
        skill_insert  = 'project' # concat | project
