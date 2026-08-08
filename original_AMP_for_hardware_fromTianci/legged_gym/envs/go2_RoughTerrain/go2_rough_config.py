from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

SKILL_DIM = 3  # pace=0, trot=1, canter=2

class GO2RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 16384
        num_actions = 15            # 12 residual joint actions + 3 skill command
        include_history_steps = 5   # Number of steps of history to include
        num_observations = 42       # gravity|cmd|dof_pos|dof_vel|last_action
        num_privileged_obs = 48     # lin_vel|ang_vel|actor_obs
        reference_state_initialization = False

        # ------ load frozen motion prior ------------------------------
        motion_prior = 'pretrained/[pos]+[lsgan]+[h2]+[project].pt'
        residual_action_scale = 0.1
        skill_cmd_dim = SKILL_DIM
        prior_num_obs = 42 + SKILL_DIM
        prior_num_actions = 12

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
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
        mesh_type = 'trimesh'
        curriculum = True
        measure_heights = False
        num_rows = 10
        num_cols = 20
        max_init_terrain_level = 1   # start on easy terrain rows
        terrain_proportions = [0.5, 0.5, 0., 0., 0.]
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]

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
        only_positive_rewards = False
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits
        class scales:
            termination         = -0.0
            tracking_lin_vel    = 1.0
            tracking_ang_vel    = 0.5
            lin_vel_z           = -0.0
            ang_vel_xy          = -0.0
            orientation         = -0.0
            torques             = -0.0000
            dof_vel             = -0.
            dof_acc             = -0.
            base_height         = -0. 
            feet_air_time       =  0.0
            collision           = -0.
            feet_stumble        = -0.0 
            action_rate         = -0.0
            stand_still         = -0.
    
    class commands( LeggedRobotCfg.commands ):
        curriculum = True # only for linear vel x
        max_curriculum = 2.0
        num_commands = 4
        resampling_time = 10. 
        heading_command = False 
        class ranges:
            lin_vel_x   = [0.0, 1.0]          # min max [m/s]
            lin_vel_y   = [-0.5, 0.5]         # min max [m/s]
            ang_vel_yaw = [-1.57, 1.57]     # min max [rad/s]
            heading     = [-3.14, 3.14]

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = 'Go2RoughRunner'

    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std        = 1.0
        num_residual_actions  = 12
        num_skill_dims        = 3
        backbone_hidden_dims  = [512, 256]
        critic_hidden_dims    = [512, 256]
        activation            = 'elu'
        # std bounds — [hip, thigh, calf] × 4 legs for residual
        min_std_residual      = [0.05, 0.02, 0.05] * 4   # shape [12]
        max_std_residual      = [3.0,  3.0,  3.0 ] * 4   # shape [12]
        # skill head: fix std so entropy bonus can't drive it to 20+
        fixed_skill_std       = True        # True = non-learnable, stays at init_noise_std
        min_std_skill         = [0.05, 0.05, 0.05]        # only active if fixed_skill_std=False
        max_std_skill         = [2.0,  2.0,  2.0 ]        # only active if fixed_skill_std=False
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # training params
        value_loss_coef         = 1.0
        use_clipped_value_loss  = True
        clip_param              = 0.2
        entropy_coef            = 0.01
        num_learning_epochs     = 5
        num_mini_batches        = 4 
        learning_rate           = 1.e-3
        schedule                = 'adaptive'
        gamma                   = 0.99
        lam                     = 0.95
        desired_kl              = 0.01
        max_grad_norm           = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name       = 'ActorCritic'
        algorithm_class_name    = 'PPO'
        amp_motion_reorder      = False
        num_steps_per_env       = 24
        max_iterations          = 50000

        # logging
        save_interval           = 200
        experiment_name         = 'go2_rough'
        run_name                = '[blind]+[rough_terrain]+[fixed_std]+[cmd_curri]+[wo_stairs]+[wo_discrete]+[vel_movedown]+[correct_prior_cmd]+[wo_reg]+[low_scale]'

        # wandb
        enable_wandb            = True
        wandb_project           = 'go2_rough'
        wandb_entity            = None   # set to your wandb username / team