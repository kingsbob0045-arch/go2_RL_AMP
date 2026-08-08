from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .go2_rough_config import GO2RoughCfg

COM_OFFSET = torch.tensor([0.021112, 0., -0.005366])   # base link inertial origin
HIP_OFFSETS = torch.tensor([
    [0.1934,  0.0465, 0.],   # FL
    [0.1934, -0.0465, 0.],   # FR
    [-0.1934,  0.0465, 0.],  # RL
    [-0.1934, -0.0465, 0.],  # RR
]) + COM_OFFSET


class Go2RoughEnv( LeggedRobot ):
    def __init__(self, cfg: GO2RoughCfg, sim_params, physics_engine, sim_device, headless):
        self.motion_prior_path = cfg.env.motion_prior
        self.residual_action_scale = cfg.env.residual_action_scale
        self.skill_cmd_dim = cfg.env.skill_cmd_dim
        self.motion_prior_num_obs = cfg.env.prior_num_obs
        self.motion_prior_num_actions = cfg.env.prior_num_actions

        # These buffers are created before super().__init__() so they must use
        # sim_device directly (self.device is set inside BaseTask.__init__).
        self.skill_cmd_buf = torch.zeros(
            cfg.env.num_envs, self.skill_cmd_dim,
            dtype=torch.float, device=sim_device)

        self.motion_prior_obs_buf = torch.zeros(
            cfg.env.num_envs, self.motion_prior_num_obs,
            dtype=torch.float, device=sim_device)

        self.motion_prior_action_buf = torch.zeros(
            cfg.env.num_envs, self.motion_prior_num_actions,
            dtype=torch.float, device=sim_device)

        self.motion_prior = torch.jit.load(
            os.path.join(LEGGED_GYM_ROOT_DIR, self.motion_prior_path),
            map_location=sim_device
        ).eval()

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ------------------------------------------------------------------
    # Buffer fix: num_actions=15 (actor output) vs num_dof=12 (joints)
    # ------------------------------------------------------------------

    def _init_buffers(self):
        """Call parent, then resize joint-control buffers from num_actions=15
        to num_dof=12.  The extra 3 dims are the skill command head and must
        not reach the PD controller or be stored as last_actions in the obs."""
        super()._init_buffers()

        n, d = self.num_envs, self.num_dof   # d = 12
        kw = dict(dtype=torch.float, device=self.device, requires_grad=False)

        # Parent already filled p/d_gains[0:12] correctly; just truncate.
        self.torques      = torch.zeros(n, d, **kw)
        self.p_gains      = self.p_gains[:d].contiguous()
        self.d_gains      = self.d_gains[:d].contiguous()
        self.actions      = torch.zeros(n, d, **kw)
        self.last_actions = torch.zeros(n, d, **kw)

        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains, self.randomized_d_gains = \
                self.compute_randomized_gains(self.num_envs)

    def compute_randomized_gains(self, num_envs):
        """Same logic as parent but uses num_dof instead of num_actions."""
        p_mult = ((
            self.cfg.domain_rand.stiffness_multiplier_range[0] -
            self.cfg.domain_rand.stiffness_multiplier_range[1]) *
            torch.rand(num_envs, self.num_dof, device=self.device) +
            self.cfg.domain_rand.stiffness_multiplier_range[1]).float()
        d_mult = ((
            self.cfg.domain_rand.damping_multiplier_range[0] -
            self.cfg.domain_rand.damping_multiplier_range[1]) *
            torch.rand(num_envs, self.num_dof, device=self.device) +
            self.cfg.domain_rand.damping_multiplier_range[1]).float()
        return p_mult * self.p_gains[:self.num_dof], d_mult * self.d_gains[:self.num_dof]

    # ------------------------------------------------------------------
    # Step: two-head actor output → residual + skill → joint command
    # ------------------------------------------------------------------

    def step(self, actions):
        """actions: [N, 15]
            [:, :12]  residual joint actions  (from the residual head)
            [:, 12:]  skill command logits     (from the skill head)

        The skill command is passed through softmax before feeding the prior,
        so the prior always receives a valid probability simplex.

        Joint command sent to the PD controller:
            joint_action = prior_action(obs | softmax(skill)) + scale * residual
        """
        residual  = actions[:, :self.motion_prior_num_actions]          # [N, 12]
        skill_cmd = torch.softmax(actions[:, self.motion_prior_num_actions:], dim=-1)  # [N, 3]

        self.skill_cmd_buf[:] = skill_cmd

        # Prior input: single-step obs (42) + skill (3) = 45
        # The prior was trained in posTracking env where:
        #   obs_buf[3:6] = [dx_body * 0.5, dy_body * 0.5, 0.0]  (commands_scale=[0.5,0.5,1.0])
        # Here obs_buf[3:6] = [lin_vel_x * 2.0, lin_vel_y * 2.0, ang_vel * 0.25] — wrong scale/semantics.
        # Fix: derive body-frame unit direction from the velocity command and apply scale 0.5.
        vel_xy   = self.commands[:, :2]                                   # [N, 2]
        norm     = vel_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dir_body = vel_xy / norm * 0.5                                    # unit dir × 0.5 (training scale)

        prior_obs = self.obs_buf.clone()
        prior_obs[:, 3] = dir_body[:, 0]
        prior_obs[:, 4] = dir_body[:, 1]
        prior_obs[:, 5] = 0.0                                             # z slot always 0 in training

        self.motion_prior_obs_buf[:] = torch.cat(
            [prior_obs, self.skill_cmd_buf], dim=-1)

        with torch.inference_mode():
            self.motion_prior_action_buf[:] = self.motion_prior(
                self.motion_prior_obs_buf)

        joint_actions = (self.motion_prior_action_buf
                         + self.residual_action_scale * residual)   # [N, 12]

        # LeggedRobot.step handles:
        #   - clips actions, stores self.actions = joint_actions (12-dim)
        #   - PD decimation loop
        #   - post_physics_step → compute_observations → obs_buf (42-dim)
        #   - obs_buf_history: reset on episode end + insert new obs
        #   - returns (policy_obs [N, 5*42=210], privileged_obs, rew, done, ...)
        return super().step(joint_actions)

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        mean_reward = torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if mean_reward > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = 0.
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

        # For decrease: only evaluate envs that were given high-speed commands (>80% of max).
        # Using all envs inflates the mean because most get easy commands even when max is high,
        # making the <0.4 threshold unreachable in practice.
        elif len(env_ids) > 0:
            max_cmd = self.command_ranges["lin_vel_x"][1]
            hard_mask = self.commands[env_ids, 0] > 0.8 * max_cmd
            hard_ids  = env_ids[hard_mask]
            if len(hard_ids) > 0:
                hard_reward = torch.mean(self.episode_sums["tracking_lin_vel"][hard_ids]) / self.max_episode_length
            else:
                hard_reward = mean_reward
            if hard_reward < 0.4 * self.reward_scales["tracking_lin_vel"]:
                self.command_ranges["lin_vel_x"][0] = 0.
                new_max_vel = self.command_ranges["lin_vel_x"][1] - 0.2
                self.command_ranges["lin_vel_x"][1] = np.clip(new_max_vel, 0.5, self.cfg.commands.max_curriculum)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.3) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]