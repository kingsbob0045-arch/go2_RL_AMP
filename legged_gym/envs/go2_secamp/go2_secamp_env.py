import math

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from legged_gym.envs.go2.go2_amp_env import Go2AmpEnv
from .go2_secamp_config import Go2SECAMPCfg


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-π, π]."""
    return (x + math.pi) % (2 * math.pi) - math.pi


class Go2SECAMPEnv(Go2AmpEnv):
    """ [Horizon]      Multi-step AMP observation window (amp_horizon frames)
        [Cat]          Categorical one-hot skill command
        [PosTracking]  3D pose waypoint tracking instead of velocity tracking """

    def __init__(self, cfg: Go2SECAMPCfg, sim_params, physics_engine, sim_device, headless):
        # ---- AMP horizon buffer (must exist before super().__init__ because
        #      post_physics_step is called during parent's __init__) --------
        self.amp_horizon = cfg.env.amp_horizon
        self.amp_num_obs = cfg.env.amp_num_obs
        self.amp_observation_buf = torch.zeros(
            cfg.env.num_envs, self.amp_horizon, self.amp_num_obs,
            dtype=torch.float, device=sim_device, requires_grad=False)

        # ---- Skill command buffer -----------------------------------------
        self.enable_skill   = cfg.env.enable_skill
        self.skill_cmd_dim  = cfg.env.skill_cmd_dim
        if self.enable_skill:
            self.skill_cmd_buf = torch.zeros(
                cfg.env.num_envs, self.skill_cmd_dim,
                dtype=torch.float, device=sim_device)

        # ---- Position tracking buffers ------------------------------------
        # All allocated before super().__init__() because reset_idx →
        # _resample_commands is called during parent's __init__.
        K = cfg.commands.num_waypoints
        N = cfg.env.num_envs
        self.normalize_cmd    = cfg.commands.normalize
        self.clamp_cmd        = cfg.commands.clamp
        self.waypoints        = torch.zeros(N, K, 2, dtype=torch.float, device=sim_device)
        # columns: [x_world, y_world]
        self.waypoint_idx     = torch.zeros(N, dtype=torch.long,  device=sim_device)
        self.pos_target_world = torch.zeros(N, 2, dtype=torch.float, device=sim_device)

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Override commands_scale AFTER super().__init__() (which creates it).
        # commands[:, :3] = [dx_body × 0.5, dy_body × 0.5, ang_vel × 1.0]
        self.commands_scale = torch.tensor([0.5, 0.5, 1.0], device=self.device)

        # Reusable range index [0, N)
        self._arange_N = torch.arange(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # Skill command helpers
    # ------------------------------------------------------------------

    def _resample_skill_cmds(self, env_ids):
        """Sample a one-hot skill command uniformly for each env in env_ids."""
        n = len(env_ids)
        idx = torch.randint(0, self.skill_cmd_dim, (n,), device=self.device)
        self.skill_cmd_buf[env_ids] = 0.0
        self.skill_cmd_buf[env_ids, idx] = 1.0

    # ------------------------------------------------------------------
    # Position-tracking command helpers
    # ------------------------------------------------------------------

    def _resample_commands(self, env_ids):
        """Sample K=2 chained waypoints (XY only) in world frame.

        wp0: random direction from robot position.
        wp1: direction constrained within ±max_turn_angle of the wp0 leg,
             preventing sharp U-turns between the two legs.
        """
        n = len(env_ids)
        if n == 0:
            return

        K            = self.cfg.commands.num_waypoints           # 2
        d_min, d_max = self.cfg.commands.pos_target_dist_range
        max_turn     = self.cfg.commands.max_turn_angle

        origin_xy  = self.root_states[env_ids, :2].clone()
        prev_theta = None  # tracks leg direction for turn constraint

        for k in range(K):
            dist = torch_rand_float(d_min, d_max, (n, 1), device=self.device).squeeze(1)

            if k == 0:
                theta = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
            else:
                # Constrain: new leg direction within ±max_turn_angle of previous leg
                delta = (torch.rand(n, device=self.device) * 2 - 1) * max_turn
                theta = _wrap_to_pi(prev_theta + delta)

            wp_x = origin_xy[:, 0] + dist * torch.cos(theta)
            wp_y = origin_xy[:, 1] + dist * torch.sin(theta)

            self.waypoints[env_ids, k, 0] = wp_x
            self.waypoints[env_ids, k, 1] = wp_y

            prev_theta = theta
            origin_xy  = torch.stack([wp_x, wp_y], dim=1)

        self.waypoint_idx[env_ids]     = 0
        self.pos_target_world[env_ids] = self.waypoints[env_ids, 0]

    def _update_pos_target(self):
        """Check arrival at current waypoint; advance index if reached.

        Uses XY-only arrival condition.  When the last waypoint is reached,
        immediately resample a fresh 2-waypoint sequence so the robot always
        has a new target.

        Lookahead: obs target switches to k+1 when within lookahead_threshold
        of k, removing the deceleration incentive at each waypoint.
        """
        K = self.cfg.commands.num_waypoints  # 2

        official_target = self.waypoints[self._arange_N, self.waypoint_idx]   # [N, 2]
        dist_xy = torch.norm(self.root_states[:, :2] - official_target, dim=-1)

        arrived = dist_xy < self.cfg.commands.arrival_threshold

        if arrived.any():
            prev_idx = self.waypoint_idx.clone()
            self.waypoint_idx = (self.waypoint_idx + arrived.long()).clamp(max=K - 1)

            at_last = arrived & (prev_idx == K - 1)
            if at_last.any():
                fresh_ids = at_last.nonzero(as_tuple=False).flatten()
                if self.enable_skill:
                    self._resample_skill_cmds(fresh_ids)
                self._resample_commands(fresh_ids)

        lookahead = self.cfg.commands.lookahead_threshold
        dist_to_official = torch.norm(
            self.root_states[:, :2] - self.waypoints[self._arange_N, self.waypoint_idx],
            dim=-1)
        can_lookahead = (dist_to_official < lookahead) & (self.waypoint_idx < K - 1)
        obs_idx = (self.waypoint_idx + can_lookahead.long()).clamp(max=K - 1)

        self.pos_target_world = self.waypoints[self._arange_N, obs_idx]  # [N, 2]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)   # calls _resample_commands(env_ids) → waypoints sampled
        if self.enable_skill and len(env_ids) > 0:
            self._resample_skill_cmds(env_ids)
            self._resample_commands(env_ids)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def compute_observations(self):
        """Build policy observations with XY waypoint direction only.

        Obs layout (45 actor / 51 critic):
          [0:3]   (critic only) base_lin_vel × 2.0
          [3:6]   (critic only) base_ang_vel × 0.25
          [6:9]   projected_gravity
          [9:11]  [dx_body × 0.5,  dy_body × 0.5]  (XY direction to waypoint)
          [11]    0.0  (dz slot unused)
          [12:24] dof_pos
          [24:36] dof_vel
          [36:48] actions
          [48:51] skill one-hot [3]
        """
        # --- Current robot yaw from quaternion ----------------------------
        qw = self.base_quat[:, 3]
        qx = self.base_quat[:, 0]
        qy = self.base_quat[:, 1]
        qz = self.base_quat[:, 2]
        yaw = torch.atan2(2 * (qw * qz + qx * qy),
                          1 - 2 * (qy ** 2 + qz ** 2))

        if self.normalize_cmd:
            # --- XY unit direction in world frame → rotate to body frame -----
            # Paper Eq. (1): d̂_w = (p - x) / ||p - x||
            delta_world = self.pos_target_world - self.root_states[:, :2]
            norm = torch.norm(delta_world, dim=1, keepdim=True).clamp(min=1e-6)
            dir_world = delta_world / norm
            cos_y = torch.cos(yaw)
            sin_y = torch.sin(yaw)
            dx_body =  cos_y * dir_world[:, 0] + sin_y * dir_world[:, 1]
            dy_body = -sin_y * dir_world[:, 0] + cos_y * dir_world[:, 1]
        elif self.clamp_cmd:
            # --- XY delta in world frame → rotate to body frame ---------------
            # d̂_w = max[(p - x), 1.0]
            delta_world = self.pos_target_world - self.root_states[:, :2]
            dist = torch.norm(delta_world, dim=1, keepdim=True).clamp(min=1e-6)
            # Clamp so robot ≥1m away gets unit-magnitude command; stops naturally when close
            delta_world = delta_world / dist * dist.clamp(max=1.0)
            cos_y = torch.cos(yaw)
            sin_y = torch.sin(yaw)
            dx_body =  cos_y * delta_world[:, 0] + sin_y * delta_world[:, 1]
            dy_body = -sin_y * delta_world[:, 0] + cos_y * delta_world[:, 1]
        else:
            # d̂_w = (p - x)
            delta_world = self.pos_target_world - self.root_states[:, :2]
            cos_y = torch.cos(yaw)
            sin_y = torch.sin(yaw)
            dx_body =  cos_y * delta_world[:, 0] + sin_y * delta_world[:, 1]
            dy_body = -sin_y * delta_world[:, 0] + cos_y * delta_world[:, 1]

        # --- Slot XY into commands; leave commands[:, 2] = 0 (no Z tracking)
        # (parent multiplies by self.commands_scale = [0.5, 0.5, 1.0])
        self.commands[:, 0] = dx_body
        self.commands[:, 1] = dy_body
        self.commands[:, 2] = 0.

        # --- Parent fills obs_buf (42-dim) and privileged_obs_buf (48-dim) -
        super().compute_observations()

        # --- Append one-hot skill command (3 dims) -------------------------
        if self.enable_skill:
            self.obs_buf            = torch.cat([self.obs_buf,            self.skill_cmd_buf], dim=-1)
            self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, self.skill_cmd_buf], dim=-1)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def post_physics_step(self):
        """Standard CAMP post-physics step with waypoint advancement."""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # Advance waypoint index for any envs that reached the current target
        self._update_pos_target()

        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # Build H-frame terminal sequences before reset
        if len(env_ids) > 0:
            terminal_amp_states = self.amp_observation_buf[env_ids].clone()
            terminal_amp_states[:, :-1] = terminal_amp_states[:, 1:].clone()
            terminal_amp_states[:, -1]  = self.get_amp_observations()[env_ids]
        else:
            terminal_amp_states = self.amp_observation_buf[env_ids]  # empty [0, H, obs_dim]

        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        self.update_amp_observation_buf()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, terminal_amp_states

    # ------------------------------------------------------------------
    # AMP buffer helpers (unchanged from camp_cat)
    # ------------------------------------------------------------------

    def get_amp_observation_buf(self):
        return self.amp_observation_buf.clone()

    def update_amp_observation_buf(self):
        self.amp_observation_buf[:, :-1] = self.amp_observation_buf[:, 1:].clone()
        self.amp_observation_buf[:, -1]  = self.get_amp_observations().clone()

    def get_skill_cmd_buf(self):
        return self.skill_cmd_buf

    # ------------------------------------------------------------------
    # Noise scale vector
    # ------------------------------------------------------------------

    def _get_noise_scale_vec(self, cfg):
        """Noise scales for the parent's 48-dim privileged obs (before appended dims).

        Heading error and skill one-hot are appended AFTER the parent's
        compute_observations (which applies noise), so they receive no noise.
        """
        base_dim = 48
        if cfg.terrain.measure_heights:
            base_dim += 187
        noise_vec = torch.zeros(base_dim, device=self.device)
        self.add_noise   = cfg.noise.add_noise
        noise_scales     = cfg.noise.noise_scales
        noise_level      = cfg.noise.noise_level
        noise_vec[:3]    = noise_scales.lin_vel  * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6]   = noise_scales.ang_vel  * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9]   = noise_scales.gravity  * noise_level
        noise_vec[9:12]  = 0.   # commands (position target — no noise)
        noise_vec[12:24] = noise_scales.dof_pos  * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel  * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.   # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = (noise_scales.height_measurements * noise_level
                                 * self.obs_scales.height_measurements)
        return noise_vec
    
    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _reward_tracking_pos(self):
        """Waypoint-direction velocity tracking reward (world frame).

        Implements Eq. (1)-(2) from the parkour paper:
            d_w  = (p - x) / ||p - x||                  unit direction to waypoint
            r    = min(<v_world, d_w>, v_max[skill])     projection capped at skill speed

        v_max is skill-conditioned so each gait is encouraged to run at its
        natural speed: pace 1.0 m/s, trot 1.5 m/s, canter 3.0 m/s.

        World frame is used (not base frame) so the robot cannot exploit the
        reward by spinning in place rather than making lateral progress.
        """
        # Robot XY velocity in world frame
        v_world = self.root_states[:, 7:9]          # [N, 2]

        # Unit direction from robot to current waypoint (world-frame XY)
        delta = self.pos_target_world - self.root_states[:, :2]          # [N, 2]
        dist  = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-6)
        d_w   = delta / dist                        # [N, 2]  unit vector

        # Scalar projection of velocity onto waypoint direction
        proj = (v_world * d_w).sum(dim=-1)          # [N]

        # Skill-conditioned speed cap: look up v_max per env via argmax of one-hot
        vmax_table = torch.tensor(
            self.cfg.rewards.tracking_vel_max, dtype=torch.float, device=self.device)  # [S]
        skill_idx  = self.skill_cmd_buf.argmax(dim=-1)          # [N]  long
        v_max      = vmax_table[skill_idx]                       # [N]

        return torch.clamp(proj, max=v_max)
