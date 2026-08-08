"""Direct-workflow Go2 AMP environments using Isaac Lab tensors only."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_apply, quat_apply_inverse, sample_uniform

from .go2_env_cfg import GO2_JOINT_NAMES, Go2AmpEnvCfg


LEGACY_ROOT = Path(__file__).resolve().parents[6]
HIP_OFFSETS = (
    (0.214512, 0.0465, -0.005366), (0.214512, -0.0465, -0.005366),
    (-0.172288, 0.0465, -0.005366), (-0.172288, -0.0465, -0.005366),
)


class Go2AmpEnv(DirectRLEnv):
    """Shared implementation for AMP, SECAMP, and residual rough terrain."""

    cfg: Go2AmpEnvCfg

    def __init__(self, cfg: Go2AmpEnvCfg, render_mode: str | None = None, **kwargs):
        self._amp_loader = None
        self._motion_prior = None
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, names = self.robot.find_joints(list(GO2_JOINT_NAMES), preserve_order=True)
        if tuple(names) != GO2_JOINT_NAMES:
            raise RuntimeError(f"Go2 joint contract mismatch: expected {GO2_JOINT_NAMES}, got {tuple(names)}")
        self._bad_contact_ids, _ = self.contact_sensor.find_bodies(["base", ".*_thigh", ".*_calf"])
        self.default_joint_pos = self.robot.data.default_joint_pos[:, self._joint_ids].clone()
        self.dof_pos_limits = self.robot.data.soft_joint_pos_limits[0, self._joint_ids].clone()

        self.num_obs = cfg.observation_space // cfg.history_steps
        self.num_privileged_obs = cfg.state_space
        self.num_actions = cfg.action_space
        self.include_history_steps = cfg.history_steps if cfg.history_steps > 1 else None
        self.amp_horizon = cfg.amp_horizon
        self.amp_num_obs = 43
        self.enable_skill = cfg.skill_dim > 0
        self.skill_cmd_dim = cfg.skill_dim
        self.dt = self.step_dt

        self.actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._command_time = torch.zeros(self.num_envs, device=self.device)
        self._history = torch.zeros(self.num_envs, cfg.history_steps, 42, device=self.device)
        self.skill_commands = torch.zeros(self.num_envs, cfg.skill_dim, device=self.device)
        self._amp_history = torch.zeros(self.num_envs, cfg.amp_horizon, 43, device=self.device)
        self._terminal_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        terminal_shape = (0, cfg.amp_horizon, 43) if cfg.amp_horizon > 1 else (0, 43)
        self._terminal_amp_states = torch.empty(terminal_shape, device=self.device)
        self._just_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._waypoints = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self._waypoint_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._target_world = torch.zeros(self.num_envs, 2, device=self.device)

        if cfg.reference_state_initialization:
            try:
                from rsl_rl.datasets.motion_loader import AMPLoader
            except ModuleNotFoundError as exc:
                raise ImportError(
                    f"Cannot import the bundled AMP motion loader because '{exc.name}' is missing. "
                    "From the isaaclab directory run: python -m pip install -e ../rsl_rl"
                ) from exc
            files = sorted(str(path) for path in LEGACY_ROOT.glob(cfg.motion_glob))
            if not files:
                raise FileNotFoundError(f"No motions match {LEGACY_ROOT / cfg.motion_glob}")
            self._amp_loader = AMPLoader(
                motion_files=files, device=self.device, time_between_frames=self.step_dt, reorder=False,
            )

        if cfg.residual_policy:
            prior_path = LEGACY_ROOT / cfg.motion_prior
            if not prior_path.is_file():
                raise FileNotFoundError(f"Frozen motion prior not found: {prior_path}")
            self._motion_prior = torch.jit.load(str(prior_path), map_location=self.device).eval()

        self._resample_commands(self._all_env_ids)
        if cfg.skill_dim:
            self._resample_skills(self._all_env_ids)
        if cfg.waypoint_mode:
            self._resample_waypoints(self._all_env_ids)

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.cfg.terrain.num_envs = self.cfg.scene.num_envs
        if self.cfg.terrain.terrain_type == "plane":
            self.cfg.terrain.env_spacing = self.cfg.scene.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=self.terrain.terrain_prim_paths)
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.cfg.residual_policy:
            actor_output = actions
            residual = actor_output[:, :12]
            skill = torch.softmax(actor_output[:, 12:15], dim=-1)
            prior_obs = self._base_policy_obs(add_noise=False).clone()
            xy = self.commands[:, :2]
            prior_obs[:, 3:5] = xy / xy.norm(dim=-1, keepdim=True).clamp(min=1.0e-6) * 0.5
            prior_obs[:, 5] = 0.0
            with torch.inference_mode():
                prior_action = self._motion_prior(torch.cat((prior_obs, skill), dim=-1))
            self.actions = (prior_action + self.cfg.residual_action_scale * residual).clamp(-1.0, 1.0)
        else:
            actor_output = actions.clamp(-1.0, 1.0)
            self.actions = actor_output[:, :12]
        self._joint_targets = self.default_joint_pos + self.cfg.action_scale * self.actions

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def _base_policy_obs(self, add_noise: bool) -> torch.Tensor:
        gravity = self.robot.data.projected_gravity_b.clone()
        joint_pos = self.robot.data.joint_pos[:, self._joint_ids] - self.default_joint_pos
        joint_vel = self.robot.data.joint_vel[:, self._joint_ids] * self.cfg.dof_vel_scale
        if add_noise and self.cfg.add_observation_noise:
            gravity += (2.0 * torch.rand_like(gravity) - 1.0) * self.cfg.gravity_noise
            joint_pos += (2.0 * torch.rand_like(joint_pos) - 1.0) * self.cfg.dof_pos_noise
            joint_vel += ((2.0 * torch.rand_like(joint_vel) - 1.0)
                          * self.cfg.dof_vel_noise * self.cfg.dof_vel_scale)
        scale = torch.tensor(self.cfg.command_scale, device=self.device)
        return torch.cat((
            gravity, self.commands * scale, joint_pos * self.cfg.dof_pos_scale,
            joint_vel, self.actions,
        ), dim=-1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        clean = self._base_policy_obs(add_noise=False)
        noisy = self._base_policy_obs(add_noise=True)
        policy_single = torch.cat((noisy, self.skill_commands), dim=-1) if self.cfg.skill_dim else noisy
        critic_single = torch.cat((clean, self.skill_commands), dim=-1) if self.cfg.skill_dim else clean
        critic = torch.cat((
            self.robot.data.root_lin_vel_b * self.cfg.lin_vel_scale,
            self.robot.data.root_ang_vel_b * self.cfg.ang_vel_scale,
            critic_single,
        ), dim=-1)

        if self.cfg.history_steps > 1:
            reset_ids = self._just_reset.nonzero(as_tuple=False).flatten()
            active_ids = (~self._just_reset).nonzero(as_tuple=False).flatten()
            if len(reset_ids):
                self._history[reset_ids] = noisy[reset_ids].unsqueeze(1)
            if len(active_ids):
                self._history[active_ids, :-1] = self._history[active_ids, 1:].clone()
                self._history[active_ids, -1] = noisy[active_ids]
            policy = self._history.flatten(1)
        else:
            policy = policy_single
        self._just_reset[:] = False
        self.last_actions.copy_(self.actions)
        self._update_amp_history()
        clip = self.cfg.observation_clip
        return {"policy": policy.clamp(-clip, clip), "critic": critic.clamp(-clip, clip)}

    def _get_rewards(self) -> torch.Tensor:
        self._command_time += self.step_dt
        ids = (self._command_time >= self.cfg.command_resampling_time_s).nonzero(as_tuple=False).flatten()
        if len(ids) and not self.cfg.waypoint_mode:
            self._resample_commands(ids)
            self._command_time[ids] = 0.0
        if self.cfg.waypoint_mode:
            self._update_waypoint_target()
            direction = self._target_world - self.robot.data.root_pos_w[:, :2]
            direction /= direction.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
            speed = (self.robot.data.root_lin_vel_w[:, :2] * direction).sum(dim=-1)
            limits = torch.tensor(self.cfg.skill_speed_limits, device=self.device)[self.skill_commands.argmax(dim=-1)]
            reward = torch.minimum(speed, limits) * self.cfg.tracking_position_scale
        else:
            lin_err = torch.square(self.commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]).sum(dim=1)
            yaw_err = torch.square(self.commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
            reward = self.cfg.tracking_lin_vel_scale * torch.exp(-lin_err / self.cfg.tracking_sigma)
            reward += self.cfg.tracking_ang_vel_scale * torch.exp(-yaw_err / self.cfg.tracking_sigma)
        return reward * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        forces = self.contact_sensor.data.net_forces_w_history
        bad_force = torch.norm(forces[:, :, self._bad_contact_ids], dim=-1).amax(dim=(1, 2))
        terminated = bad_force > self.cfg.termination_contact_force
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._terminal_env_ids = (terminated | time_out).nonzero(as_tuple=False).flatten()
        self._terminal_amp_states = self._next_amp_history()[self._terminal_env_ids].clone()
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if (
            self.cfg.residual_policy
            and self.terrain.terrain_origins is not None
            and hasattr(self, "commands")
            and len(env_ids)
        ):
            distance = torch.norm(
                self.robot.data.root_pos_w[env_ids, :2] - self.terrain.env_origins[env_ids, :2], dim=1,
            )
            move_up = distance > self.cfg.terrain.terrain_generator.size[0] / 2.0
            expected = torch.norm(self.commands[env_ids, :2], dim=1) * self.cfg.episode_length_s * 0.3
            move_down = (distance < expected) & ~move_up
            self.terrain.update_env_origins(env_ids, move_up, move_down)
        super()._reset_idx(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.terrain.env_origins[env_ids]
        root_state[:, 7:13] = sample_uniform(-0.5, 0.5, root_state[:, 7:13].shape, self.device)

        if self._amp_loader is not None and len(env_ids):
            use_ref = torch.rand(len(env_ids), device=self.device) < self.cfg.reference_state_initialization_prob
            ref_local = use_ref.nonzero(as_tuple=False).flatten()
            if len(ref_local):
                from rsl_rl.datasets.motion_loader import AMPLoader
                frames = self._amp_loader.get_full_frame_batch(len(ref_local))
                selected = env_ids[ref_local]
                joint_ids = torch.as_tensor(self._joint_ids, device=self.device)
                joint_pos[ref_local[:, None], joint_ids] = AMPLoader.get_joint_pose_batch(frames)
                joint_vel[ref_local[:, None], joint_ids] = AMPLoader.get_joint_vel_batch(frames)
                root_state[ref_local, :3] = AMPLoader.get_root_pos_batch(frames)
                root_state[ref_local, :2] += self.terrain.env_origins[selected, :2]
                root_state[ref_local, 3:7] = AMPLoader.get_root_rot_batch(frames)
                root_state[ref_local, 7:10] = quat_apply(
                    root_state[ref_local, 3:7], AMPLoader.get_linear_vel_batch(frames))
                root_state[ref_local, 10:13] = quat_apply(
                    root_state[ref_local, 3:7], AMPLoader.get_angular_vel_batch(frames))
        else:
            joint_pos[:, self._joint_ids] *= sample_uniform(
                0.5, 1.5, joint_pos[:, self._joint_ids].shape, self.device)

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self._history[env_ids] = 0.0
        self._amp_history[env_ids] = 0.0
        self._just_reset[env_ids] = True
        self._command_time[env_ids] = 0.0
        self._resample_commands(env_ids)
        if self.cfg.skill_dim:
            self._resample_skills(env_ids)
        if self.cfg.waypoint_mode:
            self._resample_waypoints(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        for i, limits in enumerate(self.cfg.command_ranges):
            self.commands[env_ids, i] = sample_uniform(limits[0], limits[1], (len(env_ids),), self.device)

    def _resample_skills(self, env_ids: torch.Tensor) -> None:
        indices = torch.randint(0, self.cfg.skill_dim, (len(env_ids),), device=self.device)
        self.skill_commands[env_ids] = 0.0
        self.skill_commands[env_ids, indices] = 1.0

    def _resample_waypoints(self, env_ids: torch.Tensor) -> None:
        origin = self.robot.data.root_pos_w[env_ids, :2].clone()
        previous = None
        for k in range(2):
            distance = sample_uniform(*self.cfg.waypoint_distance_range, (len(env_ids),), self.device)
            if previous is None:
                angle = sample_uniform(-math.pi, math.pi, (len(env_ids),), self.device)
            else:
                angle = previous + sample_uniform(
                    -self.cfg.waypoint_max_turn_angle, self.cfg.waypoint_max_turn_angle,
                    (len(env_ids),), self.device)
            origin += torch.stack((distance * torch.cos(angle), distance * torch.sin(angle)), dim=-1)
            self._waypoints[env_ids, k] = origin
            previous = angle
        self._waypoint_index[env_ids] = 0
        self._target_world[env_ids] = self._waypoints[env_ids, 0]

    def _update_waypoint_target(self) -> None:
        current = self._waypoints[self._all_env_ids, self._waypoint_index]
        arrived = torch.norm(self.robot.data.root_pos_w[:, :2] - current, dim=-1) < self.cfg.waypoint_arrival_threshold
        at_last = arrived & (self._waypoint_index == 1)
        self._waypoint_index[arrived & ~at_last] += 1
        if at_last.any():
            ids = at_last.nonzero(as_tuple=False).flatten()
            self._resample_skills(ids)
            self._resample_waypoints(ids)
        current = self._waypoints[self._all_env_ids, self._waypoint_index]
        delta = current - self.robot.data.root_pos_w[:, :2]
        distance = delta.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
        delta = delta / distance * distance.clamp(max=1.0)
        world_vector = torch.cat((delta, torch.zeros(self.num_envs, 1, device=self.device)), dim=-1)
        self.commands[:, :2] = quat_apply_inverse(self.robot.data.root_quat_w, world_vector)[:, :2]
        self.commands[:, 2] = 0.0
        self._target_world = current

    def _amp_frame(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos[:, self._joint_ids]
        return torch.cat((
            joint_pos, self._foot_positions_in_base(joint_pos),
            self.robot.data.root_lin_vel_b, self.robot.data.root_ang_vel_b,
            self.robot.data.joint_vel[:, self._joint_ids], self.robot.data.root_pos_w[:, 2:3],
        ), dim=-1)

    def _next_amp_history(self) -> torch.Tensor:
        history = self._amp_history.clone()
        if self.cfg.amp_horizon > 1:
            history[:, :-1] = history[:, 1:].clone()
        history[:, -1] = self._amp_frame()
        return history if self.cfg.amp_horizon > 1 else history[:, 0]

    def _update_amp_history(self) -> None:
        value = self._next_amp_history()
        self._amp_history.copy_(value if value.ndim == 3 else value.unsqueeze(1))

    def get_amp_observations(self) -> torch.Tensor:
        return (self._amp_history if self.cfg.amp_horizon > 1 else self._amp_history[:, 0]).clone()

    def get_amp_observation_buf(self) -> torch.Tensor:
        return self.get_amp_observations()

    def get_skill_cmd_buf(self) -> torch.Tensor:
        return self.skill_commands

    @staticmethod
    def _foot_positions_in_base(joint_pos: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(joint_pos)
        offsets = torch.tensor(HIP_OFFSETS, device=joint_pos.device, dtype=joint_pos.dtype)
        for leg in range(4):
            abduction, hip, knee = joint_pos[:, leg * 3:leg * 3 + 3].unbind(dim=-1)
            length = torch.sqrt(0.213**2 * (2.0 + 2.0 * torch.cos(knee)))
            swing = hip + knee / 2.0
            x = -length * torch.sin(swing)
            z_hip = -length * torch.cos(swing)
            hip_y = 0.0955 * (1.0 if leg % 2 == 0 else -1.0)
            y = torch.cos(abduction) * hip_y - torch.sin(abduction) * z_hip
            z = torch.sin(abduction) * hip_y + torch.cos(abduction) * z_hip
            output[:, leg * 3:leg * 3 + 3] = torch.stack((x, y, z), dim=-1) + offsets[leg]
        return output
