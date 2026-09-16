"""Direct-workflow Go2 AMP environments using Isaac Lab tensors only."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
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
        self._skill_traj_pools = None
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, names = self.robot.find_joints(list(GO2_JOINT_NAMES), preserve_order=True)
        if tuple(names) != GO2_JOINT_NAMES:
            raise RuntimeError(f"Go2 joint contract mismatch: expected {GO2_JOINT_NAMES}, got {tuple(names)}")
        self._bad_contact_ids, _ = self.contact_sensor.find_bodies(["base", ".*_thigh", ".*_calf"])
        self._foot_ids, _ = self.contact_sensor.find_bodies([".*_foot"])
        self.default_joint_pos = self.robot.data.default_joint_pos[:, self._joint_ids].clone()
        self.dof_pos_limits = self.robot.data.soft_joint_pos_limits[0, self._joint_ids].clone()
        self._penalty_parts: dict[str, torch.Tensor] = {}
        # Previous-step joint velocity, for the acceleration penalty in _regularisation().
        self._previous_joint_vel = torch.zeros(
            (self.num_envs, len(self._joint_ids)), device=self.device)

        # Control latency.  Row 0 of each ring holds the newest entry, so reading row k gives
        # the value from k control periods ago; every environment carries its own k.
        self._env_arange = torch.arange(self.num_envs, device=self.device)
        self._action_latency_range = tuple(getattr(cfg, "action_latency_steps", (0, 0)))
        self._action_delay = torch.zeros(
            (max(self._action_latency_range) + 1, self.num_envs, 12), device=self.device)
        self._action_latency = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._obs_latency_range = tuple(getattr(cfg, "observation_latency_steps", (0, 0)))
        self._obs_latency = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Persistent per-episode IMU bias, resampled on reset rather than every step.
        self._gravity_bias = torch.zeros((self.num_envs, 3), device=self.device)

        self.num_obs = cfg.observation_space // cfg.history_steps
        # Sized on first use: the policy vector is num_obs wide when history_steps == 1 but
        # the whole flattened history otherwise, and only the caller knows which.
        self._obs_delay: torch.Tensor | None = None
        self._resample_hardware(torch.arange(self.num_envs, device=self.device))
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
            self._skill_traj_pools = self._build_skill_traj_pools(files)
            self._derive_skill_speed_limits(files)

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
            clip = self.cfg.action_clip
            self.actions = (prior_action + self.cfg.residual_action_scale * residual).clamp(-clip, clip)
        else:
            actor_output = actions.clamp(-self.cfg.action_clip, self.cfg.action_clip)
            self.actions = actor_output[:, :12]
        # self.actions stays the *commanded* action, because that is what the policy sees fed
        # back as last_action; the plant receives the delayed one.  That asymmetry is what a
        # real stack has and a zero-latency simulator hides.
        executed = self._delay(self._action_delay, self._action_latency, self.actions)
        self._joint_targets = self.default_joint_pos + self.cfg.action_scale * executed

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def _base_policy_obs(self, add_noise: bool) -> torch.Tensor:
        gravity = self.robot.data.projected_gravity_b.clone()
        joint_pos = self.robot.data.joint_pos[:, self._joint_ids] - self.default_joint_pos
        joint_vel = self.robot.data.joint_vel[:, self._joint_ids] * self.cfg.dof_vel_scale
        if add_noise and self.cfg.add_observation_noise:
            # Zero-mean noise plus the episode's constant IMU bias.  The noise averages out
            # over a stride; the bias does not, so it is the one the policy has to learn to
            # tolerate rather than filter.
            gravity += (2.0 * torch.rand_like(gravity) - 1.0) * self.cfg.gravity_noise
            gravity += self._gravity_bias
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
        # Observation latency is applied to the policy's view only.  The critic is a training
        # -time construct that never runs on the robot, so handing it stale state would add
        # variance to the value estimate for nothing.
        if self._obs_delay is None:
            self._obs_delay = torch.zeros(
                (max(self._obs_latency_range) + 1,) + policy.shape, device=self.device)
        policy = self._delay(self._obs_delay, self._obs_latency, policy,
                             fill_ids=self._just_reset.nonzero(as_tuple=False).flatten())
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
            limits = self._skill_speed_limits[self.skill_commands.argmax(dim=-1)]
            reward = torch.minimum(speed, limits) * self.cfg.tracking_position_scale
        else:
            lin_err = torch.square(self.commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]).sum(dim=1)
            yaw_err = torch.square(self.commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
            reward = self.cfg.tracking_lin_vel_scale * torch.exp(-lin_err / self.cfg.tracking_sigma)
            reward += self.cfg.tracking_ang_vel_scale * torch.exp(-yaw_err / self.cfg.tracking_sigma)
        if self.cfg.only_positive_rewards:
            reward = reward.clamp(min=0.0)
        # Regularisation is applied *after* the positive clamp on purpose.  only_positive_rewards
        # exists so a negative task reward cannot make early termination the best available
        # action; it is not meant to hide the penalties.  Inside the clamp, any penalty larger
        # than the task reward would be flattened to zero and lose its gradient -- recreating
        # exactly the flat region these terms exist to escape.
        penalty = self._regularisation()
        self._log_actuator_diagnostics(reward, penalty)
        return (reward - penalty) * self.step_dt

    def _log_actuator_diagnostics(self, reward: torch.Tensor, penalty: torch.Tensor) -> None:
        """Publish how deep in torque saturation the policy is running, every step.

        This is not incidental telemetry.  The measured failure of this policy is that it
        operates 48% saturated overall and 91.5% saturated at the calves, and saturation is
        invisible in every curve that was being watched: reward, episode length and imitation
        reward all look healthy while the applied torque has stopped responding to the action
        at all.  Logging it makes the quantity the penalty targets directly readable, and the
        penalty coefficient calibratable against it rather than guessed dimensionally.

        The runner averages whatever appears under extras['episode'] into Episode/* scalars.
        """
        ids = self._joint_ids
        computed = self.robot.data.computed_torque[:, ids]
        applied = self.robot.data.applied_torque[:, ids]
        excess = computed - applied
        # A joint is saturated exactly when the actuator had to clip its demand.  1e-4 N.m is
        # far below any torque the policy commands and well above float noise.
        saturated = (excess.abs() > 1.0e-4).float()
        # A fresh dict each step, never a reused one: the runner appends this object to a list
        # and averages the list at the end of the iteration, so handing it the same dict every
        # step would make all 24 entries alias one value and report only the final step.
        # setdefault, not a fresh dict: _get_dones already made this step's dict and put the
        # termination-cause keys in it, and a reassignment here would drop them.
        episode = self.extras.setdefault("episode", {})
        episode["torque_saturation_frac"] = saturated.mean()
        episode["torque_saturation_frac_calf"] = saturated[:, 2::3].mean()
        episode["torque_excess_abs_mean"] = excess.abs().mean()
        episode["torque_demand_abs_mean"] = computed.abs().mean()
        episode["regularisation_penalty"] = penalty.mean()
        for name, term in self._penalty_parts.items():
            episode[f"penalty_{name}"] = term.mean()
        episode["task_reward_raw"] = reward.mean()
        episode["action_abs_mean"] = self.actions.abs().mean()
        # Posture and gait.  Every symptom reported from watching the v4 policies -- crouched
        # stance, a rightward roll, a forward pitch, shuffling steps -- was invisible in the
        # logs, which carried reward, episode length and torque and nothing about the shape the
        # robot was holding.  These five make each of those symptoms a number:
        #   base_height     0.33 m is the Go2's nominal stance; a crouch reads straight off it.
        #   gravity_x / _y  projected gravity is (0, 0, -1) when level.  x is pitch, y is roll,
        #                   and unlike term_contact_gravity_z these are measured every step,
        #                   so a steady lean shows up without the robot having to fall over.
        #   foot_contacts   feet on the ground, of 4.  A pace or trot averages about 2; a
        #                   shuffle keeps three or four down and sits near 3.5.
        #   stance_width    lateral foot spread in the base frame; the "legs splayed out" one.
        height = self.robot.data.root_pos_w[:, 2] - self.terrain.env_origins[:, 2]
        episode["base_height"] = height.mean()
        episode["gravity_x"] = self.robot.data.projected_gravity_b[:, 0].mean()
        episode["gravity_y"] = self.robot.data.projected_gravity_b[:, 1].mean()
        foot_force = self.contact_sensor.data.net_forces_w_history[:, :, self._foot_ids]
        episode["foot_contacts"] = (foot_force.norm(dim=-1).max(dim=1)[0] > 1.0).float().sum(dim=1).mean()
        feet_b = self._foot_positions_in_base(self.robot.data.joint_pos[:, self._joint_ids])
        episode["stance_width"] = feet_b[:, 1::3].abs().mean()
        # Per skill, so a dataset whose three pools are three real gaits can be told apart from
        # one whose pools differ only by name.  Forward speed is the discriminating quantity:
        # pace, trot and canter are defined by it in the SECAMP skill definition.
        forward = self.robot.data.root_lin_vel_b[:, 0]
        for skill in range(self.cfg.skill_dim):
            mask = self.skill_commands[:, skill]
            episode[f"speed_skill{skill}"] = (forward * mask).sum() / mask.sum().clamp(min=1.0)

    def _resample_hardware(self, env_ids: torch.Tensor) -> None:
        """Draw new per-episode latencies and IMU bias, and clear those rows' delay rings.

        Latency and IMU bias are properties of the machine, not of the moment, so they are
        constant within an episode and only resampled here.  The rings are cleared so a reset
        environment cannot execute an action left over from its previous life.
        """
        if not len(env_ids):
            return
        low, high = self._action_latency_range
        self._action_latency[env_ids] = torch.randint(
            int(low), int(high) + 1, (len(env_ids),), device=self.device)
        low, high = self._obs_latency_range
        self._obs_latency[env_ids] = torch.randint(
            int(low), int(high) + 1, (len(env_ids),), device=self.device)
        bias = float(getattr(self.cfg, "gravity_bias", 0.0))
        self._gravity_bias[env_ids] = (2.0 * torch.rand(
            (len(env_ids), 3), device=self.device) - 1.0) * bias
        self._action_delay[:, env_ids] = 0.0
        if self._obs_delay is not None:  # sized lazily on the first observation
            self._obs_delay[:, env_ids] = 0.0

    def _delay(self, ring: torch.Tensor, latency: torch.Tensor, value: torch.Tensor,
               fill_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Push `value` into a per-environment delay ring and read each row at its own lag.

        `fill_ids` names environments whose entire ring should be primed with the new value
        instead of being read back from history.  A just-reset environment has no history to
        be late about, and an empty ring would otherwise hand the policy a vector of zeros --
        which is not a stale observation but an impossible one, since zero projected gravity
        means the robot is in free fall with no orientation.  The action ring is deliberately
        left to zero instead: a zero action means "hold the default pose", which is exactly
        what a freshly reset robot should be doing.
        """
        if ring.shape[0] == 1:
            return value
        ring[1:] = ring[:-1].clone()
        ring[0] = value
        if fill_ids is not None and len(fill_ids):
            ring[:, fill_ids] = value[fill_ids]
        return ring[latency, self._env_arange]

    def _regularisation(self) -> torch.Tensor:
        """legged_gym's four penalty terms, at legged_gym's coefficients.

        Every term is gated on its scale, so zeroing all four reproduces the SECAMP
        reference's unpenalised setup exactly.  Saturation is still logged by
        _log_actuator_diagnostics as a diagnostic; it is deliberately not a penalty, because
        the right way to keep a policy off its torque ceiling is a correct effort limit, sane
        PD gains and a blanket torque term -- not a bespoke term on the overshoot.
        """
        ids = self._joint_ids
        penalty = torch.zeros(self.num_envs, device=self.device)
        parts = self._penalty_parts
        parts.clear()

        if self.cfg.torque_scale:
            parts['torque'] = self.cfg.torque_scale * torch.square(
                self.robot.data.applied_torque[:, ids]).sum(dim=1)
            penalty += parts['torque']

        if self.cfg.action_rate_scale:
            parts['action_rate'] = self.cfg.action_rate_scale * torch.square(
                self.actions - self.last_actions).sum(dim=1)
            penalty += parts['action_rate']

        if self.cfg.dof_acc_scale:
            accel = (self.robot.data.joint_vel[:, ids] - self._previous_joint_vel) / self.step_dt
            parts['dof_acc'] = self.cfg.dof_acc_scale * torch.square(accel).sum(dim=1)
            penalty += parts['dof_acc']

        if self.cfg.dof_pos_limit_scale:
            joint_pos = self.robot.data.joint_pos[:, ids]
            below = (self.dof_pos_limits[:, 0] - joint_pos).clamp(min=0.0)
            above = (joint_pos - self.dof_pos_limits[:, 1]).clamp(min=0.0)
            parts['dof_pos_limit'] = self.cfg.dof_pos_limit_scale * (below + above).sum(dim=1)
            penalty += parts['dof_pos_limit']

        self._previous_joint_vel.copy_(self.robot.data.joint_vel[:, ids])
        return penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        forces = self.contact_sensor.data.net_forces_w_history
        bad_force = torch.norm(forces[:, :, self._bad_contact_ids], dim=-1).amax(dim=(1, 2))
        terminated = bad_force > self.cfg.termination_contact_force
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # A FRESH dict per step, and this is the place to create it: the runner appends
        # infos['episode'] by reference and averages the list afterwards, so reusing one dict
        # across a rollout would make every Episode/* scalar the last step's value rather than
        # the mean.  _get_dones runs before _get_rewards (direct_rl_env.py), so
        # _log_actuator_diagnostics adds to the dict made here.
        #
        # Per-step rates, so both keys are present on every step.  Their ratio is the share of
        # episodes that ran to the time limit, and 1 / (sum) is the mean episode length --
        # neither of which the previous logs could answer, leaving "did it fall or did it
        # finish?" unanswerable for every run so far.
        episode = {}
        self.extras["episode"] = episode
        episode["term_contact_rate"] = terminated.float().mean()
        episode["term_timeout_rate"] = time_out.float().mean()
        # Uprightness at the moment of a contact termination: -1 is level, 0 is on its side.
        # Separates "fell over" from "brushed the ground while upright".
        count = terminated.sum()
        episode["term_contact_gravity_z"] = torch.where(
            count > 0,
            (self.robot.data.projected_gravity_b[:, 2] * terminated).sum() / count.clamp(min=1),
            torch.tensor(-1.0, device=self.device),
        )
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
        # New episode, new machine: redraw this row's latencies and IMU bias and clear its
        # delay rings, so nothing survives from the previous episode.
        self._resample_hardware(env_ids)
        # The skill is drawn first: both the reference clip and the first waypoint are then
        # drawn to agree with it, instead of all three being sampled independently.
        if self.cfg.skill_dim:
            self._resample_skills(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.terrain.env_origins[env_ids]
        root_state[:, 7:13] = sample_uniform(-0.5, 0.5, root_state[:, 7:13].shape, self.device)
        # Default: head where the robot is already facing.  Reference rows overwrite this with
        # the direction their clip is actually travelling, a few lines down.
        heading = self._yaw_from_quat(root_state[:, 3:7])

        if self._amp_loader is not None and len(env_ids):
            use_ref = torch.rand(len(env_ids), device=self.device) < self.cfg.reference_state_initialization_prob
            ref_local = use_ref.nonzero(as_tuple=False).flatten()
            joint_ids = torch.as_tensor(self._joint_ids, device=self.device)
            default_local = (~use_ref).nonzero(as_tuple=False).flatten()
            if len(default_local):
                # The other half starts from the standing default pose -- the state the robot
                # actually boots into and the one a prob of 1.0 never showed the policy.  The
                # spread makes it a family of standing poses rather than a single point.
                joint_pos[default_local[:, None], joint_ids] += sample_uniform(
                    -self.cfg.default_pose_joint_noise, self.cfg.default_pose_joint_noise,
                    (len(default_local), len(self._joint_ids)), self.device)
            if len(ref_local):
                from rsl_rl.datasets.motion_loader import AMPLoader
                frames = self._reference_frames(self.skill_commands[env_ids[ref_local]].argmax(dim=-1))
                selected = env_ids[ref_local]
                joint_pos[ref_local[:, None], joint_ids] = AMPLoader.get_joint_pose_batch(frames)
                joint_vel[ref_local[:, None], joint_ids] = AMPLoader.get_joint_vel_batch(frames)
                # The clip's root height is measured from the ground it was recorded on, so
                # it is the terrain origin's z that has to be added, not just x and y.  On the
                # flat plane every earlier run used, that z was 0 and dropping it cost
                # nothing; on generated terrain it is the local ground height, and dropping it
                # spawns the robot up to a box-height *inside* the ground, which PhysX then
                # resolves by ejecting it.
                root_state[ref_local, :3] = AMPLoader.get_root_pos_batch(frames)
                root_state[ref_local, :3] += self.terrain.env_origins[selected]
                root_state[ref_local, 3:7] = AMPLoader.get_root_rot_batch(frames)
                root_state[ref_local, 7:10] = quat_apply(
                    root_state[ref_local, 3:7], AMPLoader.get_linear_vel_batch(frames))
                root_state[ref_local, 10:13] = quat_apply(
                    root_state[ref_local, 3:7], AMPLoader.get_angular_vel_batch(frames))
                # Where this clip is going, in world frame.  Clips that are barely moving give
                # no usable bearing, so those fall back to the clip's own facing.
                planar = root_state[ref_local, 7:9]
                speed = planar.norm(dim=-1)
                heading[ref_local] = torch.where(
                    speed > 0.1,
                    torch.atan2(planar[:, 1], planar[:, 0]),
                    self._yaw_from_quat(root_state[ref_local, 3:7]),
                )
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
        if self.cfg.waypoint_mode:
            self._resample_waypoints(env_ids, initial_angle=heading)

    def _build_skill_traj_pools(self, files: list[str]) -> np.ndarray | None:
        """Group the loaded clips by the skill their filename names.

        Every dataset in this project follows the same convention the SECAMP loader documents:
        the base name starts with the skill ("pace0.txt", "canter_0.json", "trot_ai4_dog...").
        Clips matching no skill -- datasets/camp carries left_turn and right_turn -- are
        neutral and join every pool.

        Returns a padded (skill, slot) index matrix, or None when any skill has no clips, in
        which case _reference_frames falls back to drawing from the whole set.
        """
        if not self.cfg.skill_dim or not getattr(self.cfg, "reference_matches_skill", False):
            return None
        names = [Path(f).name for f in files]
        pools = [[i for i, n in enumerate(names) if n.startswith(skill)]
                 for skill in self.cfg.skill_names]
        matched = {i for pool in pools for i in pool}
        neutral = [i for i in range(len(names)) if i not in matched]
        pools = [pool + neutral for pool in pools]
        if not all(pools):
            empty = [s for s, pool in zip(self.cfg.skill_names, pools) if not pool]
            print(f"[Go2AmpEnv] no clips for skill(s) {empty}; reference states will be drawn "
                  f"from all {len(names)} clips regardless of skill.")
            return None
        width = max(len(pool) for pool in pools)
        # Pad by repeating the pool, so a uniform draw over [0, len) needs no per-row masking.
        self._skill_pool_len = np.array([len(pool) for pool in pools])
        return np.array([(pool * width)[:width] for pool in pools])

    def _derive_skill_speed_limits(self, files: list[str]) -> None:
        """Cap each skill's task reward at the speed its own reference clips actually travel.

        The task reward is min(speed_along_direction, limit) * tracking_position_scale, so the
        limit is where the reward stops paying for more speed.  Hardcoding (1.0, 1.5, 3.0) --
        the SECAMP paper's figures for its own mocap -- makes that cap wrong for every other
        dataset: kine2go's "canter" clips travel 0.86 m/s, so a 3.0 limit paid the policy to
        run 3.5x faster than the only canter motion the discriminator would accept, and the
        two rewards then pull against each other for the whole run.  Reading the cap from the
        clips removes the conflict without tuning anything.

        The 90th percentile, not the mean: the cap must not clip the fast clips inside a pool.
        """
        limits = list(self.cfg.skill_speed_limits)
        if self._skill_traj_pools is not None:
            from rsl_rl.datasets.motion_loader import AMPLoader
            vx = np.array([float(traj[:, AMPLoader.LINEAR_VEL_START_IDX].mean())
                           for traj in self._amp_loader.trajectories_full])
            for skill in range(len(limits)):
                pool = np.unique(self._skill_traj_pools[skill][:self._skill_pool_len[skill]])
                limits[skill] = round(float(np.percentile(vx[pool], 90.0)), 3)
            print(f"[Go2AmpEnv] skill speed limits from the clips: "
                  f"{dict(zip(self.cfg.skill_names, limits))} "
                  f"(config had {self.cfg.skill_speed_limits})")
        self._skill_speed_limits = torch.tensor(limits, device=self.device)

    def _reference_frames(self, skill_ids: torch.Tensor) -> torch.Tensor:
        """Reference frames drawn from the clips that match each row's skill.

        Drawing from every clip regardless of skill -- what get_full_frame_batch does -- spawns
        the robot mid-canter while the skill one-hot says "pace".  The discriminator is
        skill-conditioned, so it then judges that pose against pace expert data, and the
        conflict is paid at every single reset.
        """
        count = len(skill_ids)
        if self._skill_traj_pools is None:
            return self._amp_loader.get_full_frame_batch(count)
        rows = skill_ids.cpu().numpy()
        slots = (np.random.uniform(size=count) * self._skill_pool_len[rows]).astype(np.int_)
        traj_idxs = self._skill_traj_pools[rows, slots]
        times = self._amp_loader.traj_time_sample_batch(traj_idxs)
        return self._amp_loader.get_full_frame_at_time_batch(traj_idxs, times)

    @staticmethod
    def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat.unbind(dim=-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        for i, limits in enumerate(self.cfg.command_ranges):
            self.commands[env_ids, i] = sample_uniform(limits[0], limits[1], (len(env_ids),), self.device)

    def _resample_skills(self, env_ids: torch.Tensor) -> None:
        indices = torch.randint(0, self.cfg.skill_dim, (len(env_ids),), device=self.device)
        self.skill_commands[env_ids] = 0.0
        self.skill_commands[env_ids, indices] = 1.0

    def _resample_waypoints(self, env_ids: torch.Tensor,
                            initial_angle: torch.Tensor | None = None) -> None:
        """Place two waypoints ahead of each environment.

        `initial_angle` aims the *first* leg, and only reset passes it: the command the policy
        sees at spawn then agrees with the motion the discriminator is asking for.  A uniformly
        random bearing puts the two in direct conflict -- the clip travels one way, the command
        points another, and neither the task reward nor the AMP reward can be satisfied without
        giving up the other.  That conflict is what the sideways gaits were.

        The second leg still turns by up to waypoint_max_turn_angle, and mid-episode
        re-targeting (from _update_waypoint_target, which passes nothing) stays fully random,
        so nothing here removes turning from the task -- it only stops the first step from
        starting in a contradiction.
        """
        origin = self.robot.data.root_pos_w[env_ids, :2].clone()
        previous = None
        aligned = initial_angle is not None and self.cfg.waypoint_align_to_reference
        for k in range(2):
            distance = sample_uniform(*self.cfg.waypoint_distance_range, (len(env_ids),), self.device)
            if previous is None:
                angle = (initial_angle if aligned
                         else sample_uniform(-math.pi, math.pi, (len(env_ids),), self.device))
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
        official = self._waypoints[self._all_env_ids, self._waypoint_index]
        can_lookahead = (
            (torch.norm(self.robot.data.root_pos_w[:, :2] - official, dim=-1)
             < self.cfg.waypoint_lookahead_threshold)
            & (self._waypoint_index < 1)
        )
        current = self._waypoints[self._all_env_ids, self._waypoint_index + can_lookahead.long()]
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
