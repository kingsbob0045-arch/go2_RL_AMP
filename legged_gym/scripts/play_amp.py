# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def play_gait_sweep(args):
    """Sweep command velocity from -1 to 3 m/s, record per-step foot contacts, then plot."""
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.reference_state_initialization = False
    env_cfg.commands.ranges.lin_vel_y = [0., 0.]
    env_cfg.commands.ranges.lin_vel_x = [0., 0.]
    env_cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    env_cfg.env.episode_length_s = 999

    train_cfg.runner.amp_num_preload_transitions = 1

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # --- sweep parameters ---
    vel_min, vel_max = 0.0, 3.5
    n_stages   = 15     # number of velocity levels (0.0, 0.25, ..., 3.0)
    hold_steps = 500    # steps per velocity stage
    warmup_steps = 100  # skip first N steps at each stage (transient)
    velocities = np.linspace(vel_min, vel_max, n_stages)
    robot_index = 0
    foot_labels = ['FL', 'FR', 'RL', 'RR']

    # pre-allocate per-step recording arrays
    measure_per_stage = hold_steps - warmup_steps
    total_record = n_stages * measure_per_stage
    contact_log = np.zeros((total_record, 4), dtype=np.float32)   # binary contact per foot
    cmd_vel_log = np.zeros(total_record, dtype=np.float32)
    real_vel_log = np.zeros(total_record, dtype=np.float32)

    print(f"\nGait sweep: {vel_min} -> {vel_max} m/s  ({n_stages} stages x {hold_steps} steps)")

    record_idx = 0
    for cmd_vel in velocities:
        env.commands[:, 0] = cmd_vel
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0

        for step in range(hold_steps):
            actions = policy(obs.detach())
            obs, _, _, _, _, _, _ = env.step(actions.detach())
            env.commands[:, 0] = cmd_vel
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0

            if step >= warmup_steps:
                contacts = (env.contact_forces[robot_index, env.feet_indices, 2] > 1.0).cpu().numpy()
                contact_log[record_idx] = contacts.astype(np.float32)
                cmd_vel_log[record_idx] = cmd_vel
                real_vel_log[record_idx] = env.base_lin_vel[robot_index, 0].item()
                record_idx += 1

    # ---- plot ----
    dt = env.dt
    FOOT_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']  # FL FR RL RR
    N_FEET = 4
    window_steps = min(150, measure_per_stage)  # ~3 s at 50 Hz per subplot

    def draw_contact_schedule(ax, contact_window, title, is_transition=False):
        """Draw a Gantt-style contact schedule on ax."""
        t_rel = np.arange(len(contact_window)) * dt
        for fi in range(N_FEET):
            y = N_FEET - 1 - fi   # FL on top, RR on bottom
            contact = contact_window[:, fi].astype(bool)
            # find contiguous stance runs
            padded = np.concatenate(([False], contact, [False]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends   = np.where(diff == -1)[0]
            bars = [(t_rel[s], t_rel[e - 1] - t_rel[s] + dt) for s, e in zip(starts, ends)]
            if bars:
                ax.broken_barh(bars, (y + 0.08, 0.84),
                               facecolors=FOOT_COLORS[fi], edgecolors='none')
        ax.set_xlim(0, len(contact_window) * dt)
        ax.set_ylim(0, N_FEET)
        ax.set_yticks(np.arange(N_FEET) + 0.5)
        ax.set_yticklabels(foot_labels[::-1], fontsize=7)
        ax.tick_params(axis='x', labelsize=7)
        ax.set_title(title, fontsize=8, pad=3)
        ax.spines[['top', 'right']].set_visible(False)
        if is_transition:
            ax.set_facecolor('#f7f7f7')
            # mark the boundary
            ax.axvline(window_steps // 2 * dt, color='gray', lw=1.0, ls='--', alpha=0.7)
        ax.set_xlabel('Time (s)', fontsize=7, labelpad=1)

    # Build interleaved sequence: s0, t01, s1, t12, s2, ...
    # 13 stages + 12 transitions = 25 items → 5×5 grid
    items = []
    for i in range(n_stages):
        items.append(('stage', i))
        if i < n_stages - 1:
            items.append(('transition', i))

    n_cols = 5
    n_rows = int(np.ceil(len(items) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2, n_rows * 2.4),
                             constrained_layout=True)
    axes = axes.flatten()

    for idx, (item_type, i) in enumerate(items):
        ax = axes[idx]
        if item_type == 'stage':
            # centre window on the middle of this stage
            centre = i * measure_per_stage + measure_per_stage // 2
            start  = np.clip(centre - window_steps // 2, 0, total_record - window_steps)
            title  = f'v = {velocities[i]:.2f} m/s'
            is_trans = False
        else:
            # centre window on the boundary between stage i and i+1
            boundary = (i + 1) * measure_per_stage
            start    = np.clip(boundary - window_steps // 2, 0, total_record - window_steps)
            title    = f'{velocities[i]:.2f} → {velocities[i+1]:.2f} m/s'
            is_trans = True

        window = contact_log[start: start + window_steps]
        draw_contact_schedule(ax, window, title, is_transition=is_trans)

    for idx in range(len(items), len(axes)):
        axes[idx].set_visible(False)

    patches = [plt.Rectangle((0, 0), 1, 1, fc=FOOT_COLORS[fi]) for fi in range(N_FEET)]
    fig.legend(patches, foot_labels, loc='lower right', fontsize=9,
               frameon=True, title='Foot', title_fontsize=9,
               bbox_to_anchor=(1.0, 0.0))

    fig.suptitle('Foot Contact Schedule — Velocity Sweep', fontsize=12, fontweight='bold')

    save_path = os.path.join(
        LEGGED_GYM_ROOT_DIR, 'results',
        train_cfg.runner.experiment_name, 'gait_sweep_contacts.png'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved contact plot -> {save_path}")


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.reference_state_initialization = False
    env_cfg.commands.ranges.lin_vel_y = [0., 0.]
    env_cfg.commands.ranges.lin_vel_x = [1., 2.5]
    env_cfg.commands.ranges.ang_vel_yaw = [0., 0.]

    train_cfg.runner.amp_num_preload_transitions = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    GAIT_SWEEP = True   # set True to run velocity sweep (0->3 m/s) gait test
    args = get_args()
    if GAIT_SWEEP:
        play_gait_sweep(args)
    else:
        play(args)
