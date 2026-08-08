"""Evaluation script for go2_secamp.

For each skill runs one fixed episode and records:
  - XY trajectory + waypoints          (top-down view)
  - Distance-to-waypoint over time     (tracking quality)
  - Speed vs v_max over time           (gait speed adherence)
  - Foot contact schedule              (gait cycle verification)

Usage:
    python legged_gym/scripts/play_camp.py \\
        --task go2_secamp --load_run <run_name> \\
        [--record_video] [--speed 0.5]
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
import copy
import os
import sys
import time
import argparse

import isaacgym  # noqa: F401 – must import before torch
from legged_gym.envs import *          # registers all tasks
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CLI extras
# ---------------------------------------------------------------------------
_vparser = argparse.ArgumentParser(add_help=False)
_vparser.add_argument('--record_video', action='store_true', default=False,
                      help='Record MP4 to results/<run_name>/posTracking_eval.mp4')
_vparser.add_argument('--video_fps', type=int, default=50,
                      help='Video output frame rate (default: 50 = real-time)')
_vparser.add_argument('--speed', type=float, default=1.0,
                      help='Playback speed multiplier (e.g. 0.5 = half speed). '
                           'Only effective in non-headless mode.')
_vparser.add_argument('--export', action='store_true', default=False,
                      help='Export actor as TorchScript to pretrained/<run_name>.pt')
_VARGS, _remaining = _vparser.parse_known_args()
# Strip our custom flags from sys.argv so get_args() doesn't choke on them
sys.argv = [sys.argv[0]] + _remaining

_TMP_FRAME = f'/tmp/_camp_pt_frame_{os.getpid()}.png'

# ---------------------------------------------------------------------------
# Skill definitions  (pace=0, trot=1, canter=2)
# ---------------------------------------------------------------------------
SKILLS = {
    "pace":   {"one_hot": [1, 0, 0], "v_max": 1.0},
    "trot":   {"one_hot": [0, 1, 0], "v_max": 1.5},
    "canter": {"one_hot": [0, 0, 1], "v_max": 3.0},
}
SKILL_SEQUENCE = ["pace", "trot", "canter"]

STEPS_PER_SKILL = 600   # 12 s at 50 Hz
WARMUP_STEPS    = 50    # discard initial transient for contact schedule
PLOT_WINDOW     = 150   # contact window shown ≈ 3 s
ROBOT_IDX       = 0

FOOT_LABELS = ['FL', 'FR', 'RL', 'RR']
FOOT_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
WAYPOINT_COLOR = '#E53935'
TRAJ_COLOR     = '#1565C0'
CAM_OFFSET     = np.array([0.5, 1.5, 1.0])   # behind/side/above robot

# ---------------------------------------------------------------------------
# Speed continuity summary
# ---------------------------------------------------------------------------

def _speed_continuity(speed_arr: np.ndarray, dt: float) -> dict:
    """Compute smoothness metrics for the projected-speed time series."""
    speed_arr = np.asarray(speed_arr)
    jerk      = np.abs(np.diff(speed_arr)) / dt          # |Δv| / Δt  [m/s²]
    return dict(
        mean_speed   = float(np.mean(speed_arr)),
        std_speed    = float(np.std(speed_arr)),
        mean_jerk    = float(np.mean(jerk)),
        max_jerk     = float(np.max(jerk)) if len(jerk) else 0.0,
        pct_positive = float(np.mean(speed_arr > 0) * 100),
    )


# ---------------------------------------------------------------------------

def play(args):
    if _VARGS.record_video:
        args.headless = False

    # Validate speed
    speed = float(np.clip(_VARGS.speed, 0.05, 1.0))
    if speed != _VARGS.speed:
        print(f"[Speed] clamped {_VARGS.speed} → {speed}")
    if speed < 1.0 and args.headless:
        print("[Speed] --speed has no effect in headless mode.")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Minimal eval environment
    env_cfg.env.num_envs              = 1
    env_cfg.terrain.num_rows          = 5
    env_cfg.terrain.num_cols          = 5
    env_cfg.terrain.curriculum        = False
    env_cfg.noise.add_noise           = False
    env_cfg.domain_rand.randomize_friction  = False
    env_cfg.domain_rand.push_robots         = False
    env_cfg.domain_rand.randomize_gains     = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.reference_state_initialization = False
    env_cfg.env.episode_length_s      = 9999    # no timeout

    train_cfg.runner.amp_num_preload_transitions = 1

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    skill_cmd_dim = env.skill_cmd_dim   # 3

    # Save directory (needed for plots and export naming)
    _load_run = getattr(train_cfg.runner, 'load_run', None)
    run_name = str(_load_run if (_load_run and _load_run != -1)
                   else getattr(train_cfg.runner, 'run_name', 'unknown_run'))
    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'results', run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Export policy as TorchScript
    if _VARGS.export:
        pretrained_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'pretrained')
        os.makedirs(pretrained_dir, exist_ok=True)
        export_path = os.path.join(pretrained_dir, f'{run_name}.pt')
        model = copy.deepcopy(ppo_runner.alg.actor_critic.actor).to('cpu')
        traced = torch.jit.script(model)
        traced.save(export_path)
        print(f"[Export] Policy saved → {export_path}")

    # Per-step sleep for slowdown (seconds)
    sleep_per_step = env.dt * (1.0 / speed - 1.0) if speed < 1.0 and not args.headless else 0.0
    if sleep_per_step > 0:
        print(f"[Speed] {speed:.2f}× — sleeping {sleep_per_step*1000:.1f} ms/step")

    # Optional video writer
    video_writer = None
    if _VARGS.record_video:
        try:
            import imageio
            video_path   = os.path.join(save_dir, 'eval_gaits.mp4')
            video_writer = imageio.get_writer(
                video_path, fps=_VARGS.video_fps, quality=8, macro_block_size=None)
            print(f"[Video] Recording → {video_path}")
        except ImportError:
            print("[Video] imageio not installed — skipping.")

    traj_logs: dict = {}

    print("\n" + "=" * 60)
    print("  PosTracking evaluation")
    print(f"  Skills: {' → '.join(SKILL_SEQUENCE)}")
    print(f"  {STEPS_PER_SKILL} steps/skill  ({STEPS_PER_SKILL * env.dt:.1f} s)")
    print(f"  Speed: {speed:.2f}×")
    print("=" * 60 + "\n")

    for skill_name in SKILL_SEQUENCE:
        skill_info = SKILLS[skill_name]
        one_hot = torch.tensor(
            skill_info["one_hot"], dtype=torch.float32, device=env.device
        ).unsqueeze(0)  # [1, 3]
        v_max = skill_info["v_max"]

        # Force skill and resample fresh waypoints
        env.skill_cmd_buf[:] = one_hot
        env._resample_commands(torch.arange(env.num_envs, device=env.device))
        obs = env.get_observations()

        print(f"Skill → \033[1;33m{skill_name.upper()}\033[0m  "
              f"v_max={v_max:.1f} m/s  waypoints sampled")

        xy_traj    = []   # [T, 2]
        dist_log   = []   # [T]   distance to current waypoint
        speed_log  = []   # [T]   world-frame speed projected onto d_w
        contacts   = []   # [T, 4]
        wp_visited = []   # waypoint XY positions when target changes

        # Record initial target
        wp_visited.append(env.pos_target_world[ROBOT_IDX].cpu().numpy().copy())

        for local_step in range(STEPS_PER_SKILL):
            with torch.inference_mode():
                actions = policy(obs.detach())
            obs, _, _, _, _, _, _ = env.step(actions.detach())

            # Re-apply skill (reset_idx may overwrite skill_cmd_buf)
            env.skill_cmd_buf[:] = one_hot
            obs[:, -skill_cmd_dim:] = one_hot

            # Camera follow
            if not args.headless:
                look_at = env.root_states[ROBOT_IDX, :3].cpu().numpy().astype(np.float64)
                env.set_camera(look_at + CAM_OFFSET, look_at)

            # Video frame
            if video_writer is not None and env.viewer is not None:
                import imageio
                env.gym.write_viewer_image_to_file(env.viewer, _TMP_FRAME)
                frame = imageio.imread(_TMP_FRAME)
                video_writer.append_data(frame[:, :, :3])

            # --- Logging -----------------------------------------------------
            root_xy   = env.root_states[ROBOT_IDX, :2].cpu().numpy().copy()
            target_xy = env.pos_target_world[ROBOT_IDX].cpu().numpy()  # [2]

            xy_traj.append(root_xy)

            dist = float(np.linalg.norm(root_xy - target_xy))
            dist_log.append(dist)

            # Speed projected onto waypoint direction (world frame)
            delta  = target_xy - root_xy
            d_norm = np.linalg.norm(delta)
            d_w    = delta / d_norm if d_norm > 1e-6 else np.array([1.0, 0.0])
            v_world = env.root_states[ROBOT_IDX, 7:9].cpu().numpy()
            speed_log.append(float(np.dot(v_world, d_w)))

            # Track waypoint advances
            if not np.allclose(target_xy, wp_visited[-1], atol=1e-2):
                wp_visited.append(target_xy.copy())

            # Contact schedule (skip warmup)
            if local_step >= WARMUP_STEPS:
                c = (env.contact_forces[ROBOT_IDX, env.feet_indices, 2] > 1.0
                     ).cpu().numpy().astype(np.float32)
                contacts.append(c)

            if (local_step + 1) % 200 == 0:
                wp_idx = env.waypoint_idx[ROBOT_IDX].item()
                print(f"  step {local_step+1:>4}  "
                      f"d_wp={dist:.2f}m  wp_idx={wp_idx}  "
                      f"speed={speed_log[-1]:+.2f}m/s")

            # Slowdown
            if sleep_per_step > 0:
                time.sleep(sleep_per_step)

        speed_arr = np.array(speed_log)
        cont = _speed_continuity(speed_arr, env.dt)
        print(f"  [{skill_name}] mean_speed={cont['mean_speed']:+.2f}  "
              f"std={cont['std_speed']:.2f}  "
              f"mean_jerk={cont['mean_jerk']:.2f} m/s²  "
              f"fwd%={cont['pct_positive']:.0f}%")

        traj_logs[skill_name] = dict(
            xy         = np.stack(xy_traj, axis=0),          # [T, 2]
            waypoints  = np.stack(wp_visited, axis=0),        # [K_visited, 2]
            dist       = np.array(dist_log),                  # [T]
            speed      = speed_arr,                           # [T]
            continuity = cont,
            v_max      = v_max,
            contacts   = (np.stack(contacts, axis=0)
                          if contacts else np.zeros((0, 4))), # [T-warmup, 4]
        )

    if video_writer is not None:
        video_writer.close()
        if os.path.exists(_TMP_FRAME):
            os.remove(_TMP_FRAME)

    print("\nAll skills done. Generating plots…")
    _make_plots(traj_logs, env.dt, save_dir)
    _print_continuity_summary(traj_logs)
    print("Done.")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_plots(traj_logs: dict, dt: float, save_dir: str):
    _plot_trajectories(traj_logs, save_dir)
    _plot_tracking(traj_logs, dt, save_dir)
    _plot_gait_cycles(traj_logs, dt, save_dir)
    print(f"Saved plots → {save_dir}/")


def _print_continuity_summary(traj_logs: dict):
    """Print a formatted speed-continuity summary table."""
    print("\n" + "─" * 52)
    print(f"  {'Skill':<10}  {'v̄ (m/s)':>8}  {'σ':>6}  "
          f"{'jerk̄ (m/s²)':>12}  {'fwd%':>6}")
    print("─" * 52)
    for skill in SKILL_SEQUENCE:
        c = traj_logs[skill]['continuity']
        print(f"  {skill:<10}  {c['mean_speed']:>+8.2f}  "
              f"{c['std_speed']:>6.2f}  "
              f"{c['mean_jerk']:>12.2f}  "
              f"{c['pct_positive']:>5.0f}%")
    print("─" * 52 + "\n")


def _plot_trajectories(traj_logs: dict, save_dir: str):
    """Top-down XY trajectory + waypoint markers for each skill."""
    n   = len(SKILL_SEQUENCE)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, skill in zip(axes, SKILL_SEQUENCE):
        d  = traj_logs[skill]
        xy = d['xy']
        wps = d['waypoints']

        ax.plot(xy[:, 0], xy[:, 1],
                color=TRAJ_COLOR, lw=1.2, alpha=0.8, label='Trajectory')
        ax.scatter(xy[0, 0],  xy[0, 1],  color='green', s=60, zorder=5, label='Start')
        ax.scatter(xy[-1, 0], xy[-1, 1], color='red',   s=60, marker='x', zorder=5, label='End')

        if len(wps) > 0:
            ax.scatter(wps[:, 0], wps[:, 1],
                       color=WAYPOINT_COLOR, s=80, marker='*',
                       zorder=6, label='Waypoints')
            ax.plot(wps[:, 0], wps[:, 1],
                    color=WAYPOINT_COLOR, lw=0.8, ls='--', alpha=0.6)

        v_max = SKILLS[skill]["v_max"]
        ax.set_aspect('equal')
        ax.set_title(f'{skill.capitalize()}  (v_max={v_max} m/s)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=8,
               frameon=True, bbox_to_anchor=(1.0, 0.0))
    # fig.suptitle('CAMP-PosTracking — XY Trajectories', fontsize=13, fontweight='bold')

    path = os.path.join(save_dir, 'posTracking_trajectories.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_tracking(traj_logs: dict, dt: float, save_dir: str):
    """Per-skill: projected speed vs v_max (single row)."""
    n   = len(SKILL_SEQUENCE)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for col, skill in enumerate(SKILL_SEQUENCE):
        d     = traj_logs[skill]
        T     = len(d['dist'])
        t     = np.arange(T) * dt
        v_max = d['v_max']

        ax = axes[col]
        ax.plot(t, d['speed'], lw=1.0, color='#4CAF50', label='proj. speed')
        ax.axhline(v_max, color='#E53935', lw=1.0, ls='--',
                   label=f'v_max={v_max} m/s')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title(skill.capitalize(), fontsize=10, fontweight='bold')
        ax.set_ylabel('Speed along $d_w$ (m/s)', fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)

    path = os.path.join(save_dir, 'posTracking_tracking.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_gait_cycles(traj_logs: dict, dt: float, save_dir: str):
    """Gantt-style foot contact schedule for each skill."""
    n   = len(SKILL_SEQUENCE)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.2), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, skill in zip(axes, SKILL_SEQUENCE):
        contacts = traj_logs[skill]['contacts']
        window   = contacts[-PLOT_WINDOW:] if len(contacts) >= PLOT_WINDOW else contacts
        _draw_contact_schedule(ax, window, skill.capitalize(), dt)

    patches = [plt.Rectangle((0, 0), 1, 1, fc=FOOT_COLORS[fi]) for fi in range(4)]
    fig.legend(patches, FOOT_LABELS, loc='lower right', fontsize=9,
               frameon=True, title='Foot', title_fontsize=9,
               bbox_to_anchor=(1.0, 0.0))
    # fig.suptitle('CAMP-PosTracking — Gait Cycles', fontsize=13, fontweight='bold')

    path = os.path.join(save_dir, 'posTracking_gait_cycles.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _draw_contact_schedule(ax, contact_window, title, dt):
    if len(contact_window) == 0:
        ax.set_title(title, fontsize=10, fontweight='bold')
        return
    t_rel = np.arange(len(contact_window)) * dt
    for fi in range(4):
        y       = 3 - fi
        contact = contact_window[:, fi].astype(bool)
        padded  = np.concatenate(([False], contact, [False]))
        diff    = np.diff(padded.astype(int))
        starts  = np.where(diff == 1)[0]
        ends    = np.where(diff == -1)[0]
        bars    = [(t_rel[s], t_rel[e - 1] - t_rel[s] + dt) for s, e in zip(starts, ends)]
        if bars:
            ax.broken_barh(bars, (y + 0.08, 0.84),
                           facecolors=FOOT_COLORS[fi], edgecolors='none')

    total_t = len(contact_window) * dt
    ax.set_xlim(0, total_t)
    ax.set_ylim(0, 4)
    ax.set_yticks(np.arange(4) + 0.5)
    ax.set_yticklabels(FOOT_LABELS[::-1], fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel('Time (s)', fontsize=8, labelpad=2)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    ax.spines[['top', 'right']].set_visible(False)


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = get_args()
    play(args)
