"""Evaluation script for go2_secamp – gait-transition analysis.

Single 21-second episode with 50 parallel envs covering all 6 gait transitions:
  0– 3 s  pace     →  3– 6 s  trot     (pace→trot)
  6– 9 s  pace     →  9–12 s  canter   (pace→canter)
 12–15 s  trot     → 15–18 s  canter   (trot→canter)
 and reverses: trot→pace, canter→trot, canter→pace

Outputs (saved to results/<run_name>/):
  posTracking_trajectories.png   – env-0 XY path with transition markers
  posTracking_tracking.png       – speed vs v_cmd timeline with phase bands
  posTracking_gait_cycles.png    – contact schedule with phase bands
  posTracking_foot_kinematics.png – all 6 transitions, 2×3 paper layout (mean ± 1σ, N=50)

Usage:
    python legged_gym/scripts/play_camp.py \\
        --task go2_secamp --load_run <run_name> \\
        [--record_video] [--speed 0.5]
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
import copy, os, sys, time, argparse

import isaacgym  # noqa: F401
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CLI ───────────────────────────────────────────────────────────────────────
_vparser = argparse.ArgumentParser(add_help=False)
_vparser.add_argument('--record_video', action='store_true', default=False)
_vparser.add_argument('--video_fps', type=int, default=50)
_vparser.add_argument('--speed', type=float, default=1.0)
_vparser.add_argument('--export', action='store_true', default=False)
_VARGS, _remaining = _vparser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

_TMP_FRAME = f'/tmp/_camp_pt_frame_{os.getpid()}.png'

# ── Experiment constants ───────────────────────────────────────────────────────
NUM_ENVS   = 50
ROBOT_IDX  = 0        # env used for trajectory / speed / contacts logging

# (start_time_s, skill_name, one_hot, v_cmd_m_s)
# 7 phases → 6 transitions covering every ordered pair among pace/trot/canter
SCHEDULE_S = [
    ( 0.0, "pace",   [1, 0, 0], 1.0),   #          start
    ( 3.0, "trot",   [0, 1, 0], 1.5),   # pace  → trot
    ( 6.0, "pace",   [1, 0, 0], 1.0),   # trot  → pace
    ( 9.0, "canter", [0, 0, 1], 3.0),   # pace  → canter
    (12.0, "trot",   [0, 1, 0], 1.5),   # canter→ trot
    (15.0, "canter", [0, 0, 1], 3.0),   # trot  → canter
    (18.0, "pace",   [1, 0, 0], 1.0),   # canter→ pace
]
TOTAL_S     = 21.0   # total episode duration (s)
STATS_WIN_S = 1.0    # ±STATS_WIN_S window around each transition for foot stats

FOOT_LABELS = ['FL', 'FR', 'RL', 'RR']
FOOT_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
PHASE_COLORS = {'pace': '#BBDEFB', 'trot': '#C8E6C9', 'canter': '#FFE0B2'}
CAM_OFFSET   = np.array([0.5, 1.5, 1.0])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_smooth_waypoints(env, step_dist: float = 5.0):
    """Inject K straight-ahead waypoints into env.waypoints for all envs."""
    K = env.cfg.commands.num_waypoints
    with torch.no_grad():
        origin = env.root_states[:, :2].clone()  # [N, 2]
        for k in range(K):
            env.waypoints[:, k, 0] = origin[:, 0] + (k + 1) * step_dist
            env.waypoints[:, k, 1] = origin[:, 1]
        env.waypoint_idx[:] = 0
        env.pos_target_world[:] = env.waypoints[:, 0]


def _open_video(save_dir: str):
    try:
        import imageio
        path = os.path.join(save_dir, 'posTracking_eval.mp4')
        w = imageio.get_writer(path, fps=50, quality=8, macro_block_size=None)
        print(f"[Video] Recording → {path}")
        return w
    except ImportError:
        print("[Video] imageio not installed — skipping.")
        return None


def _export_policy(ppo_runner, run_name: str):
    pretrained_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'pretrained')
    os.makedirs(pretrained_dir, exist_ok=True)
    path = os.path.join(pretrained_dir, f'{run_name}.pt')
    model  = copy.deepcopy(ppo_runner.alg.actor_critic.actor).to('cpu')
    traced = torch.jit.script(model)
    traced.save(path)
    print(f"[Export] Policy saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def play(args):
    if _VARGS.record_video:
        args.headless = False

    speed = float(np.clip(_VARGS.speed, 0.05, 1.0))

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ── Config overrides ────────────────────────────────────────────────────
    env_cfg.env.num_envs                        = NUM_ENVS
    env_cfg.commands.num_waypoints              = 11     # 10 real + 1 buffer waypoint
    env_cfg.terrain.num_rows                    = 5
    env_cfg.terrain.num_cols                    = 5
    env_cfg.terrain.curriculum                  = False
    env_cfg.noise.add_noise                     = False
    env_cfg.domain_rand.randomize_friction      = False
    env_cfg.domain_rand.push_robots             = False
    env_cfg.domain_rand.randomize_gains         = False
    env_cfg.domain_rand.randomize_base_mass     = False
    env_cfg.env.reference_state_initialization  = False
    env_cfg.env.episode_length_s                = 9999
    env_cfg.asset.terminate_after_contacts_on   = []     # disable mid-episode resets
    train_cfg.runner.amp_num_preload_transitions = 1

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    _set_smooth_waypoints(env)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    skill_cmd_dim = env.skill_cmd_dim
    _load_run = getattr(train_cfg.runner, 'load_run', None)
    run_name = str(_load_run if (_load_run and _load_run != -1)
                   else getattr(train_cfg.runner, 'run_name', 'unknown_run'))
    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'results', run_name)
    os.makedirs(save_dir, exist_ok=True)

    if _VARGS.export:
        _export_policy(ppo_runner, run_name)

    # ── Build step-indexed schedule ─────────────────────────────────────────
    dt          = env.dt
    max_steps   = int(TOTAL_S / dt)   # upper-bound; loop exits early at buffer waypoint
    stats_win   = int(STATS_WIN_S / dt)

    # schedule: list of (step_int, name, one_hot, v_cmd)
    schedule = [(int(t / dt), nm, oh, vc) for t, nm, oh, vc in SCHEDULE_S]
    # transition_steps: steps at which a new skill begins (skip phase 0)
    trans_steps  = [s[0] for s in schedule[1:]]   # 6 transition steps
    trans_labels = ["pace→trot", "trot→pace",
                    "pace→canter", "canter→trot",
                    "trot→canter", "canter→pace"]

    # per-step phase index (sized to max_steps; only queried up to actual end step)
    phase_at_step = np.zeros(max_steps, dtype=int)
    for pi, (ps, *_) in enumerate(schedule):
        phase_at_step[ps:] = pi

    # Index of buffer waypoint: stop collecting when robot targets this
    K_eval    = env.cfg.commands.num_waypoints   # 11
    buf_wp    = K_eval - 1                        # index 10 = buffer

    sleep_per_step = dt * (1.0 / speed - 1.0) if (speed < 1.0 and not args.headless) else 0.0
    video_writer   = _open_video(save_dir) if _VARGS.record_video else None

    # ── Logging buffers ─────────────────────────────────────────────────────
    xy_traj      = []   # [T, 2]   env-0 XY
    speed_log    = []   # [T]      env-0 projected speed
    contacts     = []   # [T, 4]   env-0 contact booleans
    foot_pos_log = []   # [T, N, 4, 3]  all envs local foot positions
    wp_visited   = [env.pos_target_world[ROBOT_IDX].cpu().numpy().copy()]

    current_phase = -1  # will trigger first transition on step 0
    current_hot_t = None

    print("\n" + "=" * 60)
    print("  PosTracking gait-transition evaluation")
    print(f"  {NUM_ENVS} envs  |  max {max_steps} steps  ({TOTAL_S:.0f} s at {1/dt:.0f} Hz)")
    print(f"  Transitions at steps: {trans_steps}  ({[t*dt for t in trans_steps]} s)")
    print(f"  Stops when env-0 enters buffer waypoint (idx {buf_wp})")
    print("=" * 60 + "\n")

    for step in range(max_steps):

        # ── Skill transition ────────────────────────────────────────────────
        new_phase = int(phase_at_step[step])
        if new_phase != current_phase:
            current_phase = new_phase
            _, phase_name, phase_hot, _ = schedule[current_phase]
            current_hot_t = torch.tensor(
                phase_hot, dtype=torch.float32, device=env.device)
            env.skill_cmd_buf[:] = current_hot_t.unsqueeze(0)
            if step > 0:
                print(f"  step {step:>4} (t={step*dt:.1f}s)  → \033[1;33m{phase_name.upper()}\033[0m")

        # ── Policy step ─────────────────────────────────────────────────────
        with torch.inference_mode():
            actions = policy(obs.detach())
        obs, _, _, _, _, _, _ = env.step(actions.detach())

        # Re-apply skill (env resets can overwrite skill_cmd_buf)
        env.skill_cmd_buf[:] = current_hot_t.unsqueeze(0)
        obs[:, -skill_cmd_dim:] = current_hot_t.unsqueeze(0)

        # Camera follow (env-0)
        if not args.headless:
            look_at = env.root_states[ROBOT_IDX, :3].cpu().numpy().astype(np.float64)
            env.set_camera(look_at + CAM_OFFSET, look_at)

        if video_writer is not None and env.viewer is not None:
            import imageio
            env.gym.write_viewer_image_to_file(env.viewer, _TMP_FRAME)
            video_writer.append_data(imageio.imread(_TMP_FRAME)[:, :, :3])

        # ── Log env-0 ───────────────────────────────────────────────────────
        root_xy   = env.root_states[ROBOT_IDX, :2].cpu().numpy().copy()
        target_xy = env.pos_target_world[ROBOT_IDX].cpu().numpy()
        xy_traj.append(root_xy)

        if not np.allclose(target_xy, wp_visited[-1], atol=1e-2):
            wp_visited.append(target_xy.copy())

        delta  = target_xy - root_xy
        d_norm = np.linalg.norm(delta)
        d_w    = delta / d_norm if d_norm > 1e-6 else np.array([1.0, 0.0])
        v_world = env.root_states[ROBOT_IDX, 7:9].cpu().numpy()
        speed_log.append(float(np.dot(v_world, d_w)))

        c = (env.contact_forces[ROBOT_IDX, env.feet_indices, 2] > 1.0
             ).cpu().numpy().astype(np.float32)
        contacts.append(c)

        # ── Log foot kinematics (all envs) ──────────────────────────────────
        fp = env.foot_positions_in_base_frame(env.dof_pos)  # [N, 12]
        foot_pos_log.append(fp.cpu().numpy().reshape(NUM_ENVS, 4, 3))

        if (step + 1) % 100 == 0:
            print(f"  step {step+1:>4}/{max_steps}  "
                  f"speed={speed_log[-1]:+.2f} m/s  "
                  f"wp_idx={env.waypoint_idx[ROBOT_IDX].item()}")

        if sleep_per_step > 0:
            time.sleep(sleep_per_step)

        # Stop when env-0 enters the buffer waypoint (all real waypoints visited)
        if env.waypoint_idx[ROBOT_IDX].item() >= buf_wp:
            print(f"  step {step+1}: env-0 reached buffer waypoint — stopping collection")
            break

    if video_writer is not None:
        video_writer.close()
        if os.path.exists(_TMP_FRAME):
            os.remove(_TMP_FRAME)

    print("\nAll steps done. Generating plots…")
    data = dict(
        xy_traj      = np.stack(xy_traj, axis=0),        # [T, 2]
        speed        = np.array(speed_log),               # [T]
        contacts     = np.stack(contacts, axis=0),        # [T, 4]
        foot_pos     = np.stack(foot_pos_log, axis=0),    # [T, N, 4, 3]
        wp_visited   = np.stack(wp_visited, axis=0),      # [K, 2]
        schedule     = schedule,
        trans_steps  = trans_steps,
        trans_labels = trans_labels,
        stats_win    = stats_win,
        dt           = dt,
        total_steps  = len(xy_traj),   # actual collected steps
    )
    _make_plots(data, save_dir)
    print("Done.")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _make_plots(data: dict, save_dir: str):
    _plot_trajectories(data, save_dir)
    _plot_tracking(data, save_dir)
    _plot_gait_cycles(data, save_dir)
    _plot_foot_kinematics(data, save_dir)
    print(f"Saved plots → {save_dir}/")


def _add_phase_bands(ax, schedule, total_steps: int, dt: float):
    """Shade each skill phase and draw transition dashed lines."""
    phase_steps = [s[0] for s in schedule] + [total_steps]
    for i, (_, name, *_) in enumerate(schedule):
        t0 = phase_steps[i] * dt
        t1 = phase_steps[i + 1] * dt
        ax.axvspan(t0, t1, alpha=0.12, color=PHASE_COLORS[name], zorder=0)
    for i in range(1, len(schedule)):
        ax.axvline(phase_steps[i] * dt, color='#555', lw=0.8, ls='--', alpha=0.7)


def _plot_trajectories(data: dict, save_dir: str):
    xy          = data['xy_traj']
    wps         = data['wp_visited']
    trans_steps = data['trans_steps']
    trans_labels= data['trans_labels']

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    ax.plot(xy[:, 0], xy[:, 1], color='#1565C0', lw=1.2, alpha=0.8, label='Trajectory')
    ax.scatter(xy[0, 0],  xy[0, 1],  color='green', s=70, zorder=5, label='Start')
    ax.scatter(xy[-1, 0], xy[-1, 1], color='red',   s=70, marker='x', zorder=5, label='End')

    if len(wps) > 0:
        ax.scatter(wps[:, 0], wps[:, 1], color='#E53935', s=80,
                   marker='*', zorder=6, label='Waypoints')

    for ts, lbl in zip(trans_steps, trans_labels):
        if ts < len(xy):
            ax.scatter(xy[ts, 0], xy[ts, 1], s=55, color='#FF6F00', zorder=7, marker='D')
            ax.annotate(lbl, xy[ts], fontsize=6.5, color='#FF6F00',
                        xytext=(5, 5), textcoords='offset points')

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, frameon=True)
    ax.spines[['top', 'right']].set_visible(False)

    path = os.path.join(save_dir, 'posTracking_trajectories.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_tracking(data: dict, save_dir: str):
    speed       = data['speed']
    dt          = data['dt']
    schedule    = data['schedule']
    total_steps = data['total_steps']
    T  = len(speed)
    t  = np.arange(T) * dt

    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    _add_phase_bands(ax, schedule, total_steps, dt)

    ax.plot(t, speed, lw=1.0, color='#4CAF50', label='proj. speed (env 0)')
    ax.axhline(0, color='gray', lw=0.5)

    # v_cmd reference line per phase
    phase_steps = [s[0] for s in schedule] + [total_steps]
    for i, (ps, name, _, v_cmd) in enumerate(schedule):
        t0 = phase_steps[i] * dt
        t1 = phase_steps[i + 1] * dt
        ax.hlines(v_cmd, t0, t1, colors='#E53935', lw=1.2, ls='--',
                  label='v_cmd' if i == 0 else None)
        ax.text((t0 + t1) / 2, v_cmd + 0.12, name,
                ha='center', fontsize=7.5, color='#333', style='italic')

    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('Speed along $d_w$ (m/s)', fontsize=8)
    ax.set_xlim(0, T * dt)
    ax.legend(fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=7)

    path = os.path.join(save_dir, 'posTracking_tracking.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_gait_cycles(data: dict, save_dir: str):
    contacts    = data['contacts']   # [T, 4]
    dt          = data['dt']
    schedule    = data['schedule']
    total_steps = data['total_steps']
    T     = len(contacts)
    t_rel = np.arange(T) * dt

    fig, ax = plt.subplots(figsize=(10, 2.8), constrained_layout=True)
    _add_phase_bands(ax, schedule, total_steps, dt)

    for fi in range(4):
        y       = 3 - fi
        contact = contacts[:, fi].astype(bool)
        padded  = np.concatenate(([False], contact, [False]))
        diff    = np.diff(padded.astype(int))
        starts  = np.where(diff == 1)[0]
        ends    = np.where(diff == -1)[0]
        bars    = [(t_rel[s], t_rel[min(e, T) - 1] - t_rel[s] + dt)
                   for s, e in zip(starts, ends)]
        if bars:
            ax.broken_barh(bars, (y + 0.08, 0.84),
                           facecolors=FOOT_COLORS[fi], edgecolors='none')

    ax.set_xlim(0, T * dt)
    ax.set_ylim(0, 4)
    ax.set_yticks(np.arange(4) + 0.5)
    ax.set_yticklabels(FOOT_LABELS[::-1], fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    patches = [plt.Rectangle((0, 0), 1, 1, fc=FOOT_COLORS[fi]) for fi in range(4)]
    fig.legend(patches, FOOT_LABELS, loc='lower right', fontsize=8,
               frameon=True, title='Foot', title_fontsize=8)

    path = os.path.join(save_dir, 'posTracking_gait_cycles.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling average with edge-value padding — no zero-artifact at boundaries."""
    pad    = w // 2
    padded = np.pad(x, pad, mode='edge')
    return np.convolve(padded, np.ones(w) / w, mode='valid')[:len(x)]


def _panel(ax, t_rel, raw_mean, raw_std, smooth_w, foot_idx, show_legend):
    """Draw one foot's data (band + raw mean + smoothed trend) onto ax."""
    trend = _smooth(raw_mean, smooth_w)
    c = FOOT_COLORS[foot_idx]
    ax.fill_between(t_rel, raw_mean - raw_std, raw_mean + raw_std,
                    color=c, alpha=0.12)
    ax.plot(t_rel, raw_mean, color=c, lw=0.7, alpha=0.35)
    ax.plot(t_rel, trend,    color=c, lw=2.0,
            label=FOOT_LABELS[foot_idx] if show_legend else None)


def _plot_foot_kinematics(data: dict, save_dir: str):
    """Local foot speed and |acceleration| — mean ± 1σ over N envs.

    Paper layout: 4 rows × 3 cols
      Row 0: foot speed   — speed-up   (pace→trot | pace→canter | trot→canter)
      Row 1: foot |acc|   — speed-up
      Row 2: foot speed   — speed-down (trot→pace | canter→trot | canter→pace)
      Row 3: foot |acc|   — speed-down

    Smoothing uses edge-value padding so trend lines are valid at both ends.
    """
    foot_pos    = data['foot_pos']      # [T, N, 4, 3]
    dt          = data['dt']
    trans_steps = data['trans_steps']   # 6 entries
    trans_labels= data['trans_labels']
    stats_win   = data['stats_win']

    T = foot_pos.shape[0]

    foot_vel   = np.diff(foot_pos, axis=0) / dt            # [T-1, N, 4, 3]
    foot_acc   = np.diff(foot_vel, axis=0) / dt            # [T-2, N, 4, 3]
    foot_speed = np.linalg.norm(foot_vel, axis=-1)          # [T-1, N, 4]
    foot_amag  = np.linalg.norm(foot_acc, axis=-1)          # [T-2, N, 4]

    SMOOTH_W = 10   # ~0.2 s at 50 Hz

    # Transition grouping
    # trans order: pace→trot(0), trot→pace(1), pace→canter(2),
    #              canter→trot(3), trot→canter(4), canter→pace(5)
    UP   = [0, 2, 4]   # speed-up   transitions (cols 0-2)
    DOWN = [1, 3, 5]   # speed-down transitions (cols 0-2)

    fig, axes = plt.subplots(4, 3, figsize=(13, 9), constrained_layout=True)

    # Share y-axis within each metric row group
    for col in range(1, 3):
        axes[0, col].sharey(axes[0, 0])   # speed-up   speed
        axes[1, col].sharey(axes[1, 0])   # speed-up   |acc|
        axes[2, col].sharey(axes[2, 0])   # speed-down speed
        axes[3, col].sharey(axes[3, 0])   # speed-down |acc|

    for group_row, trans_list in enumerate([UP, DOWN]):
        spd_row = group_row * 2       # 0 or 2
        acc_row = group_row * 2 + 1   # 1 or 3

        for col, ti in enumerate(trans_list):
            ts  = trans_steps[ti]
            lbl = trans_labels[ti]
            show_legend = (group_row == 0 and col == 2)

            # ── Speed panel ─────────────────────────────────────────────────
            w0_s = max(0,     ts - stats_win)
            w1_s = min(T - 1, ts + stats_win)
            t_s  = (np.arange(w0_s, w1_s) - ts) * dt
            spd  = foot_speed[w0_s:w1_s]   # [W, N, 4]

            ax_s = axes[spd_row, col]
            for fi in range(4):
                _panel(ax_s, t_s,
                       spd[:, :, fi].mean(axis=1),
                       spd[:, :, fi].std(axis=1),
                       SMOOTH_W, fi, show_legend)

            ax_s.axvline(0, color='#333', lw=1.0, ls='--', alpha=0.8)
            ax_s.set_title(lbl, fontsize=10, fontweight='bold')
            ax_s.spines[['top', 'right']].set_visible(False)
            ax_s.tick_params(labelsize=8)
            ax_s.set_xlabel('')   # no x-label on speed row (acc row carries it)
            if col == 0:
                ax_s.set_ylabel('Foot speed (m/s)', fontsize=9)

            # ── Acceleration panel ───────────────────────────────────────────
            w0_a = max(0,     ts - stats_win)
            w1_a = min(T - 2, ts + stats_win)
            t_a  = (np.arange(w0_a, w1_a) - ts) * dt
            acc  = foot_amag[w0_a:w1_a]    # [W, N, 4]

            ax_a = axes[acc_row, col]
            for fi in range(4):
                _panel(ax_a, t_a,
                       acc[:, :, fi].mean(axis=1),
                       acc[:, :, fi].std(axis=1),
                       SMOOTH_W, fi, False)

            ax_a.axvline(0, color='#333', lw=1.0, ls='--', alpha=0.8)
            ax_a.spines[['top', 'right']].set_visible(False)
            ax_a.tick_params(labelsize=8)
            ax_a.set_xlabel('Time relative to transition (s)', fontsize=8)
            if col == 0:
                ax_a.set_ylabel('Foot |acc| (m/s²)', fontsize=9)

    # Row-group labels on the left margin
    for g, label in enumerate(['Speed-up', 'Speed-down']):
        mid_ax = axes[g * 2, 0]
        mid_ax.annotate(
            label, xy=(0, 0.5), xytext=(-0.32, 0.5),
            xycoords='axes fraction', textcoords='axes fraction',
            fontsize=10, fontweight='bold', color='#444',
            rotation=90, va='center', ha='center')

    # Single legend in top-right speed panel
    axes[0, 2].legend(title='Foot', title_fontsize=8, fontsize=8,
                      loc='upper right', frameon=True,
                      framealpha=0.85, edgecolor='#ccc')

    path = os.path.join(save_dir, 'posTracking_foot_kinematics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = get_args()
    play(args)
