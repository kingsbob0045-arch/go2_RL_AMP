from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
import argparse

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Video recording flag (parsed separately so it doesn't conflict with get_args)
# Usage: python play_camp.py --task ... --record_video
# ---------------------------------------------------------------------------
_vparser = argparse.ArgumentParser(add_help=False)
_vparser.add_argument('--record_video', action='store_true', default=False,
                      help='Record MP4 video to results/<run_name>/camp_eval.mp4')
_vparser.add_argument('--video_fps', type=int, default=50,
                      help='Video output frame rate (default: 50 = real-time)')
_VARGS, _ = _vparser.parse_known_args()

_TMP_FRAME = f'/tmp/_camp_frame_{os.getpid()}.png'   # per-process temp file


# One-hot encoding for each skill (same as camp_motion_loader.py)
_SKILLS_OneHot = {
    "pace":    [0, 0, 0, 0, 1],
    "trot":    [0, 0, 0, 1, 0],
    "gallop":  [0, 0, 1, 0, 0],
    "canter":  [0, 1, 0, 0, 0],
    "jump":    [1, 0, 0, 0, 0],
}

# Ordered skill sweep sequence
SKILL_SEQUENCE = ["pace", "trot", "gallop", "canter", "jump"]

# Per-skill command velocities derived from dataset vx ranges:
#   pace   vx [0.60, 1.14] → 1.0 m/s
#   trot   vx [0.86, 2.03] → 1.5 m/s
#   gallop vx [1.56, 2.33] → 2.0 m/s
#   canter vx [1.11, 3.50] → 2.5 m/s
#   jump   vx [0.72, 1.76] → 1.2 m/s
SKILL_VEL = {
    "pace":   1.0,
    "trot":   2.0,
    "gallop": 2.5,
    "canter": 3.0,
    "jump":   0.5,
}

# Steps to hold each skill (50 Hz × 10 s = 500 steps)
STEPS_PER_SKILL = 500
# Warm-up steps to discard at the beginning of each skill phase (transient)
WARMUP_STEPS = 100
# How many steady-state steps to include in the contact plot
PLOT_WINDOW = 150   # ≈ 3 s at 50 Hz

FOOT_LABELS = ['FL', 'FR', 'RL', 'RR']
FOOT_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

# Camera offset relative to the robot base (world frame, robot moves in +x)
# [behind, side, height]
CAM_OFFSET = np.array([0.5, 1.5, 1.0])


def play_camp(args):
    # Video recording requires a viewer
    if _VARGS.record_video:
        args.headless = False

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 4)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.reference_state_initialization = False
    # Velocity will be overridden per-skill; set a safe default range here
    env_cfg.commands.ranges.lin_vel_x = [0.9, 0.9]
    env_cfg.commands.ranges.lin_vel_y = [0., 0.]
    env_cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    env_cfg.env.episode_length_s = 999

    train_cfg.runner.amp_num_preload_transitions = 1

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    skill_cmd_dim = env.skill_cmd_dim
    num_envs = env.num_envs
    robot_idx = 0   # which env to record contacts from

    # ---- Save directory (needed for both plots and video) ----------------
    _load_run = getattr(train_cfg.runner, 'load_run', None)
    run_name = str(_load_run if (_load_run and _load_run != -1)
                   else getattr(train_cfg.runner, 'run_name', 'unknown_run'))
    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'results', run_name)
    os.makedirs(save_dir, exist_ok=True)

    # ---- Video writer ----------------------------------------------------
    video_writer = None
    video_path   = None
    if _VARGS.record_video:
        try:
            import imageio
            video_path   = os.path.join(save_dir, 'camp_eval.mp4')
            video_writer = imageio.get_writer(
                video_path, fps=_VARGS.video_fps, quality=8, macro_block_size=None)
            print(f"[Video] Recording → {video_path}  (fps={_VARGS.video_fps})")
        except ImportError:
            print("[Video] imageio not installed — skipping. "
                  "Install with: pip install imageio[ffmpeg]")

    # Build skill tensors: [num_envs, skill_cmd_dim]
    skill_tensors = {
        name: torch.tensor(vec, dtype=torch.float32, device=env.device)
               .unsqueeze(0).expand(num_envs, -1)
        for name, vec in _SKILLS_OneHot.items()
    }

    def set_skill(skill_name: str):
        """Force all envs to the given one-hot skill command and patch obs."""
        env.skill_cmd_buf[:] = skill_tensors[skill_name]
        # Patch the skill slice so the policy sees the new command immediately
        obs[:, -skill_cmd_dim:] = skill_tensors[skill_name]

    def set_vel(vx: float):
        """Override command velocity for all envs."""
        env.commands[:, 0] = vx
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0

    # Storage for contact logs: skill → [PLOT_WINDOW, 4]
    contact_logs: dict[str, np.ndarray] = {}

    print("\n" + "=" * 55)
    print("  CAMP skill sweep")
    print(f"  Sequence: {' → '.join(SKILL_SEQUENCE)}")
    print(f"  {STEPS_PER_SKILL} steps/skill  ({STEPS_PER_SKILL * env.dt:.1f} s)  "
          f"warmup={WARMUP_STEPS} steps")
    print("=" * 55 + "\n")

    step = 0
    for skill_name in SKILL_SEQUENCE:
        vx = SKILL_VEL[skill_name]
        set_skill(skill_name)
        set_vel(vx)
        print(f"[step {step:>5}]  Gait → \033[1;33m{skill_name.upper()}\033[0m  "
              f"cmd_vx={vx:.1f} m/s  one-hot={_SKILLS_OneHot[skill_name]}")

        phase_contacts = []

        for local_step in range(STEPS_PER_SKILL):
            with torch.inference_mode():
                actions = policy(obs.detach())
            obs, _, _, _, _, _, _ = env.step(actions.detach())

            # Re-apply skill + vel after every step (reset_idx may resample)
            set_skill(skill_name)
            set_vel(vx)

            # Camera follow
            if not args.headless:
                look_at = env.root_states[robot_idx, :3].cpu().numpy().astype(np.float64)
                env.set_camera(look_at + CAM_OFFSET, look_at)

            # Frame capture for video
            if video_writer is not None and env.viewer is not None:
                import imageio
                env.gym.write_viewer_image_to_file(env.viewer, _TMP_FRAME)
                frame = imageio.imread(_TMP_FRAME)
                video_writer.append_data(frame[:, :, :3])

            step += 1

            # Record foot contacts after warm-up
            if local_step >= WARMUP_STEPS:
                contacts = (
                    env.contact_forces[robot_idx, env.feet_indices, 2] > 1.0
                ).cpu().numpy().astype(np.float32)   # [4]
                phase_contacts.append(contacts)

            if local_step > 0 and local_step % 100 == 0:
                base_vel = env.base_lin_vel[robot_idx, 0].item()
                print(f"  step {step:>5}  skill={skill_name:<7}  "
                      f"base_vel_x={base_vel:+.2f} m/s")

        # Keep only the last PLOT_WINDOW frames for plotting
        arr = np.stack(phase_contacts, axis=0)          # [N_record, 4]
        contact_logs[skill_name] = arr[-PLOT_WINDOW:]   # [PLOT_WINDOW, 4]

    # Close video writer
    if video_writer is not None:
        video_writer.close()
        if os.path.exists(_TMP_FRAME):
            os.remove(_TMP_FRAME)
        print(f"[Video] Saved → {video_path}")

    print("\nSkill sweep complete. Plotting gait cycles…")
    _plot_gait_cycles(contact_logs, env.dt, save_dir)
    print("Done.")


# ---------------------------------------------------------------------------
# Gait cycle plot
# ---------------------------------------------------------------------------

def _draw_contact_schedule(ax, contact_window, title, dt):
    """Gantt-style stance/swing chart on ax."""
    n_feet = 4
    t_rel = np.arange(len(contact_window)) * dt
    for fi in range(n_feet):
        y = n_feet - 1 - fi          # FL on top, RR on bottom
        contact = contact_window[:, fi].astype(bool)
        padded = np.concatenate(([False], contact, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        bars = [(t_rel[s], t_rel[e - 1] - t_rel[s] + dt)
                for s, e in zip(starts, ends)]
        if bars:
            ax.broken_barh(bars, (y + 0.08, 0.84),
                           facecolors=FOOT_COLORS[fi], edgecolors='none')

    total_t = len(contact_window) * dt
    ax.set_xlim(0, total_t)
    ax.set_ylim(0, n_feet)
    ax.set_yticks(np.arange(n_feet) + 0.5)
    ax.set_yticklabels(FOOT_LABELS[::-1], fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel('Time (s)', fontsize=8, labelpad=2)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    ax.spines[['top', 'right']].set_visible(False)


def _plot_gait_cycles(contact_logs: dict, dt: float, save_dir: str):
    n_skills = len(SKILL_SEQUENCE)
    fig, axes = plt.subplots(1, n_skills,
                             figsize=(4.5 * n_skills, 3.2),
                             constrained_layout=True)
    if n_skills == 1:
        axes = [axes]

    for ax, skill_name in zip(axes, SKILL_SEQUENCE):
        vx = SKILL_VEL[skill_name]
        title = f"{skill_name.capitalize()}\n(cmd vx = {vx:.1f} m/s)"
        _draw_contact_schedule(ax, contact_logs[skill_name], title, dt)

    # Legend
    patches = [plt.Rectangle((0, 0), 1, 1, fc=FOOT_COLORS[fi])
               for fi in range(4)]
    fig.legend(patches, FOOT_LABELS,
               loc='lower right', fontsize=9, frameon=True,
               title='Foot', title_fontsize=9,
               bbox_to_anchor=(1.0, 0.0))

    fig.suptitle('CAMP — Gait Cycle Contact Schedules', fontsize=13, fontweight='bold')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'camp_gait_cycles.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {save_path}")


if __name__ == '__main__':
    args = get_args()
    play_camp(args)
