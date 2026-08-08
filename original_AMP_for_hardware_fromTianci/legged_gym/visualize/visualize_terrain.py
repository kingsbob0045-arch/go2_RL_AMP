"""
Visualize the Go2 task-policy curriculum terrain in IsaacGym.

No policy checkpoint is needed — the environment is created and stepped with
zero actions so you can inspect the terrain geometry in the viewer.

Usage:
    python legged_gym/scripts/visualize_terrain_isaacgym.py \
        --task go2_task_policy \
        --num_envs 64 \
        --sim_device cuda:0

Viewer controls (IsaacGym defaults):
    Left-drag       : rotate
    Right-drag      : pan
    Scroll          : zoom
    V               : toggle viewer sync
    ESC             : quit
"""

import isaacgym          # must be first
import torch

from legged_gym.envs import *          # registers all tasks
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args()

    # ── Load config ──────────────────────────────────────────────────────────
    env_cfg, _ = task_registry.get_cfgs(name=args.task)

    # ── Overrides for visualization ──────────────────────────────────────────
    env_cfg.env.num_envs          = min(getattr(args, 'num_envs', 64), 200)
    env_cfg.noise.add_noise       = False
    env_cfg.domain_rand.push_robots          = False
    env_cfg.domain_rand.randomize_friction   = False
    env_cfg.domain_rand.randomize_base_mass  = False
    env_cfg.domain_rand.randomize_gains      = False
    env_cfg.env.episode_length_s  = 20    # let robots stand ~20 s before reset

    # Keep the full curriculum terrain grid (num_rows, num_cols, proportions
    # stay as configured) so all terrain types are visible.
    # Optionally reduce terrain size for faster mesh generation:
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 10

    # ── Camera: overview of the whole grid ──────────────────────────────────
    # Terrain origin starts at border_size from (0,0).
    # Each cell is terrain_length × terrain_width = 8 m × 8 m.
    # With num_rows=10, num_cols=20 → 80 m × 160 m of terrain.
    border     = getattr(env_cfg.terrain, 'border_size', 25.)
    t_len      = getattr(env_cfg.terrain, 'terrain_length', 8.)
    t_wid      = getattr(env_cfg.terrain, 'terrain_width',  8.)
    nr         = env_cfg.terrain.num_rows
    nc         = env_cfg.terrain.num_cols
    cx = border + nr * t_len / 2.   # centre X
    cy = border + nc * t_wid / 2.   # centre Y
    # Camera elevated and offset so the grid fills the view
    cam_height = max(nr * t_len, nc * t_wid) * 0.6
    env_cfg.viewer.pos    = [cx - cam_height * 0.5, cy, cam_height]
    env_cfg.viewer.lookat = [cx, cy, 0.]

    # ── Create environment ───────────────────────────────────────────────────
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, _ = env.reset()

    print("\n" + "=" * 60)
    print(" IsaacGym Terrain Viewer")
    print(f"  Terrain: {nr} rows × {nc} cols")
    print(f"  Proportions: {env_cfg.terrain.terrain_proportions}")
    print(f"  Types: smooth-slope / rough-slope / stairs↓ / stairs↑ / obstacles")
    print(f"  num_envs: {env.num_envs}")
    print("  Viewer controls: drag=rotate, scroll=zoom, V=sync, ESC=quit")
    print("=" * 60 + "\n")

    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    # ── Render loop ──────────────────────────────────────────────────────────
    while True:
        obs, _, _, dones, _, _, _ = env.step(zero_actions)

        # Quit when the viewer window is closed
        if hasattr(env, 'gym') and hasattr(env, 'viewer'):
            if env.gym.query_viewer_has_closed(env.viewer):
                break


if __name__ == '__main__':
    main()
