#!/usr/bin/env python3
"""Evaluate a SECAMP checkpoint with the DETERMINISTIC policy that gets deployed.

Training-time TensorBoard metrics are collected with the stochastic policy, so they say
little about deployment once Policy/mean_noise_std has drifted.  This script runs
act_inference (the distribution mean, exactly what play.py exports and what
deploy_secamp.py loads) and reports survival and reward.

Usage:
    python isaaclab/scripts/eval_checkpoint.py --checkpoint path/to/model_2500.pt
    python isaaclab/scripts/eval_checkpoint.py --checkpoint ... --skill trot --steps 1000
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Go2-SECAMP-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1000, help="Environment steps to simulate")
parser.add_argument("--skill", choices=("pace", "trot", "canter", "all"), default="all")
parser.add_argument("--amp_dataset",
                    choices=("baseline", "mocap_gaits", "kine2go_gaits", "nju_agility", "dogml_gaits"),
                    default=None)
parser.add_argument("--loader_clips", type=int, default=12,
                    help="Cap on motion files loaded for the AMP loader. Evaluation never reads "
                         "expert motions: reference-state initialisation is disabled and the "
                         "imitation reward comes from the checkpoint's own discriminator and "
                         "normalizer, so this only avoids a slow multi-minute load. Use 0 for all.")
parser.add_argument("--json", type=str, default=None, help="Append the result to this JSON file")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import go2_amp_isaaclab  # noqa: E402, F401
from go2_amp_isaaclab.tasks.go2_amp.runner_cfg import go2_secamp_runner_cfg  # noqa: E402
from go2_amp_isaaclab.wrappers import LegacyAmpVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import Go2SECAMPRunner  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AMP_DATASETS = {
    "baseline": "datasets/camp/*",
    "mocap_gaits": "datasets/converted/mocap_gaits/*",
    "kine2go_gaits": "datasets/converted/kine2go_gaits/*.json",
    "nju_agility": "datasets/converted/nju_agility/*.json",
    "dogml_gaits": "datasets/converted/dogml_gaits/*.json",
}
SKILLS = ("pace", "trot", "canter")


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # Deployment conditions: no observation noise, no domain randomisation, and start from
    # the robot's own default pose rather than a motion frame.
    env_cfg.add_observation_noise = False
    env_cfg.reference_state_initialization = False
    env_cfg.events = None
    train_cfg = go2_secamp_runner_cfg()
    train_cfg["runner"]["amp_num_preload_transitions"] = 2000
    if args.amp_dataset:
        pattern = AMP_DATASETS[args.amp_dataset]
        env_cfg.motion_glob = pattern
        motion_files = sorted(str(path) for path in PROJECT_ROOT.glob(pattern))
        if args.loader_clips > 0:
            motion_files = motion_files[:args.loader_clips]
        train_cfg["runner"]["amp_motion_files"] = motion_files

    env = LegacyAmpVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    print("[eval] environment created", flush=True)
    runner = Go2SECAMPRunner(env, train_cfg, log_dir=None, device=args.device)
    print("[eval] runner built", flush=True)
    runner.load(args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    print("[eval] checkpoint loaded", flush=True)
    task_env = env.env

    reported_std = float(runner.alg.actor_critic.std.mean())
    results = {"checkpoint": args.checkpoint, "reported_noise_std": round(reported_std, 3),
               "steps": args.steps, "num_envs": args.num_envs, "skills": {}}

    skills = SKILLS if args.skill == "all" else (args.skill,)
    for skill_name in skills:
        print(f"[eval] running skill {skill_name}", flush=True)
        observations, _ = env.reset()
        skill_index = SKILLS.index(skill_name)

        completed_lengths: list[float] = []
        current_length = torch.zeros(env.num_envs, device=env.device)
        task_total = torch.zeros(env.num_envs, device=env.device)
        imitation_total = torch.zeros(env.num_envs, device=env.device)
        terminations = 0

        for _ in range(args.steps):
            # Pin the skill every step; _reset_idx() re-randomises it on episode restart.
            task_env.skill_commands.zero_()
            task_env.skill_commands[:, skill_index] = 1.0
            amp_observations = env.get_amp_observations()
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, rewards, dones, _, _, terminal_amp = env.step(actions)
                next_amp = env.get_amp_observations()
                _, _, imitation, task = runner.alg.discriminator.predict_amp_reward(
                    amp_observations, next_amp, rewards.to(env.device), dt=env.dt,
                    normalizer=runner.alg.amp_normalizer)

            current_length += 1
            task_total += task
            imitation_total += imitation
            done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            if len(done_ids):
                # max_episode_length is a time-out, anything shorter is a fall.
                lengths = current_length[done_ids]
                terminations += int((lengths < env.max_episode_length - 1).sum())
                completed_lengths.extend(lengths.cpu().tolist())
                current_length[done_ids] = 0.0
                task_total[done_ids] = 0.0
                imitation_total[done_ids] = 0.0

        survived = float(current_length.mean())
        mean_length = (sum(completed_lengths) / len(completed_lengths)
                       if completed_lengths else float(args.steps))
        imitation_ceiling = 2.0 * env.dt  # amp_reward_coef * 1.0 * dt
        alive_mask = current_length > 0
        imitation_rate = float((imitation_total[alive_mask] / current_length[alive_mask]).mean()
                               / imitation_ceiling) if alive_mask.any() else 0.0
        task_rate = float((task_total[alive_mask] / current_length[alive_mask]).mean()
                          / env.dt) if alive_mask.any() else 0.0

        results["skills"][skill_name] = {
            "completed_episodes": len(completed_lengths),
            "falls": terminations,
            "mean_completed_length": round(mean_length, 1),
            "mean_uninterrupted_length": round(survived, 1),
            "imitation_fraction_of_ceiling": round(imitation_rate, 4),
            "forward_speed_m_s": round(task_rate / 2.0, 3),  # tracking_position_scale = 2.0
        }
        entry = results["skills"][skill_name]
        print(f"  {skill_name:<7} falls={entry['falls']:<5} "
              f"mean_len={entry['mean_completed_length']:<7.1f} "
              f"imitation={entry['imitation_fraction_of_ceiling']:.1%} "
              f"speed={entry['forward_speed_m_s']:.2f} m/s", flush=True)

    print(json.dumps(results, indent=2))
    if args.json:
        path = Path(args.json)
        existing = json.loads(path.read_text()) if path.exists() else []
        existing.append(results)
        path.write_text(json.dumps(existing, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
