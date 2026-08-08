"""Configs for the bundled AMP/RSL-RL fork."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]


def _motions(pattern: str) -> list[str]:
    files = sorted(str(path) for path in PROJECT_ROOT.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No motion files match {PROJECT_ROOT / pattern}")
    return files


def _base() -> dict:
    return {
        "seed": 1,
        "policy": {
            "init_noise_std": 1.0, "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128], "activation": "elu",
        },
        "algorithm": {
            "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2,
            "entropy_coef": 0.01, "num_learning_epochs": 5, "num_mini_batches": 4,
            "learning_rate": 1.0e-3, "schedule": "adaptive", "gamma": 0.99,
            "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0,
        },
        "runner": {
            "policy_class_name": "ActorCritic", "algorithm_class_name": "AMPPPO",
            "num_steps_per_env": 24, "max_iterations": 50000, "save_interval": 1000,
            "experiment_name": "go2_amp_isaaclab", "run_name": "",
            "resume": False, "load_run": -1, "checkpoint": -1, "wandb_enable": False,
        },
    }


def go2_amp_runner_cfg() -> dict:
    cfg = _base()
    cfg["algorithm"]["amp_replay_buffer_size"] = 1_000_000
    cfg["runner"].update({
        "amp_reward_coef": 2.0,
        "amp_motion_files": _motions("datasets/mocap_motions_go2/*"),
        "amp_motion_reorder": False,
        "amp_num_preload_transitions": 2_000_000,
        "amp_task_reward_lerp": 0.3,
        "amp_discr_hidden_dims": [1024, 512],
        "min_normalized_std": [0.05, 0.02, 0.05] * 4,
    })
    return cfg


def go2_secamp_runner_cfg() -> dict:
    cfg = go2_amp_runner_cfg()
    cfg["seed"] = 77
    cfg["policy"].update(actor_hidden_dims=[1024, 512, 256], critic_hidden_dims=[1024, 512, 256])
    cfg["algorithm"]["disc_learning_rate"] = 1.0e-3
    cfg["runner"].update({
        "algorithm_class_name": "SECAMPPPO", "experiment_name": "go2_secamp_isaaclab",
        "save_interval": 200, "amp_motion_files": _motions("datasets/camp/*"),
        "lerp_schedule_enabled": False, "skill_insert": "project",
    })
    return cfg


def go2_rough_runner_cfg() -> dict:
    cfg = _base()
    cfg["policy"] = {
        "init_noise_std": 1.0, "num_residual_actions": 12, "num_skill_dims": 3,
        "backbone_hidden_dims": [512, 256], "critic_hidden_dims": [512, 256],
        "activation": "elu", "min_std_residual": [0.05, 0.02, 0.05] * 4,
        "max_std_residual": [3.0, 3.0, 3.0] * 4, "fixed_skill_std": True,
        "min_std_skill": [0.05] * 3, "max_std_skill": [2.0] * 3,
    }
    cfg["runner"].update({
        "policy_class_name": "ActorCriticTwoHead", "algorithm_class_name": "PPO",
        "experiment_name": "go2_rough_isaaclab", "save_interval": 200,
    })
    return cfg

