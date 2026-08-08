"""Gym registrations for the migrated Go2 tasks."""

import gymnasium as gym


def _register(task_id: str, cfg_name: str, runner_name: str) -> None:
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.go2_env:Go2AmpEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_env_cfg:{cfg_name}",
            "amp_runner_cfg_entry_point": f"{__name__}.runner_cfg:{runner_name}",
        },
    )


_register("Isaac-Go2-AMP-Direct-v0", "Go2AmpEnvCfg", "go2_amp_runner_cfg")
_register("Isaac-Go2-SECAMP-Direct-v0", "Go2SecampEnvCfg", "go2_secamp_runner_cfg")
_register("Isaac-Go2-Rough-Residual-Direct-v0", "Go2RoughResidualEnvCfg", "go2_rough_runner_cfg")

