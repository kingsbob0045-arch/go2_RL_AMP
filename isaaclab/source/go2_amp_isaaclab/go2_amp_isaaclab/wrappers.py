"""Adapter from Gymnasium DirectRLEnv to the legacy custom AMP runner API."""

import torch


class LegacyAmpVecEnvWrapper:
    def __init__(self, env):
        self.gym_env = env
        self.env = env.unwrapped
        for name in (
            "num_envs", "num_obs", "num_privileged_obs", "num_actions", "max_episode_length",
            "episode_length_buf", "device", "dt", "include_history_steps", "dof_pos_limits",
            "amp_horizon", "amp_num_obs", "enable_skill", "skill_cmd_dim",
        ):
            setattr(self, name, getattr(self.env, name))
        self._observations = None

    def reset(self):
        self._observations, _ = self.gym_env.reset()
        return self.get_observations(), self.get_privileged_observations()

    def step(self, actions: torch.Tensor):
        obs, rewards, terminated, truncated, extras = self.gym_env.step(actions)
        self._observations = obs
        return (
            obs["policy"], obs.get("critic"), rewards, terminated | truncated, extras,
            self.env._terminal_env_ids, self.env._terminal_amp_states,
        )

    def get_observations(self):
        if self._observations is None:
            self.reset()
        return self._observations["policy"]

    def get_privileged_observations(self):
        if self._observations is None:
            self.reset()
        return self._observations.get("critic")

    def get_amp_observations(self):
        return self.env.get_amp_observations()

    def get_amp_observation_buf(self):
        return self.env.get_amp_observation_buf()

    def get_skill_cmd_buf(self):
        return self.env.get_skill_cmd_buf()

    def close(self):
        self.gym_env.close()

