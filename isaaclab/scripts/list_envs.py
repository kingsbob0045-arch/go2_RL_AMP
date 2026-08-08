#!/usr/bin/env python3
import gymnasium as gym
import go2_amp_isaaclab  # noqa: F401

for env_id in sorted(spec.id for spec in gym.registry.values() if spec.id.startswith("Isaac-Go2-")):
    print(env_id)

