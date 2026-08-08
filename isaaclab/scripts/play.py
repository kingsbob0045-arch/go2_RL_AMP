#!/usr/bin/env python3
"""Play and optionally export a migrated Isaac Lab checkpoint."""

import argparse
import copy
import json
import math
import os
import struct
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Go2-AMP-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--export", type=str, default=None, help="Output TorchScript .pt path")
parser.add_argument("--steps", type=int, default=None, help="Optional replay-step limit; useful for smoke tests")
parser.add_argument(
    "--mode", choices=("autonomous", "joystick", "waypoint"), default="autonomous",
    help="SECAMP command source: environment waypoints, PS4 joystick, or the legacy fixed waypoint demo.",
)
parser.add_argument("--joystick-device", default="/dev/input/js0", help="Linux joystick device used with --mode joystick")
parser.add_argument("--deadzone", type=float, default=0.1, help="PS4 left-stick deadzone")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import go2_amp_isaaclab  # noqa: E402, F401
from go2_amp_isaaclab.tasks.go2_amp.go2_env_cfg import GO2_JOINT_NAMES  # noqa: E402
from go2_amp_isaaclab.tasks.go2_amp.runner_cfg import (  # noqa: E402
    go2_amp_runner_cfg, go2_rough_runner_cfg, go2_secamp_runner_cfg,
)
from go2_amp_isaaclab.wrappers import LegacyAmpVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import AMPOnPolicyRunner, Go2RoughRunner, Go2SECAMPRunner  # noqa: E402

TASKS = {
    "Isaac-Go2-AMP-Direct-v0": (go2_amp_runner_cfg, AMPOnPolicyRunner, "amp"),
    "Isaac-Go2-SECAMP-Direct-v0": (go2_secamp_runner_cfg, Go2SECAMPRunner, "secamp"),
    "Isaac-Go2-Rough-Residual-Direct-v0": (go2_rough_runner_cfg, Go2RoughRunner, "rough_residual"),
}

SKILL_NAMES = ("pace", "trot", "canter")
# Keep this schedule and its 0=pace, 1=trot, 2=canter convention in sync with deploy_secamp.py.
LEGACY_SECAMP_WAYPOINTS = (
    (0.0, 0.0, 0), (8.0, 0.0, 0), (16.0, 0.0, 1),
    (24.0, 0.0, 1), (32.0, 0.0, 2), (40.0, 0.0, 2), (48.0, 0.0, -1),
)


class LegacyPs4Joystick:
    """Read Linux joystick events directly, matching the former ROS2 /joy mapping.

    Button indices: X=0 (pace), O=1 (trot), triangle=2 (canter), L1=4, R1=5.
    Axis indices: left stick vertical=1 (forward), horizontal=0 (lateral).  L2 and R2
    are axes 2 and 5; pressing both fully exits the replay, as in low_level_ctrl.cpp.
    """

    def __init__(self, device_path: str, deadzone: float):
        self._device_path = device_path
        print(f"Opening joystick device: {device_path}", flush=True)
        try:
            self._fd = os.open(device_path, os.O_RDONLY | os.O_NONBLOCK)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Joystick device not found: {device_path}. Check /dev/input/js* and use --joystick-device if needed."
            ) from exc
        except PermissionError as exc:
            raise RuntimeError(
                f"Cannot read {device_path}. Add the current user to the input group, then log out and back in."
            ) from exc
        self._deadzone = deadzone
        self._axes: list[float] = []
        self._buttons: list[bool] = []
        self._previous_buttons: list[bool] = []
        print(f"PS4 controller: {device_path}")
        print("Hold L1+R1 to command; left stick controls direction; X/O/triangle select pace/trot/canter; L2+R2 exits.")

    def update(self, task_env) -> bool:
        """Write the command/skill tensors.  Return True when replay should exit."""
        self._read_events()
        axes, buttons = self._axes, self._buttons

        # Same enable, direction, dead-zone, and no-backwards rules as deploy_secamp.py.
        enabled = len(buttons) > 5 and buttons[4] and buttons[5]
        forward = axes[1] if len(axes) > 1 and abs(axes[1]) > self._deadzone else 0.0
        lateral = axes[0] if len(axes) > 0 and abs(axes[0]) > self._deadzone else 0.0
        norm = math.hypot(forward, lateral)
        if not enabled or forward < 0.0 or norm < 1.0e-6:
            forward, lateral = 0.0, 0.0
        else:
            forward, lateral = forward / norm, lateral / norm
        task_env.commands[:, 0] = forward
        task_env.commands[:, 1] = lateral
        task_env.commands[:, 2] = 0.0

        if len(self._previous_buttons) != len(buttons):
            self._previous_buttons = [False] * len(buttons)
        for skill, button_id in enumerate((0, 1, 2)):
            if button_id < len(buttons) and buttons[button_id] and not self._previous_buttons[button_id]:
                _set_skill(task_env, skill)
                print(f"Skill -> {SKILL_NAMES[skill]}")
        self._previous_buttons = buttons

        left_trigger = axes[2] if len(axes) > 2 else 1.0
        right_trigger = axes[5] if len(axes) > 5 else 1.0
        return left_trigger <= -0.99 and right_trigger <= -0.99

    def _read_events(self) -> None:
        """Drain /dev/input/js*; Linux joystick_event is uint32, int16, uint8, uint8."""
        while True:
            try:
                raw = os.read(self._fd, 8)
            except BlockingIOError:
                return
            if not raw:
                return
            _, value, event_type, number = struct.unpack("IhBB", raw)
            event_type &= ~0x80  # JS_EVENT_INIT
            if event_type == 0x02:  # JS_EVENT_AXIS
                self._ensure_size(self._axes, number, 0.0)
                self._axes[number] = max(-1.0, min(1.0, value / 32767.0))
            elif event_type == 0x01:  # JS_EVENT_BUTTON
                self._ensure_size(self._buttons, number, False)
                self._buttons[number] = bool(value)

    @staticmethod
    def _ensure_size(values: list, index: int, fill_value) -> None:
        if len(values) <= index:
            values.extend([fill_value] * (index + 1 - len(values)))


def _set_skill(task_env, skill: int) -> None:
    """Set a single SECAMP one-hot skill for every displayed environment."""
    task_env.skill_commands.zero_()
    task_env.skill_commands[:, skill] = 1.0


def _apply_legacy_waypoint(task_env, start_time: float) -> None:
    """Apply the former deploy_secamp.py fixed pace/trot/canter demonstration."""
    elapsed = time.monotonic() - start_time
    _, direction_deg, skill = next(item for item in reversed(LEGACY_SECAMP_WAYPOINTS) if elapsed >= item[0])
    if skill < 0:
        task_env.commands.zero_()
        return
    direction = math.radians(direction_deg)
    task_env.commands[:, 0] = math.cos(direction)
    task_env.commands[:, 1] = math.sin(direction)
    task_env.commands[:, 2] = 0.0
    _set_skill(task_env, skill)


class ExportedPolicy(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = copy.deepcopy(actor_critic).cpu().eval()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_critic.act_inference(observations)


def main():
    cfg_factory, runner_type, policy_kind = TASKS[args.task]
    if args.mode != "autonomous" and policy_kind != "secamp":
        raise ValueError("--mode joystick/waypoint is currently defined for Isaac-Go2-SECAMP-Direct-v0 only")
    if args.mode == "joystick" and args.num_envs != 1:
        raise ValueError("Use --num_envs 1 with --mode joystick so the displayed robot has one controller")
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.add_observation_noise = False
    env_cfg.reference_state_initialization = False
    env_cfg.events = None
    if args.mode != "autonomous":
        # Commands are overwritten by the legacy source below.  Disable random waypoint and command updates.
        env_cfg.waypoint_mode = False
        env_cfg.command_resampling_time_s = env_cfg.episode_length_s + 1.0
    train_cfg = cfg_factory()
    if "amp_num_preload_transitions" in train_cfg["runner"]:
        train_cfg["runner"]["amp_num_preload_transitions"] = 1
    env = LegacyAmpVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    runner = runner_type(env, train_cfg, log_dir=None, device=args.device)
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    runner.load(args.checkpoint, load_optimizer=False)
    print("Checkpoint loaded.", flush=True)
    policy = runner.get_inference_policy(device=env.device)
    print("Inference policy ready.", flush=True)
    observations, _ = env.reset()
    print("Environment reset.", flush=True)
    joystick = LegacyPs4Joystick(args.joystick_device, args.deadzone) if args.mode == "joystick" else None
    if joystick is not None:
        print("Joystick input ready.", flush=True)
    waypoint_start = time.monotonic()

    if args.export:
        output = Path(args.export).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        traced = torch.jit.trace(ExportedPolicy(runner.alg.actor_critic), observations[:1].cpu())
        traced.save(str(output))
        metadata = {
            "format_version": 1, "source": "Isaac Lab", "task": args.task,
            "policy_kind": policy_kind, "joint_order": list(GO2_JOINT_NAMES),
            "observation_dim": observations.shape[-1], "action_dim": env.num_actions,
            "base_observation_layout": [
                "projected_gravity", "command", "joint_position", "joint_velocity", "last_action",
            ],
            "action_scale": env.env.cfg.action_scale, "control_dt": env.dt,
        }
        sidecar = output.with_suffix(".json")
        sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Exported {output} and {sidecar}")

    replay_steps = 0
    while simulation_app.is_running():
        if joystick is not None:
            if joystick.update(env.env):
                print("L2+R2 pressed: stopping Isaac Sim replay.")
                break
        elif args.mode == "waypoint":
            _apply_legacy_waypoint(env.env, waypoint_start)
        with torch.inference_mode():
            observations, _, _, _, _, _, _ = env.step(policy(observations))
        replay_steps += 1
        if args.steps is not None and replay_steps >= args.steps:
            break
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
