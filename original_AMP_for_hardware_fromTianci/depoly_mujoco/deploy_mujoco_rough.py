"""
Sim2Sim deployment for go2_rough: residual policy + frozen motion prior.

Inference pipeline (per control step):
  1. Build single-step obs (42-dim) from MuJoCo state
  2. Insert into rolling history buffer  →  policy_obs (5 × 42 = 210-dim)
  3. actor_policy(policy_obs)            →  [residual(12), skill_logits(3)]
  4. skill_cmd = softmax(skill_logits)
  5. Build prior_obs:
       - copy single-step obs
       - patch obs[3:6] with body-frame direction × 0.5  (match training)
       - append skill_cmd                                 →  45-dim
  6. motion_prior(prior_obs)             →  prior_actions (12)
  7. joint_actions = prior_actions + residual_scale × residual
  8. target_dof_pos = joint_actions × action_scale + default_angles

Usage:
    python deploy_mujoco_rough.py go2_rough.yaml
"""

import math
import time
import threading
import argparse

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from legged_gym import LEGGED_GYM_ROOT_DIR


# -----------------------------------------------------------------------
# Joystick reader (background daemon thread)
# -----------------------------------------------------------------------

class JoystickReader(threading.Thread):
    """Polls a gamepad at ~100 Hz and writes velocity commands into `cmd`.

    cmd[0] = lin_vel_x,  cmd[1] = lin_vel_y,  cmd[2] = ang_vel_yaw

    Thread-safe in CPython: numpy in-place slice assignment (cmd[:] = ...)
    is protected by the GIL, so no explicit lock is needed.
    """

    def __init__(self, cfg_joy: dict, cmd: np.ndarray):
        super().__init__(daemon=True, name="joystick")
        self._cfg  = cfg_joy
        self._cmd  = cmd
        self._stop = threading.Event()

    def run(self):
        try:
            import pygame
        except ImportError:
            print("[joystick] pygame not installed — joystick disabled.")
            return

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("[joystick] No gamepad found — using yaml cmd_init.")
            return

        joy = pygame.joystick.Joystick(0)
        joy.init()
        print(f"[joystick] Connected: {joy.get_name()}")

        cfg = self._cfg
        dz  = float(cfg.get("deadzone", 0.05))

        def read_axis(idx: int, sign: float) -> float:
            v = joy.get_axis(int(idx)) * sign
            return v if abs(v) > dz else 0.0

        while not self._stop.is_set():
            pygame.event.pump()
            self._cmd[0] = read_axis(cfg["axis_vx"], cfg.get("sign_vx", 1)) * cfg["max_vx"]
            self._cmd[1] = read_axis(cfg["axis_vy"], cfg.get("sign_vy", 1)) * cfg["max_vy"]
            self._cmd[2] = read_axis(cfg["axis_wz"], cfg.get("sign_wz", 1)) * cfg["max_wz"]
            time.sleep(0.01)   # 100 Hz

    def stop(self):
        self._stop.set()


# -----------------------------------------------------------------------
# Joint reorder: MuJoCo XML (FR/FL/RR/RL) ↔ Isaac Gym URDF (FL/FR/RL/RR)
#
# go2.xml joint order:  FR_hip(0) FR_thigh(1) FR_calf(2)
#                       FL_hip(3) FL_thigh(4) FL_calf(5)
#                       RR_hip(6) RR_thigh(7) RR_calf(8)
#                       RL_hip(9) RL_thigh(10) RL_calf(11)
#
# Isaac Gym URDF order: FL_hip(0) FL_thigh(1) FL_calf(2)
#                       FR_hip(3) FR_thigh(4) FR_calf(5)
#                       RL_hip(6) RL_thigh(7) RL_calf(8)
#                       RR_hip(9) RR_thigh(10) RR_calf(11)
#
# This permutation is self-inverse: REORDER[REORDER] == arange(12)
# - Read  (MuJoCo → Isaac): isaac_q  = mj_q[REORDER]
# - Write (Isaac → MuJoCo): mj_ctrl  = isaac_ctrl[REORDER]
# -----------------------------------------------------------------------
REORDER = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int32)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Project world gravity [0,0,-1] into body frame.

    Args:
        quaternion: MuJoCo convention [w, x, y, z]
    """
    qw, qx, qy, qz = quaternion
    return np.array([
         2.0 * (-qz * qx + qw * qy),
        -2.0 * ( qz * qy + qw * qx),
         1.0 - 2.0 * (qw * qw + qz * qz),
    ], dtype=np.float32)


def pd_control(
    target_q: np.ndarray, q: np.ndarray, kp: np.ndarray,
    dq: np.ndarray, kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp - dq * kd


class HistoryBuffer:
    """Rolling FIFO buffer matching ObservationBuffer in Isaac Gym training.

    Layout: [oldest_obs | ... | newest_obs]  (all concatenated)
    """
    def __init__(self, num_obs: int, num_steps: int):
        self.num_obs = num_obs
        self.num_steps = num_steps
        self.buf = np.zeros(num_obs * num_steps, dtype=np.float32)

    def reset(self, obs: np.ndarray):
        """Fill all history slots with the same obs (called on episode start)."""
        for i in range(self.num_steps):
            self.buf[i * self.num_obs: (i + 1) * self.num_obs] = obs

    def insert(self, obs: np.ndarray):
        """Shift buffer left and append new obs at the end."""
        self.buf[: self.num_obs * (self.num_steps - 1)] = \
            self.buf[self.num_obs: self.num_obs * self.num_steps]
        self.buf[-self.num_obs:] = obs

    def get(self) -> np.ndarray:
        return self.buf.copy()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="YAML config filename in depoly_mujoco/")
    parser.add_argument("--follow-cam", action="store_true",
                        help="Follow robot with a fixed-angle free camera")
    parser.add_argument("--cam-azimuth",   type=float, default=-30.0,
                        help="Follow-cam azimuth in degrees (default: 150)")
    parser.add_argument("--cam-elevation", type=float, default=-20.0,
                        help="Follow-cam elevation in degrees (default: -20)")
    parser.add_argument("--cam-distance",  type=float, default=3.5,
                        help="Follow-cam distance in metres (default: 3.5)")
    parser.add_argument("--record", metavar="FILE.mp4",
                        help="Record simulation to video (requires imageio[ffmpeg])")
    parser.add_argument("--record-res", nargs=2, type=int, default=[1280, 720],
                        metavar=("W", "H"), help="Video resolution (default: 1280 720)")
    args = parser.parse_args()

    config_dir = f"{LEGGED_GYM_ROOT_DIR}/depoly_mujoco"
    with open(f"{config_dir}/{args.config_file}", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # --- Resolve paths --------------------------------------------------
    def resolve(p): return p.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    policy_path      = resolve(cfg["policy_path"])
    prior_path       = resolve(cfg["motion_prior_path"])
    xml_path         = resolve(cfg["xml_path"])

    # --- Sim params -----------------------------------------------------
    sim_duration     = cfg["simulation_duration"]
    sim_dt           = cfg["simulation_dt"]
    decimation       = cfg["control_decimation"]

    kps              = np.array(cfg["kps"],             dtype=np.float32)
    kds              = np.array(cfg["kds"],             dtype=np.float32)
    default_angles   = np.array(cfg["default_angles"],  dtype=np.float32)

    ang_vel_scale    = cfg["ang_vel_scale"]
    dof_pos_scale    = cfg["dof_pos_scale"]
    dof_vel_scale    = cfg["dof_vel_scale"]
    cmd_lin_scale    = cfg["cmd_lin_scale"]
    cmd_ang_scale    = cfg["cmd_ang_scale"]

    action_scale     = cfg["action_scale"]
    res_scale        = cfg["residual_action_scale"]

    num_single_obs   = cfg["num_single_obs"]     # 42
    num_history      = cfg["num_history_steps"]  # 5
    num_dof          = cfg["num_dof"]            # 12
    num_actor_out    = cfg["num_actor_output"]   # 15
    num_prior_obs    = cfg["num_prior_obs"]      # 45

    cmd              = np.array(cfg["cmd_init"], dtype=np.float32)  # [vx, vy, wz]

    # --- Joystick (optional) --------------------------------------------
    joy_reader = None
    joy_cfg    = cfg.get("joystick", {})
    if joy_cfg.get("enabled", False):
        joy_reader = JoystickReader(joy_cfg, cmd)
        joy_reader.start()

    # --- Load models ----------------------------------------------------
    print(f"Loading actor policy:  {policy_path}")
    actor_policy = torch.jit.load(policy_path, map_location="cpu").eval()

    print(f"Loading motion prior:  {prior_path}")
    motion_prior = torch.jit.load(prior_path,  map_location="cpu").eval()

    # --- Load MuJoCo model ----------------------------------------------
    print(f"Loading MuJoCo model:  {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    # --- Video recorder (optional) --------------------------------------
    video_writer = None
    renderer     = None
    if args.record:
        try:
            import imageio
        except ImportError:
            print("[record] imageio not installed. Run:  pip install imageio[ffmpeg]")
            args.record = None
        else:
            rec_fps      = int(round(1.0 / (sim_dt * decimation)))   # 50 Hz
            rec_W, rec_H = args.record_res
            import os
            os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
            m.vis.global_.offwidth  = rec_W
            m.vis.global_.offheight = rec_H
            renderer     = mujoco.Renderer(m, height=rec_H, width=rec_W)
            video_writer = imageio.get_writer(args.record, fps=rec_fps, quality=8)
            print(f"[record] {rec_W}×{rec_H} @ {rec_fps} fps → {args.record}")

    # --- Precompute MuJoCo-order default angles (for PD control) --------
    # default_angles in yaml is Isaac order; reorder for MuJoCo
    default_angles_mj = default_angles[REORDER]

    # --- Runtime buffers ------------------------------------------------
    history          = HistoryBuffer(num_single_obs, num_history)
    joint_actions    = np.zeros(num_dof,     dtype=np.float32)  # Isaac order
    target_dof_pos_mj = default_angles_mj.copy()               # MuJoCo order
    skill_cmd        = np.ones(3, dtype=np.float32) / 3.0  # uniform init

    counter = 0
    first_step = True

    # -----------------------------------------------------------------------
    def build_single_obs(d) -> np.ndarray:
        """Build 42-dim single-step actor observation from MuJoCo state.

        Layout (matches legged_robot.compute_observations, actor view):
          [0:3]   projected_gravity
          [3:6]   [cmd_vx * cmd_lin_scale, cmd_vy * cmd_lin_scale, cmd_wz * cmd_ang_scale]
          [6:18]  (dof_pos - default) * dof_pos_scale
          [18:30] dof_vel * dof_vel_scale
          [30:42] last joint_actions  (12-dim)
        """
        # MuJoCo qpos[3:7] = [w, x, y, z]
        quat   = d.qpos[3:7]
        # Reorder MuJoCo joint data → Isaac Gym order before building obs
        qj     = d.qpos[7 : 7 + num_dof][REORDER]
        dqj    = d.qvel[6 : 6 + num_dof][REORDER]

        gravity = get_gravity_orientation(quat)
        dof_pos = (qj  - default_angles) * dof_pos_scale   # default_angles is Isaac order
        dof_vel = dqj  * dof_vel_scale
        cmd_obs = np.array([
            cmd[0] * cmd_lin_scale,
            cmd[1] * cmd_lin_scale,
            cmd[2] * cmd_ang_scale,
        ], dtype=np.float32)

        obs = np.concatenate([gravity, cmd_obs, dof_pos, dof_vel, joint_actions])
        assert obs.shape[0] == num_single_obs, f"obs shape {obs.shape[0]} != {num_single_obs}"
        return obs.astype(np.float32)


    def build_prior_obs(single_obs: np.ndarray, skill: np.ndarray) -> np.ndarray:
        """Build 45-dim prior obs matching posTracking training format.

        The prior was trained with obs[3:6] = [dx_body*0.5, dy_body*0.5, 0.0].
        Here we convert the velocity command to a unit body-frame direction × 0.5.
        obs[5] (z slot) is always 0.

        Args:
            single_obs: 42-dim obs (already built by build_single_obs)
            skill:      3-dim skill command (softmax probabilities)
        """
        prior_obs = single_obs.copy()

        # Convert velocity command direction to unit vector × 0.5
        vel_xy = cmd[:2]
        norm   = np.linalg.norm(vel_xy)
        if norm > 1e-6:
            dir_body = vel_xy / norm * 0.5
        else:
            dir_body = np.zeros(2, dtype=np.float32)

        prior_obs[3] = dir_body[0]
        prior_obs[4] = dir_body[1]
        prior_obs[5] = 0.0          # z slot unused in training

        return np.concatenate([prior_obs, skill]).astype(np.float32)


    # --- Fixed-angle follow camera for renderer (recording) ----------------
    rec_cam = None
    if video_writer is not None:
        rec_cam            = mujoco.MjvCamera()
        rec_cam.type       = mujoco.mjtCamera.mjCAMERA_FREE
        rec_cam.distance   = args.cam_distance
        rec_cam.azimuth    = args.cam_azimuth
        rec_cam.elevation  = args.cam_elevation

    # -----------------------------------------------------------------------
    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            # Fixed-angle free camera: set angles once, update lookat every step
            if args.follow_cam:
                viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.distance  = args.cam_distance
                viewer.cam.azimuth   = args.cam_azimuth
                viewer.cam.elevation = args.cam_elevation

            start = time.time()

            while viewer.is_running() and time.time() - start < sim_duration:
                step_start = time.time()

                # Body-frame follow: track position + yaw so camera
                # stays at a fixed offset in the robot's base frame.
                if args.follow_cam:
                    qw, qx, qy, qz = d.qpos[3:7]
                    yaw_deg = math.degrees(
                        math.atan2(2.0*(qw*qz + qx*qy),
                                   1.0 - 2.0*(qy*qy + qz*qz)))
                    viewer.cam.lookat[:]  = d.qpos[:3]
                    viewer.cam.azimuth    = yaw_deg + args.cam_azimuth

                # PD torque → physics step (all in MuJoCo order)
                tau = pd_control(target_dof_pos_mj, d.qpos[7: 7 + num_dof],
                                 kps, d.qvel[6: 6 + num_dof], kds)
                d.ctrl[:num_dof] = tau
                mujoco.mj_step(m, d)

                counter += 1
                if counter % decimation == 0:
                    # ----------------------------------------------------------
                    # 1. Build single-step obs
                    single_obs = build_single_obs(d)

                    # 2. Init history on first control step
                    if first_step:
                        history.reset(single_obs)
                        first_step = False
                    else:
                        history.insert(single_obs)

                    # 3. Actor inference on stacked history (210-dim)
                    policy_obs_t = torch.from_numpy(history.get()).unsqueeze(0)
                    with torch.inference_mode():
                        actor_out = actor_policy(policy_obs_t).squeeze(0).numpy()

                    residual    = actor_out[:num_dof]                          # (12,)
                    skill_logits = actor_out[num_dof:]                         # (3,)
                    skill_cmd   = F.softmax(
                        torch.from_numpy(skill_logits), dim=-1).numpy()        # (3,)

                    # 4. Build prior obs and run motion prior
                    prior_obs_np = build_prior_obs(single_obs, skill_cmd)
                    prior_obs_t  = torch.from_numpy(prior_obs_np).unsqueeze(0)
                    with torch.inference_mode():
                        prior_actions = motion_prior(prior_obs_t).squeeze(0).numpy()  # (12,)

                    # 5. Combine: prior + scaled residual → joint targets (Isaac order)
                    joint_actions     = prior_actions + res_scale * residual
                    target_isaac      = joint_actions * action_scale + default_angles
                    # Reorder Isaac → MuJoCo for PD control
                    target_dof_pos_mj = target_isaac[REORDER]

                    # Periodic log
                    if counter % (decimation * 100) == 0:
                        skill_name = ["pace", "trot", "canter"][np.argmax(skill_cmd)]
                        vx = d.qvel[0]
                        print(f"t={time.time()-start:.1f}s | skill={skill_name} "
                              f"({skill_cmd[0]:.2f}/{skill_cmd[1]:.2f}/{skill_cmd[2]:.2f}) | "
                              f"vx={vx:.2f} m/s | cmd=[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}]")

                    # Record one frame per policy step
                    if video_writer is not None:
                        rec_cam.lookat[:] = d.qpos[:3]
                        renderer.update_scene(d, camera=rec_cam)
                        video_writer.append_data(renderer.render())

                viewer.sync()

                # Real-time pacing
                elapsed = time.time() - step_start
                if sim_dt - elapsed > 0:
                    time.sleep(sim_dt - elapsed)

    except KeyboardInterrupt:
        print("\n[info] Interrupted.")
    finally:
        if joy_reader is not None:
            joy_reader.stop()
        if video_writer is not None:
            video_writer.close()
            renderer.close()
            print(f"[record] Saved → {args.record}")
