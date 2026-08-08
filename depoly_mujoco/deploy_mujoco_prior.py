"""
Sim2Sim deployment for the motion prior in standalone mode.

No residual policy. The prior is driven directly with a body-frame walking
direction and a manually chosen skill (pace / trot / canter).

Two control modes (--mode):
  waypoint  (default) — body-frame direction follows a hard-coded time schedule
                        (DIR_WAYPOINTS list). Optional joystick can override the
                        skill buttons, but direction comes from the schedule.
  joystick            — left stick on gamepad drives the body-frame direction in
                        real time; face buttons switch skill. Joystick is required
                        in this mode.

Inference pipeline (per control step):
  1. Build 42-dim single-step obs from MuJoCo state
       gravity (3) | cmd_dir*0.5 (3) | dof_pos (12) | dof_vel (12) | last_action (12)
  2. Append skill one-hot (3)  →  45-dim prior_obs
  3. motion_prior(prior_obs)   →  joint_actions (12)
  4. target_dof_pos = joint_actions × action_scale + default_angles

Joint reorder (self-inverse permutation [3,4,5,0,1,2,9,10,11,6,7,8]):
  - MuJoCo go2.xml: FR / FL / RR / RL × (hip / thigh / calf)
  - Isaac Gym URDF: FL / FR / RL / RR × (hip / thigh / calf)
  Reading: isaac_q  = mj_q[REORDER]
  Writing: mj_ctrl  = isaac_ctrl[REORDER]

Usage:
    python deploy_mujoco_prior.py go2_prior.yaml
    python deploy_mujoco_prior.py go2_prior.yaml --mode joystick
"""

import math
import time
import threading
import queue
import argparse

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from legged_gym import LEGGED_GYM_ROOT_DIR


# -----------------------------------------------------------------------
# Joystick reader for motion prior (background daemon thread)
#
# Left stick  → walking direction in body frame (updates cmd_dir_obs)
# Face buttons → skill switching (edge-triggered, updates skill_onehot)
# -----------------------------------------------------------------------

class JoystickPriorReader(threading.Thread):
    """Polls a gamepad at ~100 Hz.

    Shared mutable state (all numpy arrays, GIL-safe for in-place writes):
      cmd_dir_obs  (3,)  — obs[3:6]: [dx*0.5, dy*0.5, 0.0]
      skill_onehot (3,)  — one-hot skill command
    """

    SKILL_NAMES = ["pace", "trot", "canter"]

    def __init__(self, cfg_joy: dict, cmd_dir_obs: np.ndarray,
                 skill_onehot: np.ndarray, cmd_dir_scale: float):
        super().__init__(daemon=True, name="joystick")
        self._cfg          = cfg_joy
        self._cmd_dir_obs  = cmd_dir_obs    # shared, updated in-place
        self._skill_onehot = skill_onehot   # shared, updated in-place
        self._scale        = cmd_dir_scale
        self._stop         = threading.Event()

    def run(self):
        try:
            import pygame
        except ImportError:
            print("[joystick] pygame not installed — joystick disabled.")
            return

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("[joystick] No gamepad found — using yaml defaults.")
            return

        joy = pygame.joystick.Joystick(0)
        joy.init()
        print(f"[joystick] Connected: {joy.get_name()}")

        cfg  = self._cfg
        dz   = float(cfg.get("deadzone", 0.15))
        btns = [int(cfg["btn_pace"]), int(cfg["btn_trot"]), int(cfg["btn_canter"])]
        btn_lb = int(cfg.get("btn_lb", 4))
        btn_rb = int(cfg.get("btn_rb", 5))
        prev_btn = [False] * 3

        def read_axis(idx: int, sign: float) -> float:
            v = joy.get_axis(int(idx)) * sign
            return v if abs(v) > dz else 0.0

        while not self._stop.is_set():
            pygame.event.pump()

            # --- Dead-man's switch: LB + RB must both be held to enable motion ---
            enabled = bool(joy.get_button(btn_lb)) and bool(joy.get_button(btn_rb))

            if not enabled:
                self._cmd_dir_obs[0] = 0.0
                self._cmd_dir_obs[1] = 0.0
            else:
                # --- Direction: left stick → body-frame XY unit vector × scale ---
                # Pulling stick backward (dx < 0) → stand still (zero direction).
                dx = read_axis(cfg["axis_dir_x"], cfg.get("sign_dir_x", 1))
                dy = read_axis(cfg["axis_dir_y"], cfg.get("sign_dir_y", 1))
                norm = math.sqrt(dx * dx + dy * dy)
                if dx < 0:
                    self._cmd_dir_obs[0] = 0.0
                    self._cmd_dir_obs[1] = 0.0
                elif norm > 1e-6:
                    self._cmd_dir_obs[0] = dx / norm * self._scale
                    self._cmd_dir_obs[1] = dy / norm * self._scale
                else:
                    # Stick in deadzone: stand still
                    self._cmd_dir_obs[0] = 0.0
                    self._cmd_dir_obs[1] = 0.0
            # self._cmd_dir_obs[2] always stays 0.0

            # --- Skill buttons: edge-triggered (press, not hold) -------------
            for i, btn_id in enumerate(btns):
                pressed = bool(joy.get_button(btn_id))
                if pressed and not prev_btn[i]:
                    self._skill_onehot[:] = 0.0
                    self._skill_onehot[i] = 1.0
                    print(f"[joystick] Skill → {self.SKILL_NAMES[i]}")
                prev_btn[i] = pressed

            time.sleep(0.01)   # 100 Hz

    def stop(self):
        self._stop.set()


# -----------------------------------------------------------------------
# Joint reorder: MuJoCo XML (FR/FL/RR/RL) ↔ Isaac Gym URDF (FL/FR/RL/RR)
# Self-inverse permutation: REORDER[REORDER] == arange(12)
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


def make_skill_onehot(skill_idx: int, dim: int = 3) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[skill_idx] = 1.0
    return v


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="YAML config filename in depoly_mujoco/")
    parser.add_argument("--mode", choices=["waypoint", "joystick"], default="waypoint",
                        help="waypoint: follow DIR_WAYPOINTS schedule; "
                             "joystick: left stick drives direction in real time")
    parser.add_argument("--record", metavar="FILE.mp4",
                        help="Record simulation to video (requires imageio[ffmpeg])")
    parser.add_argument("--record-res", nargs=2, type=int, default=[1280, 720],
                        metavar=("W", "H"), help="Video resolution (default: 1280 720)")
    parser.add_argument("--direction-deg", type=float, default=None,
                        help="Override initial direction in waypoint mode (e.g. 45=left, -45=right)")
    args = parser.parse_args()

    config_dir = f"{LEGGED_GYM_ROOT_DIR}/depoly_mujoco"
    with open(f"{config_dir}/{args.config_file}", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    def resolve(p): return p.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    prior_path     = resolve(cfg["motion_prior_path"])
    xml_path       = resolve(cfg["xml_path"])

    sim_duration   = cfg["simulation_duration"]
    sim_dt         = cfg["simulation_dt"]
    decimation     = cfg["control_decimation"]

    kps            = np.array(cfg["kps"],            dtype=np.float32)
    kds            = np.array(cfg["kds"],            dtype=np.float32)
    default_angles = np.array(cfg["default_angles"], dtype=np.float32)  # Isaac order

    dof_pos_scale  = cfg["dof_pos_scale"]
    dof_vel_scale  = cfg["dof_vel_scale"]
    cmd_dir_scale  = cfg["cmd_dir_scale"]   # 0.5
    action_scale   = cfg["action_scale"]

    num_obs        = cfg["num_obs"]   # 45
    num_dof        = cfg["num_dof"]   # 12

    # Waypoint schedule: (start_time_sec, direction_deg, skill_idx)
    # direction_deg: angle from body +X axis  (0=forward, 90=left, -90=right)
    # skill_idx:     0=pace, 1=trot, 2=canter, -1=stop (zeros direction, freezes skill)
    # Max consecutive direction change kept ≤ 20° for smooth transitions.
    SKILL_NAMES = ["pace", "trot", "canter"]
    DIR_WAYPOINTS = [
        ( 0.0,   0.0,  1),   # trot  — warmup, forward
        ( 4.0, -10.0,  1),   # trot  — gentle right curve
        ( 7.0, -20.0,  0),   # pace  — slow down, continue right
        (10.0, -10.0,  0),   # pace  — ease back
        (13.0,   0.0,  0),   # pace  — forward
        (17.0,   0.0,  2),   # canter — accelerate forward
        (20.0,  15.0,  2),   # canter — gentle left
        (23.0,  30.0,  2),   # canter — left arc
        (26.0,  20.0,  2),   # canter — straightening
        (29.0,   0.0,  1),   # trot  — forward
        (32.0, -15.0,  1),   # trot  — gentle right
        (35.0,   0.0,  1),   # trot  — forward
        (38.0,  15.0,  0),   # pace  — gentle left, slow down
        (41.0,   0.0,  0),   # pace  — forward, decelerate
        (44.0,   0.0, -1),   # stop  — stand in place
    ]
    # In waypoint mode the simulation ends automatically 3 s after the stop entry.
    _stop_time    = next(t for t, _, sk in DIR_WAYPOINTS if sk == -1)
    _waypoint_dur = _stop_time + 3.0

    def _dir_obs(deg: float) -> np.ndarray:
        r = math.radians(deg)
        return np.array([math.cos(r) * cmd_dir_scale,
                         math.sin(r) * cmd_dir_scale,
                         0.0], dtype=np.float32)

    cmd_dir_obs  = _dir_obs(DIR_WAYPOINTS[0][1])
    skill_onehot = make_skill_onehot(DIR_WAYPOINTS[0][2])  # init from first waypoint

    # --- Precompute MuJoCo-order default angles -------------------------
    default_angles_mj = default_angles[REORDER]

    # --- Load models ----------------------------------------------------
    print(f"Loading motion prior: {prior_path}")
    motion_prior = torch.jit.load(prior_path, map_location="cpu").eval()

    # --- Load MuJoCo model ----------------------------------------------
    print(f"Loading MuJoCo model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    # --- Video recorder (optional) --------------------------------------
    video_writer  = None
    renderer      = None
    frames_dir    = None
    frame_idx     = 0
    frame_queue   = None   # background PNG writer queue
    frame_thread  = None
    if args.record:
        try:
            import imageio
            import os
        except ImportError:
            print("[record] imageio not installed. Run:  pip install imageio[ffmpeg]")
            args.record = None
        else:
            rec_fps      = int(round(1.0 / (sim_dt * decimation)))   # 50 Hz
            rec_W, rec_H = args.record_res
            rec_abs      = os.path.abspath(args.record)
            rec_folder   = os.path.dirname(rec_abs)
            os.makedirs(rec_folder, exist_ok=True)
            frames_dir   = os.path.join(rec_folder, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            m.vis.global_.offwidth  = rec_W
            m.vis.global_.offheight = rec_H
            renderer     = mujoco.Renderer(m, height=rec_H, width=rec_W)
            video_writer = imageio.get_writer(rec_abs, fps=rec_fps, quality=8)

            # Background thread: drain queue and write PNGs without blocking sim
            frame_queue = queue.Queue()
            def _frame_writer(q, fdir):
                while True:
                    item = q.get()
                    if item is None:
                        break
                    idx, buf = item
                    imageio.imwrite(f"{fdir}/frame_{idx:06d}.png", buf)
                    q.task_done()
            frame_thread = threading.Thread(
                target=_frame_writer, args=(frame_queue, frames_dir), daemon=True)
            frame_thread.start()

            print(f"[record] {rec_W}×{rec_H} @ {rec_fps} fps → {rec_abs}")
            print(f"[record] frames → {frames_dir}/")

    # --- Joystick -------------------------------------------------------
    # In joystick mode: always required; thread owns cmd_dir_obs updates.
    # In waypoint mode: optional (skill buttons only; direction from schedule).
    joy_reader = None
    joy_cfg    = cfg.get("joystick", {})
    if args.mode == "joystick":
        joy_reader = JoystickPriorReader(joy_cfg, cmd_dir_obs,
                                         skill_onehot, cmd_dir_scale)
        joy_reader.start()
        init_skill = SKILL_NAMES[int(np.argmax(skill_onehot))]
        print(f"[mode] joystick  |  left-stick → direction"
              f"  |  skill: {init_skill} (face buttons to switch)")
    else:
        print(f"[mode] waypoint  |  schedule: {DIR_WAYPOINTS}")

    # --- Runtime buffers ------------------------------------------------
    last_action       = np.zeros(num_dof, dtype=np.float32)  # Isaac order
    target_dof_pos_mj = default_angles_mj.copy()

    counter = 0

    # -----------------------------------------------------------------------
    def build_prior_obs(d) -> np.ndarray:
        """Build 45-dim prior obs.

        Layout (matches posTracking training):
          [0:3]   projected_gravity
          [3:6]   [cos(dir)*0.5,  sin(dir)*0.5,  0.0]
          [6:18]  (dof_pos - default) * dof_pos_scale    (Isaac order)
          [18:30] dof_vel * dof_vel_scale                (Isaac order)
          [30:42] last joint actions                     (Isaac order)
          [42:45] skill one-hot
        """
        quat = d.qpos[3:7]                          # [w, x, y, z]
        qj   = d.qpos[7 : 7 + num_dof][REORDER]    # MuJoCo → Isaac order
        dqj  = d.qvel[6 : 6 + num_dof][REORDER]

        gravity = get_gravity_orientation(quat)
        dof_pos = (qj  - default_angles) * dof_pos_scale
        dof_vel = dqj  * dof_vel_scale

        obs_42 = np.concatenate([gravity, cmd_dir_obs, dof_pos, dof_vel, last_action])
        assert obs_42.shape[0] == 42
        return np.concatenate([obs_42, skill_onehot]).astype(np.float32)


    # --- body id for camera tracking ----------------------------------------
    base_link_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    # --- shared camera params (viewer + recorder use the same values) -------
    CAM_AZIMUTH   = 135.0   # degrees: rear-right follow view
    CAM_ELEVATION = -30.0   # degrees: looking down
    CAM_DISTANCE  = 3.5     # meters

    # Free camera for off-screen renderer (recording / frames)
    rec_cam            = mujoco.MjvCamera()
    rec_cam.type       = mujoco.mjtCamera.mjCAMERA_FREE
    rec_cam.azimuth    = CAM_AZIMUTH
    rec_cam.elevation  = CAM_ELEVATION
    rec_cam.distance   = CAM_DISTANCE

    # -----------------------------------------------------------------------
    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            # Rear-right follow camera: free camera that tracks robot position
            viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance  = CAM_DISTANCE
            viewer.cam.azimuth   = CAM_AZIMUTH
            viewer.cam.elevation = CAM_ELEVATION

            start = time.time()

            duration = _waypoint_dur if args.mode == "waypoint" else sim_duration
            while viewer.is_running() and time.time() - start < duration:
                step_start = time.time()

                # PD torque → physics step (MuJoCo order throughout)
                tau = pd_control(target_dof_pos_mj, d.qpos[7: 7 + num_dof],
                                 kps, d.qvel[6: 6 + num_dof], kds)
                d.ctrl[:num_dof] = tau
                mujoco.mj_step(m, d)

                # Keep camera lookat on robot body
                viewer.cam.lookat[:] = d.xpos[base_link_id]

                # Update direction and skill from waypoints (waypoint mode only).
                # In joystick mode the JoystickPriorReader thread owns both.
                if args.mode == "waypoint":
                    elapsed = time.time() - start
                    for t, deg, sk in reversed(DIR_WAYPOINTS):
                        if elapsed >= t:
                            if sk == -1:
                                cmd_dir_obs[:] = 0.0   # zero direction → stand still
                            else:
                                cmd_dir_obs[:] = _dir_obs(deg)
                                skill_onehot[:] = 0.0
                                skill_onehot[sk] = 1.0
                            break

                counter += 1
                if counter % decimation == 0:
                    # Build obs and run prior
                    prior_obs_np = build_prior_obs(d)
                    prior_obs_t  = torch.from_numpy(prior_obs_np).unsqueeze(0)

                    with torch.inference_mode():
                        joint_actions_t = motion_prior(prior_obs_t)
                    joint_actions = joint_actions_t.squeeze(0).numpy()   # Isaac order

                    # Store as last_action for next obs
                    last_action[:] = joint_actions

                    # Compute target in Isaac order, then reorder to MuJoCo
                    target_isaac      = joint_actions * action_scale + default_angles
                    target_dof_pos_mj = target_isaac[REORDER]

                    if counter % (decimation * 200) == 0:
                        current_skill = SKILL_NAMES[int(np.argmax(skill_onehot))]
                        vx = d.qvel[0]
                        print(f"t={time.time()-start:.1f}s | skill={current_skill} | "
                              f"dir=[{cmd_dir_obs[0]:.2f},{cmd_dir_obs[1]:.2f}] | "
                              f"vx_world={vx:.2f} m/s")

                    # Record one frame per policy step
                    if video_writer is not None:
                        rec_cam.lookat[:] = d.xpos[base_link_id]
                        renderer.update_scene(d, camera=rec_cam)
                        frame = renderer.render()
                        video_writer.append_data(frame)
                        frame_queue.put((frame_idx, frame.copy()))
                        frame_idx += 1

                viewer.sync()

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
            if frame_queue is not None:
                frame_queue.join()          # wait for all PNGs to finish
                frame_queue.put(None)       # signal thread to exit
                frame_thread.join()
            print(f"[record] Saved → {args.record}")
