import glob
import json
import re
import torch
import numpy as np

from rsl_rl.datasets import pose3d
from rsl_rl.datasets import motion_util
from rsl_rl.datasets.motion_loader import AMPLoader

# Default skill maps for different skill_cmd_dim values.
_SKILL_MAP_5 = {
    "pace":   4,
    "trot":   3,
    "gallop": 2,
    "canter": 1,
    "jump":   0,
}

_SKILL_MAP_3 = {
    "pace":   0,
    "trot":   1,
    "canter": 2,
}

# Legacy module-level constants (kept for backward compat with external code).
_SKILL_NAME_TO_ID = _SKILL_MAP_5
_SKILL_ID_TO_ONEHOT = {
    sid: torch.tensor([1.0 if j == sid else 0.0 for j in range(5)],
    dtype=torch.float32)
    for sid in _SKILL_MAP_5.values()}


class SECAMPLoader(AMPLoader):
    """AMP motion loader with per-sample skill-conditioned expert sampling.

    Motion files are labeled by filename convention:
        <skill_name><index>.json   e.g.  trot0.json, pace1.json
    Files whose base name does not match any known skill are treated as
    "turn" (neutral) motions that can be mixed into any skill batch.

    Key method
    ----------
    sample(skill_cmd)
        Given a one-hot skill tensor [B, C], returns a matched expert AMP
        observation buffer [B, H, amp_obs_dim] where each row is sampled from
        the pool of trajectories labelled with the requested skill (plus an
        optional turn-motion mix-in).

    feed_forward_generator(num_mini_batch, mini_batch_size)
        Backward-compatible unconditional generator (uniform sampling).
        Use `sample()` for skill-conditioned expert draws.
    """

    def __init__(
            self,
            device,
            time_between_frames,
            data_dir='',
            preload_transitions=False,
            num_preload_transitions=1_000_000,
            motion_files=glob.glob('datasets/motion_files2/*'),
            reorder=False,
            num_amp_obs=43,
            amp_horizon=5,
            skill_cmd_dim=5,
            skill_map=None,
            turn_mix_ratio=0.1,
            jump_turn_mix_ratio=0.0,
            ):
        self.device = device
        self.time_between_frames = time_between_frames
        self.amp_num_obs = num_amp_obs
        self.amp_horizon = amp_horizon
        self.reorder = reorder
        self.preload_transitions = preload_transitions
        self.skill_cmd_dim = skill_cmd_dim

        # Resolve skill name → id mapping
        if skill_map is not None:
            self._skill_name_to_id = skill_map
        elif skill_cmd_dim == 3:
            self._skill_name_to_id = _SKILL_MAP_3
        else:
            self._skill_name_to_id = _SKILL_MAP_5
        self._skill_ids = set(self._skill_name_to_id.values())

        self.turn_mix_ratio = turn_mix_ratio
        # Per-skill override: jump gets its own ratio (default 0.0 = no mixing)
        self._per_skill_mix_ratio = {sid: turn_mix_ratio for sid in self._skill_ids}
        if 'jump' in self._skill_name_to_id:
            _jump_sid = self._skill_name_to_id['jump']
            self._per_skill_mix_ratio[_jump_sid] = jump_turn_mix_ratio

        self.load_motions(motion_files)
        self.preload(num_preload_transitions)

    # ------------------------------------------------------------------
    # Motion loading
    # ------------------------------------------------------------------

    def load_motions(self, motion_files):
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        # skill_id (int) → list of traj indices in self.trajectories_*
        self.skill_id_to_traj_indices = {sid: [] for sid in self._skill_ids}
        self.turn_idxs = []           # traj indices for unlabelled / turn motions
        self.traj_idx_to_skill_id = []  # length == num_trajs; -1 = turn

        for i, motion_file in enumerate(motion_files):
            # Parse skill tag from filenames such as trot0.json or trot_source.json.
            base = motion_file.split('/')[-1].split('.')[0]  # e.g. "trot0"
            skill_tag = next((
                name for name in sorted(self._skill_name_to_id, key=len, reverse=True)
                if re.match(rf"^{re.escape(name)}(?:$|\d|[_-])", base)
            ), "")

            self.trajectory_names.append(base)

            if skill_tag in self._skill_name_to_id:
                sid = self._skill_name_to_id[skill_tag]
                self.skill_id_to_traj_indices[sid].append(i)
                self.traj_idx_to_skill_id.append(sid)
            else:
                self.turn_idxs.append(i)
                self.traj_idx_to_skill_id.append(-1)

            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                if self.reorder:
                    motion_data = self.reorder_from_pybullet_to_isaac(motion_data)

                for f_i in range(motion_data.shape[0]):
                    root_rot = AMPLoader.get_root_rot(motion_data[f_i])
                    root_rot = pose3d.QuaternionNormalize(root_rot)
                    root_rot = motion_util.standardize_quaternion(root_rot)
                    motion_data[f_i,
                                AMPLoader.POS_SIZE:(AMPLoader.POS_SIZE + AMPLoader.ROT_SIZE)
                                ] = root_rot

                self.trajectories.append(torch.tensor(
                    motion_data[:, AMPLoader.ROOT_ROT_END_IDX:AMPLoader.JOINT_VEL_END_IDX],
                    dtype=torch.float32, device=self.device))
                self.trajectories_full.append(torch.tensor(
                    motion_data[:, :AMPLoader.JOINT_VEL_END_IDX],
                    dtype=torch.float32, device=self.device))

                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len:.2f}s motion from {motion_file}  (skill: {skill_tag})")

        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)
        self.traj_idx_to_skill_id = np.array(self.traj_idx_to_skill_id, dtype=np.int32)

    # ------------------------------------------------------------------
    # Preloading
    # ------------------------------------------------------------------

    def preload(self, num_preload):
        if self.preload_transitions:
            print(f'Preloading {num_preload} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload)

            # Horizon-aware time sampling
            subst = ((self.amp_horizon - 1) * self.time_between_frames
                     + self.trajectory_frame_durations[traj_idxs])
            time_samples = (self.trajectory_lens[traj_idxs]
                            * np.random.uniform(size=num_preload) - subst)
            times = np.maximum(np.zeros_like(time_samples), time_samples)

            full_frame_dim = self.trajectories_full[0].shape[-1]
            self.preload_buffer = torch.zeros(
                num_preload, self.amp_horizon, full_frame_dim, device=self.device)
            for step in range(self.amp_horizon):
                self.preload_buffer[:, step] = self.get_full_frame_at_time_batch(
                    traj_idxs, times + step * self.time_between_frames)

            # Record which skill each preloaded sample belongs to
            self.preload_skill_ids = self.traj_idx_to_skill_id[traj_idxs]  # [N], int32

            # Build fast lookup: skill_id → numpy array of indices into preload_buffer
            self.preload_idx_by_skill: dict[int, np.ndarray] = {}
            for sid in self.skill_id_to_traj_indices:
                mask = self.preload_skill_ids == sid
                self.preload_idx_by_skill[sid] = np.where(mask)[0]
            self.preload_turn_idx = np.where(self.preload_skill_ids == -1)[0]

            print(f'Preload buffer shape: {self.preload_buffer.shape}')
            id_to_name = {v: k for k, v in self._skill_name_to_id.items()}
            for sid, idxs in self.preload_idx_by_skill.items():
                skill_name = id_to_name.get(sid, f'id{sid}')
                print(f'  skill {sid:2d} ({skill_name:>6}): {len(idxs):>8} samples')
            print(f'  turn motions         : {len(self.preload_turn_idx):>8} samples')

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    # ------------------------------------------------------------------
    # Skill-conditioned sampling
    # ------------------------------------------------------------------

    def sample(self, skill_cmd: torch.Tensor) -> torch.Tensor:
        """Sample expert AMP obs matched to the given skill commands.

        Args:
            skill_cmd: one-hot tensor [B, skill_cmd_dim], any device.

        Returns:
            amp_obs: [B, H, amp_obs_dim] on self.device.
        """
        if not self.preload_transitions:
            raise NotImplementedError("SECAMPLoader.sample() requires preload_transitions=True")

        skill_ids = torch.argmax(skill_cmd.cpu(), dim=-1).numpy()  # [B]
        B = len(skill_ids)
        chosen = np.empty(B, dtype=np.int64)

        has_turn_pool = len(self.preload_turn_idx) > 0

        # Decide per-sample whether to draw from turn pool (per-skill ratio)
        if has_turn_pool:
            rand_vals = np.random.random(B)
            is_turn = np.array([
                rand_vals[i] < self._per_skill_mix_ratio.get(int(skill_ids[i]), self.turn_mix_ratio)
                for i in range(B)
            ], dtype=bool)
        else:
            is_turn = np.zeros(B, dtype=bool)

        # Turn samples — uniformly from the turn pool
        n_turn = int(is_turn.sum())
        if n_turn > 0:
            chosen[is_turn] = np.random.choice(
                self.preload_turn_idx, size=n_turn, replace=True)

        # Skill-matched samples — grouped by skill_id for efficiency
        skill_mask = ~is_turn
        if skill_mask.any():
            skill_ids_masked = skill_ids[skill_mask]
            chosen_masked = np.empty(int(skill_mask.sum()), dtype=np.int64)
            for sid in np.unique(skill_ids_masked):
                group_mask = skill_ids_masked == sid
                n_group = int(group_mask.sum())
                pool = self.preload_idx_by_skill.get(int(sid), np.array([], dtype=np.int64))
                if len(pool) == 0:
                    # No labelled data for this skill — fall back to uniform
                    chosen_masked[group_mask] = np.random.randint(
                        0, self.preload_buffer.shape[0], size=n_group)
                else:
                    chosen_masked[group_mask] = np.random.choice(
                        pool, size=n_group, replace=True)
            chosen[skill_mask] = chosen_masked

        return self._extract_amp_obs(self.preload_buffer[chosen])

    def _extract_amp_obs(self, buffer):
        """Extract amp obs from full-frame buffer [B, H, full_dim].

        Matches env get_amp_observations():
          joint_pos(12)+foot_pos(12)+lin_vel(3)+ang_vel(3)+joint_vel(12)+z_pos(1) = 43
        root_pos[2] (index 2) provides z_pos; remaining 42 dims at [7:49].
        """
        obs   = buffer[:, :, AMPLoader.ROOT_ROT_END_IDX:AMPLoader.JOINT_VEL_END_IDX]  # [B,H,42]
        z_pos = buffer[:, :, 2:3]                                                       # [B,H,1]
        return torch.cat([obs, z_pos], dim=-1)                                          # [B,H,43]

    # ------------------------------------------------------------------
    # Backward-compatible unconditional generator
    # ------------------------------------------------------------------

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Unconditional uniform sampling — same interface as SECAMPLoader base.
        Use sample(skill_cmd) for skill-conditioned expert draws.
        """
        if not self.preload_transitions:
            raise NotImplementedError("Must use preload_transitions=True")
        for _ in range(num_mini_batch):
            idxs = np.random.choice(self.preload_buffer.shape[0], size=mini_batch_size)
            yield self._extract_amp_obs(self.preload_buffer[idxs])
