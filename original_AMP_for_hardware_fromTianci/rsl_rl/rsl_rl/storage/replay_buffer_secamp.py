import torch
import numpy as np


class ReplayBuffer_SECAMP:
    """Circular buffer of H-frame AMP observation sequences.

    Parameters
    ----------
    H : int
        Trajectory horizon (discriminator input length).
    amp_obs_dim : int
        Dimension of a single AMP frame.
    skill_cmd_dim: int
        Dimension of skill command
    buf_size : int
        Maximum number of H-sequences stored.
    device : str
    """

    def __init__(self,
                 H: int = 5,
                 amp_obs_dim: int = 43,
                 skill_cmd_dim: int = 5,
                 buf_size: int = 100_000,
                 device: str = "cpu"):

        self.H              = H
        self.obs_dim        = amp_obs_dim
        self.skill_cmd_dim  = skill_cmd_dim
        self.buf_size       = buf_size
        self.device         = device

        # Circular buffer: amp obs [buf_size, H, obs_dim], skill cmd [buf_size, skill_cmd_dim]
        self._buffer    = torch.zeros(buf_size, H, amp_obs_dim, device=device)
        self._skill_buf = torch.zeros(buf_size, skill_cmd_dim, device=device)
        self._ptr       = 0   # next write position
        self._filled    = 0   # number of valid entries

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, amp_obs_seq: torch.Tensor, skill_dim_seq: torch.Tensor) -> None:
        """Store one step of data from the environment.

        Parameters
        ----------
        amp_obs_seq   : Tensor[k, H, obs_dim]  — full horizon AMP buffer per env
        skill_dim_seq : Tensor[k, skill_cmd_dim] — current skill command per env (no H)
        """
        k = amp_obs_seq.shape[0]
        end = self._ptr + k
        if end <= self.buf_size:
            self._buffer[self._ptr:end] = amp_obs_seq
            self._skill_buf[self._ptr:end] = skill_dim_seq
        else:
            first = self.buf_size - self._ptr
            self._buffer[self._ptr:] = amp_obs_seq[:first]
            self._buffer[:k - first] = amp_obs_seq[first:]
            self._skill_buf[self._ptr:] = skill_dim_seq[:first]
            self._skill_buf[:k - first] = skill_dim_seq[first:]
        self._ptr    = (self._ptr + k) % self.buf_size
        self._filled = min(self._filled + k, self.buf_size)

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """Yield (amp_obs, skill_cmd) tuples.

        Returns
        -------
        amp_obs   : Tensor[batch, H, obs_dim]
        skill_cmd : Tensor[batch, skill_cmd_dim]
        """
        if self._filled == 0:
            raise RuntimeError("ReplayBuffer_SECAMP is empty — cannot sample.")
        for _ in range(num_mini_batch):
            idxs = np.random.choice(self._filled, mini_batch_size)
            yield self._buffer[idxs], self._skill_buf[idxs]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_sequences(self) -> int:
        return self._filled

    def __len__(self) -> int:
        return self._filled
