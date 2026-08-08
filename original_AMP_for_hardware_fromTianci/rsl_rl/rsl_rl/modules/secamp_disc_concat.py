import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd

from rsl_rl.utils import utils


class SECAMPDiscriminator_Concat(nn.Module):
    """Conditional AMP discriminator (LSGAN): skill_cmd is concatenated to flattened AMP obs.

    Input to trunk: [amp_obs_flat, skill_cmd]
                     dim = amp_obs_dim * H + skill_cmd_dim
    """

    def __init__(
            self,
            amp_obs_dim,
            amp_obs_horizon,
            amp_reward_coef,
            hidden_layer_sizes,
            device,
            task_reward_lerp=0.0,
            skill_cmd_dim=0,
            ):
        super(SECAMPDiscriminator_Concat, self).__init__()

        self.device          = device
        self.amp_obs_dim     = amp_obs_dim
        self.amp_obs_horizon = amp_obs_horizon
        self.input_dim       = amp_obs_dim * amp_obs_horizon
        self.skill_cmd_dim   = skill_cmd_dim if skill_cmd_dim is not None else 0
        self.skill_insert    = 'concat'

        self.amp_reward_coef  = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

        amp_layers = []
        curr_in_dim = self.input_dim + self.skill_cmd_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk      = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, x, c):
        """
        x : [batch, H*obs_dim] or [batch, H, obs_dim]
        c : [batch, skill_cmd_dim]
        """
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        inp = torch.cat([x, c], dim=-1)
        return self.amp_linear(self.trunk(inp))

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def predict_amp_reward(
            self,
            obs_buf: torch.Tensor,
            skill_cmd_buf: torch.Tensor,
            task_reward,
            dt,
            normalizer=None,
        ):
        with torch.no_grad():
            self.eval()
            inp = obs_buf.flatten(start_dim=1).clone() if obs_buf.dim() > 2 else obs_buf.clone()
            if normalizer is not None:
                inp = normalizer.normalize_torch(inp, self.device)

            d = self.forward(inp, skill_cmd_buf)
            amp_reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0) * dt

            if self.task_reward_lerp > 0:
                combined_reward = self._lerp_reward(amp_reward, task_reward.unsqueeze(-1))
            else:
                combined_reward = amp_reward
            self.train()
        return combined_reward.squeeze(), d, amp_reward.squeeze(), task_reward.squeeze()

    def _lerp_reward(self, disc_r, task_r):
        return (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r

    # ------------------------------------------------------------------
    # LSGAN — zero-centred GP on expert data
    # ------------------------------------------------------------------

    def _lsgan_grad_pen(self,
                        traj_ref: torch.Tensor,
                        skill_cmd_ref: torch.Tensor,
                        lambda_: float = 10.0) -> torch.Tensor:
        """GP = λ · E[‖∇_{[τ,c]} D([τ_ref, c])‖²]"""
        if traj_ref.dim() > 2:
            traj_ref = traj_ref.flatten(start_dim=1)
        inp = torch.cat([traj_ref, skill_cmd_ref], dim=-1).detach().requires_grad_(True)
        d = self.amp_linear(self.trunk(inp))
        grad = autograd.grad(
            outputs=d, inputs=inp,
            grad_outputs=torch.ones_like(d),
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

    def _lsgan_loss(self,
                    traj_ref: torch.Tensor,
                    traj_pol: torch.Tensor,
                    skill_cmd_ref: torch.Tensor,
                    skill_cmd_pol: torch.Tensor,
                    lambda_gp: float) -> torch.Tensor:
        """L_D = 0.5·(E[(D(ref,c)-1)²] + E[(D(pol,c)+1)²]) + λ·GP"""
        d_real = self.forward(traj_ref, skill_cmd_ref)
        d_fake = self.forward(traj_pol, skill_cmd_pol)
        loss = (torch.mean((d_real - 1.0) ** 2) + torch.mean((d_fake + 1.0) ** 2)) * 0.5
        return loss + self._lsgan_grad_pen(traj_ref, skill_cmd_ref, lambda_=lambda_gp)

    # ------------------------------------------------------------------
    # Legacy zero-centred GP (called directly by algorithm update loop)
    # ------------------------------------------------------------------

    def compute_grad_pen(self,
                         obs_buf_ref: torch.Tensor,
                         skill_cmd_ref: torch.Tensor,
                         lambda_=10):
        """Zero-centred GP on concatenated [obs, skill_cmd] for LSGAN."""
        if obs_buf_ref.dim() > 2:
            obs_buf_ref = obs_buf_ref.flatten(start_dim=1)
        inp = torch.cat([obs_buf_ref, skill_cmd_ref], dim=-1).requires_grad_(True)
        disc = self.amp_linear(self.trunk(inp))
        grad = autograd.grad(
            outputs=disc, inputs=inp,
            grad_outputs=torch.ones_like(disc),
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        return lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
