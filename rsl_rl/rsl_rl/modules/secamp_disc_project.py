import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd

from rsl_rl.utils import utils


class SECAMPDiscriminator_Project(nn.Module):
    """Conditional AMP discriminator (LSGAN): skill_cmd is projected to feature space.

    score = <trunk(x), embed(z)> + linear(trunk(x))
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
        super(SECAMPDiscriminator_Project, self).__init__()

        self.device          = device
        self.amp_obs_dim     = amp_obs_dim
        self.amp_obs_horizon = amp_obs_horizon
        self.input_dim       = amp_obs_dim * amp_obs_horizon
        self.skill_cmd_dim   = skill_cmd_dim if skill_cmd_dim is not None else 0
        self.skill_insert    = 'project'

        self.amp_reward_coef  = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

        amp_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk      = nn.Sequential(*amp_layers).to(device)

        # Projection embedding V: z → feature space
        self.embed = nn.Linear(skill_cmd_dim, hidden_layer_sizes[-1], bias=False).to(device)

        # Unconditional head: feature → scalar
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()
        self.embed.train()

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: [N, obs_dim * obs_horizon]
        z: [N, skill_cmd_dim]
        Returns: [N, 1]
        """
        feature       = self.trunk(x)
        embed         = self.embed(z)
        inner         = (feature * embed).sum(dim=-1, keepdim=True)
        unconditional = self.amp_linear(feature)
        return inner + unconditional

    def compute_grad_pen(self,
                         obs_buf_ref: torch.Tensor,
                         skill_cmd_ref: torch.Tensor,
                         lambda_=10):
        """Zero-centred GP: ∇_x D(x, z) w.r.t. obs only."""
        x = obs_buf_ref.flatten(start_dim=1) if obs_buf_ref.dim() > 2 else obs_buf_ref
        x = x.detach().requires_grad_(True)
        z = skill_cmd_ref.detach()
        disc = self.forward(x, z)
        grad = autograd.grad(
            outputs=disc, inputs=x,
            grad_outputs=torch.ones_like(disc),
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

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
            combined_reward = self._lerp_reward(amp_reward, task_reward.unsqueeze(-1))
            self.train()
        return combined_reward.squeeze(), d, amp_reward.squeeze(), task_reward.squeeze()

    def _lerp_reward(self, disc_r, task_r):
        return (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
