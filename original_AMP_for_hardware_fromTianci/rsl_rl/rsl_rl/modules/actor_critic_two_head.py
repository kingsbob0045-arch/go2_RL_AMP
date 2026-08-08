import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation


class ActorCriticTwoHead(nn.Module):
    """Actor with a shared backbone and two output heads.

    Head 1 — residual actions : num_residual_actions dims (default 12)
    Head 2 — skill command    : num_skill_dims dims        (default  3)

    The two head outputs are concatenated to form the full 15-dim action
    vector that the env's step() receives.  PPO sees a single
    Normal(mean, std) over all 15 dims and is otherwise unchanged.

    Args:
        num_actor_obs       : flattened history obs dim  (e.g. 5 × 42 = 210)
        num_critic_obs      : privileged obs dim         (e.g. 48)
        num_actions         : total actor output dim     (residual + skill = 15)
        num_residual_actions: dims for the residual head (default 12)
        num_skill_dims      : dims for the skill head    (default  3)
        backbone_hidden_dims: hidden layers shared by both heads
        critic_hidden_dims  : hidden layers for the critic
        activation          : activation name
        init_noise_std      : initial exploration std
    """

    is_recurrent = False

    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 num_residual_actions=12,
                 num_skill_dims=3,
                 backbone_hidden_dims=[512, 256],
                 critic_hidden_dims=[512, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        min_std_r        = kwargs.pop('min_std_residual', None)
        min_std_s        = kwargs.pop('min_std_skill', None)
        max_std_r        = kwargs.pop('max_std_residual', None)
        max_std_s        = kwargs.pop('max_std_skill', None)
        fixed_skill_std  = kwargs.pop('fixed_skill_std', False)
        if kwargs:
            print("ActorCriticTwoHead got unexpected kwargs (ignored): "
                  + str(list(kwargs.keys())))
        super().__init__()

        assert num_residual_actions + num_skill_dims == num_actions, (
            f"num_residual_actions({num_residual_actions}) + "
            f"num_skill_dims({num_skill_dims}) must equal "
            f"num_actions({num_actions})")

        act = get_activation(activation)

        # ── Shared backbone ──────────────────────────────────────────────
        backbone_layers = []
        in_dim = num_actor_obs
        for out_dim in backbone_hidden_dims:
            backbone_layers += [nn.Linear(in_dim, out_dim), act]
            in_dim = out_dim
        self.backbone = nn.Sequential(*backbone_layers)

        # ── Two output heads ─────────────────────────────────────────────
        self.residual_head = nn.Linear(in_dim, num_residual_actions)
        self.skill_head    = nn.Linear(in_dim, num_skill_dims)

        # ── Critic ───────────────────────────────────────────────────────
        critic_layers = []
        c_in = num_critic_obs
        for c_out in critic_hidden_dims:
            critic_layers += [nn.Linear(c_in, c_out), act]
            c_in = c_out
        critic_layers.append(nn.Linear(c_in, 1))
        self.critic = nn.Sequential(*critic_layers)

        print(f"ActorCriticTwoHead backbone : {self.backbone}")
        print(f"  residual_head             : {self.residual_head}")
        print(f"  skill_head                : {self.skill_head}")
        print(f"  critic                    : {self.critic}")

        # ── Exploration noise (separate per head, then concatenated) ─────
        std_residual = init_noise_std * torch.ones(num_residual_actions)
        self.std_residual = nn.Parameter(std_residual)

        # skill std: fixed (non-learnable) or learnable
        std_skill = init_noise_std * torch.ones(num_skill_dims)
        if fixed_skill_std:
            # register as buffer so it moves with .to(device) but gets no gradient
            self.register_buffer('std_skill', std_skill)
        else:
            self.std_skill = nn.Parameter(std_skill)
        self.fixed_skill_std = fixed_skill_std

        # ── Per-dim std bounds — registered as buffers so they follow .to(device)
        # register_buffer(name, None) is valid and returns None on access
        def _to_buf(name, val):
            t = torch.tensor(val, dtype=torch.float) if val is not None else None
            self.register_buffer(name, t)

        _to_buf('_min_std_r', min_std_r)
        _to_buf('_max_std_r', max_std_r)
        _to_buf('_min_std_s', min_std_s)
        _to_buf('_max_std_s', max_std_s)

        self.distribution = None
        Normal.set_default_validate_args = False

    # ------------------------------------------------------------------

    def reset(self, _dones=None):
        pass

    def clamp_std(self):
        """Apply per-dim min/max bounds to std parameters after each PPO update.

        skill std is skipped when fixed_skill_std=True (it has no gradient anyway).
        """
        self.std_residual.data.clamp_(
            min=self._min_std_r if self._min_std_r is not None else float('-inf'),
            max=self._max_std_r if self._max_std_r is not None else float('inf'),
        )
        if not self.fixed_skill_std:
            self.std_skill.data.clamp_(
                min=self._min_std_s if self._min_std_s is not None else float('-inf'),
                max=self._max_std_s if self._max_std_s is not None else float('inf'),
            )

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _actor_mean(self, observations):
        feat = self.backbone(observations)
        residual = self.residual_head(feat)
        skill    = self.skill_head(feat)
        return torch.cat([residual, skill], dim=-1)   # [N, 15]

    def update_distribution(self, observations):
        mean = self._actor_mean(observations)
        std  = torch.cat([self.std_residual, self.std_skill]).to(mean.device)
        self.distribution = Normal(mean, mean * 0. + std)

    def act(self, observations, **_kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self._actor_mean(observations)

    def evaluate(self, critic_observations, **_kwargs):
        return self.critic(critic_observations)
