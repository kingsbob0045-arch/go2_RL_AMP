import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer_secamp import ReplayBuffer_SECAMP


class SECAMPPPO:
    """PPO + Conditional AMP for go2_secamp waypoint-tracking locomotion (LSGAN)."""

    actor_critic: ActorCritic

    def __init__(self,
                 actor_critic,
                 discriminator,
                 amp_data,
                 amp_normalizer,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 disc_learning_rate=None,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 amp_replay_buffer_size=100_000,
                 min_std=None,
                 max_std=None,
                 disc_target_expert=None,
                 disc_lr_bounds=(1.0e-5, 3.0e-3),
                 disc_update_ratio_min=0.2,
                 disc_adapt_gain=1.15,
                 ):

        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.disc_learning_rate = disc_learning_rate if disc_learning_rate is not None else learning_rate
        self.min_std = min_std
        self.max_std = max_std

        # ---- Discriminator balance controller ----------------------------
        # The AMP reward is coef * clamp(1 - (d-1)^2/4, 0) * dt, maximal at d = +1.  A
        # discriminator that collapses towards d = 0 for everything therefore hands the policy
        # 75% of the maximum imitation reward for free, while carrying no information about
        # whether the motion actually resembles the expert.  That is the failure this run
        # keeps hitting: d_expert drifting down from 0.77 to 0.63 while imi_reward rises.
        #
        # A fixed disc_learning_rate cannot prevent it, because the right learning rate
        # depends on how fast the policy is currently moving.  So d_expert is treated as the
        # controlled variable: when it falls below the band the discriminator is losing and is
        # given more learning rate and more update steps; when it rises above the band the
        # discriminator is winning so hard that its gradient to the policy vanishes, and both
        # are pulled back.  Set disc_target_expert=None to restore the fixed-rate behaviour.
        self.disc_target_expert = tuple(disc_target_expert) if disc_target_expert else None
        self.disc_lr_bounds = tuple(disc_lr_bounds)
        self.disc_update_ratio_min = float(disc_update_ratio_min)
        self.disc_adapt_gain = float(disc_adapt_gain)
        self.disc_update_ratio = 1.0

        # Discriminator
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = None
        self.amp_replay_buffer_size = amp_replay_buffer_size
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # Policy
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None

        # Policy optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # Discriminator parameters
        self.skill_insert = self.discriminator.skill_insert
        if self.skill_insert == 'concat':
            disc_params = [
                {'params': self.discriminator.trunk.parameters(),      'weight_decay': 10e-4},
                {'params': self.discriminator.amp_linear.parameters(), 'weight_decay': 10e-2},
            ]
        elif self.skill_insert == 'project':
            disc_params = [
                {'params': self.discriminator.trunk.parameters(),      'weight_decay': 10e-4},
                {'params': self.discriminator.amp_linear.parameters(), 'weight_decay': 10e-2},
                {'params': self.discriminator.embed.parameters(),      'weight_decay': 0},
            ]

        # Discriminator optimizer — always Adam (LSGAN only)
        self.disc_optimizer = optim.Adam(disc_params, lr=self.disc_learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO hyper-params
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    # ------------------------------------------------------------------

    def init_storage(self, num_envs, num_transitions_per_env,
                     actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env,
            actor_obs_shape, critic_obs_shape, action_shape, self.device)
        self.amp_storage = ReplayBuffer_SECAMP(
            H=self.discriminator.amp_obs_horizon,
            amp_obs_dim=self.discriminator.amp_obs_dim,
            skill_cmd_dim=self.discriminator.skill_cmd_dim,
            buf_size=self.amp_replay_buffer_size,
            device=self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    # ------------------------------------------------------------------

    def act(self, obs, critic_obs, amp_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()
        self.transition.actions = self.actor_critic.act(aug_obs).detach()
        self.transition.values = self.actor_critic.evaluate(aug_critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions).detach()
        self.transition.action_mean  = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos,
                         amp_obs, skill_cmd,
                         amp_valid_mask=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        if amp_valid_mask is not None:
            valid_amp_obs = amp_obs[amp_valid_mask]
            valid_skill   = skill_cmd[amp_valid_mask]
        else:
            valid_amp_obs = amp_obs
            valid_skill   = skill_cmd

        if valid_amp_obs.shape[0] > 0:
            self.amp_storage.insert(valid_amp_obs, valid_skill)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs.detach()).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    # ------------------------------------------------------------------

    def _adapt_discriminator(self, mean_expert_pred: float) -> None:
        """Hold d_expert inside the target band by moving the discriminator's own budget.

        d_expert is the discriminator's LSGAN score on real expert motion, whose target is
        +1.  Read as a health metric it says how much of the expert/policy gap the
        discriminator can still resolve:

          * below the band, the discriminator has been out-run by the policy.  Its output
            collapses towards 0, where the AMP reward is 0.75 of maximum for *any* motion, so
            imitation reward rises while imitation quality does not.  Fix: train it harder.
          * above the band, the discriminator separates the two distributions so cleanly that
            the policy's reward gradient flattens out -- the mirror image of the torque
            saturation problem, in reward space instead of actuator space.  Fix: hold it back.

        Both knobs move together because they act on the same quantity (gradient steps per
        policy update) at different granularities, and one alone saturates: the learning rate
        hits its bound while the imbalance persists.
        """
        if self.disc_target_expert is None:
            return
        low, high = self.disc_target_expert
        gain = self.disc_adapt_gain
        lr_min, lr_max = self.disc_lr_bounds
        if mean_expert_pred < low:
            self.disc_learning_rate = min(lr_max, self.disc_learning_rate * gain)
            self.disc_update_ratio = min(1.0, self.disc_update_ratio * gain)
        elif mean_expert_pred > high:
            self.disc_learning_rate = max(lr_min, self.disc_learning_rate / gain)
            self.disc_update_ratio = max(self.disc_update_ratio_min,
                                         self.disc_update_ratio / gain)
        else:
            return
        for group in self.disc_optimizer.param_groups:
            group['lr'] = self.disc_learning_rate

    def update(self):
        mean_value_loss     = 0
        mean_surrogate_loss = 0
        mean_amp_loss       = 0
        mean_grad_pen_loss  = 0
        mean_policy_pred    = 0
        mean_expert_pred    = 0
        disc_steps          = 0
        # Fractional-step accumulator: a ratio of 0.4 takes a discriminator step on 2 of
        # every 5 minibatches, spread evenly, rather than on a random subset.
        disc_budget         = 0.0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        batch_size = (self.storage.num_envs * self.storage.num_transitions_per_env
                      // self.num_mini_batches)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            num_updates, batch_size)

        for sample, sample_amp_policy in zip(generator, amp_policy_generator):

            (obs_batch, critic_obs_batch, actions_batch, target_values_batch,
             advantages_batch, returns_batch, old_actions_log_prob_batch,
             old_mu_batch, old_sigma_batch,
             hid_states_batch, masks_batch) = sample

            # ---- Policy forward -----------------------------------------
            aug_obs_batch = obs_batch.detach()
            self.actor_critic.act(aug_obs_batch, masks=masks_batch,
                                  hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch.detach(), masks=masks_batch,
                hidden_states=hid_states_batch[1])
            mu_batch      = self.actor_critic.action_mean
            sigma_batch   = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # ---- Adaptive LR -------------------------------------------
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5) +
                        (torch.square(old_sigma_batch) +
                         torch.square(old_mu_batch - mu_batch)) /
                        (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = self.learning_rate

            # ---- Surrogate loss ----------------------------------------
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ---- Value loss --------------------------------------------
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2)).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # ---- LSGAN discriminator loss ------------------------------
            pol_seq, pol_skill = sample_amp_policy
            exp_seq = self.amp_data.sample(pol_skill)      # [B, H, D]

            pol_flat = pol_seq.flatten(1)                  # [B, H*D]
            exp_flat = exp_seq.flatten(1)                  # [B, H*D]
            exp_skill = pol_skill                          # condition expert on policy skill

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    pol_flat = self.amp_normalizer.normalize_torch(pol_flat, self.device)
                    exp_flat = self.amp_normalizer.normalize_torch(exp_flat, self.device)

            policy_d = self.discriminator(pol_flat, pol_skill)
            expert_d = self.discriminator(exp_flat, exp_skill)

            expert_loss  = nn.MSELoss()(expert_d, torch.ones_like(expert_d))
            policy_loss  = nn.MSELoss()(policy_d, -torch.ones_like(policy_d))
            amp_loss     = 0.5 * (expert_loss + policy_loss)

            # Whether this minibatch trains the discriminator.  The forward passes above are
            # always run, so d_expert/d_policy stay measured over the whole iteration even
            # when the controller has throttled the updates.  The gradient penalty is a
            # double backward and is by far the most expensive part, so it is only paid for
            # on minibatches that actually take a step.
            disc_budget += self.disc_update_ratio
            take_disc_step = disc_budget >= 1.0
            if take_disc_step:
                disc_budget -= 1.0
                grad_pen_loss = self.discriminator.compute_grad_pen(
                    exp_flat, exp_skill, lambda_=10)

            # ---- Policy gradient step ----------------------------------
            policy_loss_total = (surrogate_loss
                                 + self.value_loss_coef * value_loss
                                 - self.entropy_coef * entropy_batch.mean())
            self.optimizer.zero_grad()
            policy_loss_total.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # ---- Discriminator gradient step ---------------------------
            if take_disc_step:
                disc_loss = amp_loss + grad_pen_loss
                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                self.disc_optimizer.step()
                mean_grad_pen_loss += grad_pen_loss.item()
                disc_steps += 1

            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)
            # The entropy bonus has no upper bound of its own, and Adam normalises its
            # gradient, so std otherwise grows by roughly learning_rate per update
            # regardless of how large it already is.  Actions are clamped to [-1, 1], so
            # once std reaches the action range the extra noise only produces bang-bang
            # control and destroys the imitation signal.
            if not self.actor_critic.fixed_std and self.max_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(max=self.max_std)

            if self.amp_normalizer is not None:
                self.amp_normalizer.update(pol_flat.cpu().numpy())
                self.amp_normalizer.update(exp_flat.cpu().numpy())

            mean_value_loss     += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss       += amp_loss.item()
            mean_policy_pred    += policy_d.mean().item()
            mean_expert_pred    += expert_d.mean().item()

        mean_value_loss     /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss       /= num_updates
        mean_grad_pen_loss  /= max(disc_steps, 1)
        mean_policy_pred    /= num_updates
        mean_expert_pred    /= num_updates
        self._adapt_discriminator(mean_expert_pred)
        self.storage.clear()

        return (mean_value_loss, mean_surrogate_loss, mean_amp_loss,
                mean_grad_pen_loss, mean_policy_pred, mean_expert_pred)
