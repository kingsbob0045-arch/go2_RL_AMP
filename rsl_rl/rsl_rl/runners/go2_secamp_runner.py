import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
import wandb

from rsl_rl.algorithms.secamp_ppo import SECAMPPPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from rsl_rl.modules.secamp_disc_concat import SECAMPDiscriminator_Concat
from rsl_rl.modules.secamp_disc_project import SECAMPDiscriminator_Project
from rsl_rl.datasets.secamp_motion_loader import SECAMPLoader
from rsl_rl.utils.utils import Normalizer
from rsl_rl.utils.seconds_to_hours import change_seconds_to_hours_fromat


class Go2SECAMPRunner:
    """Runner for go2_secamp waypoint-tracking locomotion (pace / trot / canter)."""

    def __init__(
            self,
            env: VecEnv,
            train_cfg,
            log_dir=None,
            device='cpu'
        ):

        self.cfg        = train_cfg["runner"]
        self.alg_cfg    = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device     = device
        self.env        = env

        num_critic_obs = (self.env.num_privileged_obs
                          if self.env.num_privileged_obs is not None
                          else self.env.num_obs)
        num_actor_obs  = (self.env.num_obs * self.env.include_history_steps
                          if self.env.include_history_steps is not None
                          else self.env.num_obs)

        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCritic = actor_critic_class(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=self.env.num_actions,
            **self.policy_cfg).to(self.device)

        # ----- CAMP env attributes ----------------------------------------
        self.amp_horizon   = self.env.amp_horizon
        self.amp_num_obs   = self.env.amp_num_obs
        self.enable_skill  = self.env.enable_skill
        self.skill_cmd_dim = self.env.skill_cmd_dim

        # ----- Expert motion loader ---------------------------------------
        amp_data = SECAMPLoader(
            device,
            time_between_frames=self.env.dt,
            preload_transitions=True,
            num_preload_transitions=self.cfg['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"],
            reorder=self.cfg.get("amp_motion_reorder", False),
            num_amp_obs=self.amp_num_obs,
            amp_horizon=self.amp_horizon,
            skill_cmd_dim=self.skill_cmd_dim,
        )

        amp_normalizer = Normalizer(self.amp_num_obs * self.amp_horizon)

        # ----- Conditional discriminator (LSGAN only) ---------------------
        _skill_insert = self.cfg.get('skill_insert', 'concat')
        _disc_kwargs = dict(
            amp_obs_dim=self.amp_num_obs,
            amp_obs_horizon=self.amp_horizon,
            amp_reward_coef=self.cfg['amp_reward_coef'],
            hidden_layer_sizes=self.cfg['amp_discr_hidden_dims'],
            device=device,
            task_reward_lerp=self.cfg['amp_task_reward_lerp'],
            skill_cmd_dim=self.skill_cmd_dim,
        )
        self._disc_kwargs     = _disc_kwargs   # saved for potential re-instantiation at load
        self._skill_insert    = _skill_insert

        if _skill_insert == 'project':
            discriminator = SECAMPDiscriminator_Project(**_disc_kwargs).to(self.device)
        else:
            discriminator = SECAMPDiscriminator_Concat(**_disc_kwargs).to(self.device)

        # ----- Algorithm --------------------------------------------------
        min_std = (
            torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
            torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0]))
        self.alg = SECAMPPPO(
            actor_critic, discriminator, amp_data, amp_normalizer,
            device=self.device, min_std=min_std, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval     = self.cfg["save_interval"]

        self.alg.init_storage(
            self.env.num_envs, self.num_steps_per_env,
            [num_actor_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Logging
        self.log_dir  = log_dir
        self.writer   = None
        self.wandb_run = None
        self.tot_timesteps = 0
        self.tot_time      = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

        # ----- Lerp schedule: ramp task_reward_lerp from lerp_init to final
        self.lerp_schedule_enabled = self.cfg.get('lerp_schedule_enabled', True)
        self.lerp_init         = 0.15
        self.lerp_final        = self.cfg['amp_task_reward_lerp']
        self.lerp_warmup_iters = 10000
        if self.lerp_schedule_enabled:
            self.current_lerp = self.lerp_init
            self.alg.discriminator.task_reward_lerp = self.lerp_init
        else:
            self.current_lerp = self.lerp_final

    # ------------------------------------------------------------------
    # Learning loop
    # ------------------------------------------------------------------

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            if self.cfg["wandb_enable"]:
                self.wandb_run = wandb.init(
                    project=self.cfg.get("experiment_name", "legged_gym"),
                    name=self.cfg.get("run_name", None),
                    dir=self.log_dir,
                    config={**self.alg_cfg, **self.policy_cfg},
                    resume="allow",
                )

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length))

        obs            = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs_buf    = self.env.get_amp_observation_buf()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_obs_buf = (obs.to(self.device),
                                        critic_obs.to(self.device),
                                        amp_obs_buf.to(self.device))

        self.alg.actor_critic.train()
        self.alg.discriminator.train()

        ep_infos       = []
        rewbuffer      = deque(maxlen=100)
        lenbuffer      = deque(maxlen=100)
        imi_rewbuffer  = deque(maxlen=100)
        task_rewbuffer = deque(maxlen=100)
        cur_reward_sum    = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_imirew_sum    = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_taskrew_sum   = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        # Block cross-episode sequences from replay buffer for H-1 steps
        amp_cooldown = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    skill_cmd = self.env.get_skill_cmd_buf().to(self.device).clone()

                    actions = self.alg.act(obs, critic_obs, amp_obs_buf)
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = \
                        self.env.step(actions)
                    next_amp_obs_buf = self.env.get_amp_observation_buf()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs_buf, rewards, dones = (
                        obs.to(self.device), critic_obs.to(self.device),
                        next_amp_obs_buf.to(self.device),
                        rewards.to(self.device), dones.to(self.device))

                    # Substitute terminal sequences for reset envs
                    next_amp_obs_buf_with_term = torch.clone(next_amp_obs_buf)
                    next_amp_obs_buf_with_term[reset_env_ids] = terminal_amp_states

                    # Cooldown: block cross-episode sequences
                    if len(reset_env_ids) > 0:
                        amp_cooldown[reset_env_ids] = self.amp_horizon - 1
                    amp_valid_mask = (amp_cooldown == 0)
                    amp_cooldown = (amp_cooldown - 1).clamp(min=0)

                    # AMP reward
                    rewards, _, imi_rewards, task_rewards = \
                        self.alg.discriminator.predict_amp_reward(
                            amp_obs_buf, skill_cmd, rewards,
                            dt=self.env.dt, normalizer=self.alg.amp_normalizer)

                    amp_obs_buf = torch.clone(next_amp_obs_buf)

                    self.alg.process_env_step(
                        rewards, dones, infos,
                        next_amp_obs_buf_with_term, skill_cmd,
                        amp_valid_mask=amp_valid_mask)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum     += rewards
                        cur_imirew_sum     += imi_rewards
                        cur_taskrew_sum    += task_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        imi_rewbuffer.extend(cur_imirew_sum[new_ids][:, 0].cpu().numpy().tolist())
                        task_rewbuffer.extend(cur_taskrew_sum[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids]     = 0
                        cur_episode_length[new_ids] = 0
                        cur_imirew_sum[new_ids]     = 0
                        cur_taskrew_sum[new_ids]    = 0

            stop = time.time()
            collection_time = stop - start
            start = stop

            self.alg.compute_returns(critic_obs.clone())

            (mean_value_loss, mean_surrogate_loss, mean_amp_loss,
             mean_grad_pen_loss, mean_policy_pred, mean_expert_pred) = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            # Lerp schedule
            if self.lerp_schedule_enabled:
                self.current_lerp = (
                    self.lerp_init +
                    (self.lerp_final - self.lerp_init) *
                    min(it / max(self.lerp_warmup_iters, 1), 1.0))
                self.alg.discriminator.task_reward_lerp = self.current_lerp

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir,
                               'model_{}.pt'.format(self.current_learning_iteration)))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs /
                  (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function',  locs['mean_value_loss'],     locs['it'])
        self.writer.add_scalar('Loss/surrogate',        locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP',              locs['mean_amp_loss'],       locs['it'])
        self.writer.add_scalar('Loss/AMP_grad',         locs['mean_grad_pen_loss'],  locs['it'])
        self.writer.add_scalar('Loss/learning_rate',    self.alg.learning_rate,      locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(),             locs['it'])
        self.writer.add_scalar('Perf/total_fps',        fps,                         locs['it'])
        self.writer.add_scalar('Perf/collection_time',  locs['collection_time'],     locs['it'])
        self.writer.add_scalar('Perf/learning_time',    locs['learn_time'],          locs['it'])
        self.writer.add_scalar('Train/lerp',            self.current_lerp,           locs['it'])

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward',         statistics.mean(locs['rewbuffer']),      locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']),      locs['it'])
            self.writer.add_scalar('Train/mean_task_reward',    statistics.mean(locs['task_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_imi_reward',     statistics.mean(locs['imi_rewbuffer']),  locs['it'])
            self.writer.add_scalar('Disc/d_expert',             locs['mean_expert_pred'],                locs['it'])
            self.writer.add_scalar('Disc/d_policy',             locs['mean_policy_pred'],                locs['it'])

        header = (f" \033[1m Learning iteration "
                  f"{locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']}"
                  f" \033[0m ")

        if len(locs['rewbuffer']) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean task reward:':>{pad}} {statistics.mean(locs['task_rewbuffer']):.2f}\n"""
                f"""{'Mean imi reward:':>{pad}} {statistics.mean(locs['imi_rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        log_string += ep_string
        eta_secs = (self.tot_time / (locs['it'] + 1) *
                    (locs['num_learning_iterations'] - locs['it']))
        eta_str = change_seconds_to_hours_fromat(eta_secs)
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {eta_str}h\n"""
        )

        if self.wandb_run is not None:
            wandb_log = {
                'Loss/value_function':   locs['mean_value_loss'],
                'Loss/surrogate':        locs['mean_surrogate_loss'],
                'Loss/AMP':              locs['mean_amp_loss'],
                'Loss/AMP_grad':         locs['mean_grad_pen_loss'],
                'Loss/learning_rate':    self.alg.learning_rate,
                'Policy/mean_noise_std': mean_std.item(),
                'Perf/total_fps':        fps,
                'Perf/collection_time':  locs['collection_time'],
                'Perf/learning_time':    locs['learn_time'],
                'Perf/total_timesteps':  self.tot_timesteps,
                'Train/lerp':            self.current_lerp,
            }
            if len(locs['rewbuffer']) > 0:
                wandb_log.update({
                    'Train/mean_reward':         statistics.mean(locs['rewbuffer']),
                    'Train/mean_episode_length': statistics.mean(locs['lenbuffer']),
                    'Train/mean_task_reward':    statistics.mean(locs['task_rewbuffer']),
                    'Train/mean_imi_reward':     statistics.mean(locs['imi_rewbuffer']),
                    'Disc/d_expert':             locs['mean_expert_pred'],
                    'Disc/d_policy':             locs['mean_policy_pred'],
                })
            if locs['ep_infos']:
                for key in locs['ep_infos'][0]:
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in locs['ep_infos']:
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                    wandb_log['Episode/' + key] = torch.mean(infotensor).item()
            wandb.log(wandb_log, step=locs['it'])

        print(log_string)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict':           self.alg.actor_critic.state_dict(),
            'optimizer_state_dict':       self.alg.optimizer.state_dict(),
            'disc_optimizer_state_dict':  self.alg.disc_optimizer.state_dict(),
            'discriminator_state_dict':   self.alg.discriminator.state_dict(),
            'amp_normalizer':             self.alg.amp_normalizer,
            'iter':                       self.current_learning_iteration,
            'infos':                      infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])

        disc_state = loaded_dict['discriminator_state_dict']

        # Auto-detect discriminator type from checkpoint keys and rebuild if
        # it doesn't match the currently instantiated discriminator.
        ckpt_is_project = 'embed.weight' in disc_state
        curr_is_project = isinstance(self.alg.discriminator, SECAMPDiscriminator_Project)
        disc_type_switched = ckpt_is_project != curr_is_project
        if disc_type_switched:
            saved_type = 'project' if ckpt_is_project else 'concat'
            print(f"[Go2SECAMPRunner] Checkpoint discriminator type "
                  f"('{saved_type}') differs from config ('{self._skill_insert}'). "
                  f"Switching discriminator to '{saved_type}' to match checkpoint.")
            if ckpt_is_project:
                new_disc = SECAMPDiscriminator_Project(**self._disc_kwargs).to(self.device)
            else:
                new_disc = SECAMPDiscriminator_Concat(**self._disc_kwargs).to(self.device)
            self.alg.discriminator = new_disc
            # Rebuild disc optimiser with the new parameters
            import torch.optim as optim
            self.alg.disc_optimizer = optim.Adam(
                new_disc.parameters(), lr=self.alg.disc_learning_rate)

        self.alg.discriminator.load_state_dict(disc_state)
        self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            if 'disc_optimizer_state_dict' in loaded_dict and not disc_type_switched:
                self.alg.disc_optimizer.load_state_dict(loaded_dict['disc_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
