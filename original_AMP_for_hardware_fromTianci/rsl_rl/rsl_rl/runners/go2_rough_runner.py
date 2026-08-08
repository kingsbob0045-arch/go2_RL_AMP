import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCriticTwoHead
from rsl_rl.env import VecEnv
from rsl_rl.utils.seconds_to_hours import change_seconds_to_hours_fromat

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class Go2RoughRunner:
    """OnPolicy runner for Go2RoughEnv.

    Differences from the base OnPolicyRunner:
    1. Always uses ActorCriticTwoHead (no eval() lookup).
    2. Multiplies num_obs by include_history_steps for the actor input dim.
    3. log() reports separate noise std for residual and skill heads.
    """

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg        = train_cfg["runner"]
        self.alg_cfg    = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device     = device
        self.env        = env

        num_critic_obs = (self.env.num_privileged_obs
                          if self.env.num_privileged_obs is not None
                          else self.env.num_obs)

        # Actor input: flattened history obs
        if self.env.include_history_steps is not None:
            num_actor_obs = self.env.num_obs * self.env.include_history_steps
        else:
            num_actor_obs = self.env.num_obs

        actor_critic = ActorCriticTwoHead(
            num_actor_obs,
            num_critic_obs,
            self.env.num_actions,          # 15 (12 residual + 3 skill)
            **self.policy_cfg,
        ).to(self.device)

        self.alg: PPO = PPO(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval     = self.cfg["save_interval"]

        # Storage uses the full stacked obs shape
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer  = None
        self.tot_timesteps = 0
        self.tot_time      = 0
        self.current_learning_iteration = 0

        # wandb
        self.use_wandb = (
            self.cfg.get("enable_wandb", False)
            and _WANDB_AVAILABLE
            and log_dir is not None
        )
        if self.use_wandb:
            wandb.init(
                project=self.cfg.get("wandb_project", "go2_rough"),
                entity=self.cfg.get("wandb_entity", None),
                name=self.cfg.get("run_name", None),
                dir=log_dir,
                config={
                    "policy":    train_cfg.get("policy",    {}),
                    "algorithm": train_cfg.get("algorithm", {}),
                    "runner":    train_cfg.get("runner",    {}),
                },
                sync_tensorboard=False,
            )

        _, _ = self.env.reset()

    # ------------------------------------------------------------------

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos   = []
        rewbuffer  = deque(maxlen=100)
        lenbuffer  = deque(maxlen=100)
        cur_reward_sum     = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos, _, _ = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device), critic_obs.to(self.device),
                        rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids]     = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            self.alg.actor_critic.clamp_std()
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir,
                               f'model_{self.current_learning_iteration}.pt'))
        if self.use_wandb:
            wandb.finish()

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
                ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"

        ac = self.alg.actor_critic
        mean_std_residual = ac.std_residual.mean().item()
        mean_std_skill    = ac.std_skill.mean().item()
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)

        self.writer.add_scalar('Loss/value_function',    locs['mean_value_loss'],     locs['it'])
        self.writer.add_scalar('Loss/surrogate',         locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate',     self.alg.learning_rate,      locs['it'])
        self.writer.add_scalar('Policy/std_residual',    mean_std_residual,           locs['it'])
        self.writer.add_scalar('Policy/std_skill',       mean_std_skill,              locs['it'])
        self.writer.add_scalar('Perf/total_fps',         fps,                         locs['it'])
        self.writer.add_scalar('Perf/collection_time',   locs['collection_time'],     locs['it'])
        self.writer.add_scalar('Perf/learning_time',     locs['learn_time'],          locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward',          statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length',  statistics.mean(locs['lenbuffer']), locs['it'])

        if self.use_wandb:
            wandb_log = {
                'Loss/value_function':   locs['mean_value_loss'],
                'Loss/surrogate':        locs['mean_surrogate_loss'],
                'Loss/learning_rate':    self.alg.learning_rate,
                'Policy/std_residual':   mean_std_residual,
                'Policy/std_skill':      mean_std_skill,
                'Perf/total_fps':        fps,
                'Perf/collection_time':  locs['collection_time'],
                'Perf/learning_time':    locs['learn_time'],
            }
            if len(locs['rewbuffer']) > 0:
                wandb_log['Train/mean_reward']         = statistics.mean(locs['rewbuffer'])
                wandb_log['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            if locs['ep_infos']:
                for key in locs['ep_infos'][0]:
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in locs['ep_infos']:
                        v = ep_info[key]
                        if not isinstance(v, torch.Tensor):
                            v = torch.Tensor([v])
                        if len(v.shape) == 0:
                            v = v.unsqueeze(0)
                        infotensor = torch.cat((infotensor, v.to(self.device)))
                    wandb_log['Episode/' + key] = torch.mean(infotensor).item()
            wandb.log(wandb_log, step=locs['it'])

        str_ = (f" \033[1m Learning iteration "
                f"{locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']}"
                f" \033[0m ")
        common = (f"{'Computation:':>{pad}} {fps:.0f} steps/s "
                  f"(collection: {locs['collection_time']:.3f}s, "
                  f"learning {locs['learn_time']:.3f}s)\n"
                  f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
                  f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
                  f"{'Residual noise std:':>{pad}} {mean_std_residual:.2f}\n"
                  f"{'Skill noise std:':>{pad}} {mean_std_skill:.2f}\n")
        if len(locs['rewbuffer']) > 0:
            common += (f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                       f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n")

        eta_secs = self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it'])
        eta_str  = change_seconds_to_hours_fromat(eta_secs)
        log_string = f"{'#' * width}\n{str_.center(width, ' ')}\n\n{common}{ep_string}"
        log_string += (f"{'-' * width}\n"
                       f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
                       f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
                       f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
                       f"""{'ETA:':>{pad}} {eta_str}h\n""")
        print(log_string)

    # ------------------------------------------------------------------

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict':     self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter':                 self.current_learning_iteration,
            'infos':                infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
