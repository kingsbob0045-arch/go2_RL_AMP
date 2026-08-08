# go2_RoughTerrain — Residual Policy over a Frozen Motion Prior

## Overview

This task trains a **two-headed residual actor** to control a Go2 quadruped on rough terrain.
A frozen TorchScript motion prior (pace / trot / canter) handles low-level gait generation;
the learned policy adjusts that output via residual actions and selects the gait via a skill command.

```
                ┌──────────────────────────────────────────┐
                │          ActorCriticTwoHead               │
obs_history     │  backbone MLP (210→512→256)               │
[N, 5×42=210] ──►                                           │
                │  residual_head (256→12)  skill_head(256→3)│
                └────────┬─────────────────────┬────────────┘
                         │ residual [N,12]      │ skill logits [N,3]
                         │                      │ softmax ↓
                         │              skill_cmd [N,3]  ──► frozen prior
                         │                                     [N,45]→[N,12]
                         │         prior_action [N,12] ◄──────┘
                         │
              joint_action = prior_action + scale × residual   [N,12]
                         │
                    PD controller → torques [N,12]
```

---

## Key Files

| File | Role |
|------|------|
| `go2_rough_env.py` | `Go2RoughEnv` — env override |
| `go2_rough_config.py` | `GO2RoughCfg / GO2RoughCfgPPO` — hyperparameters |
| `rsl_rl/modules/actor_critic_two_head.py` | `ActorCriticTwoHead` — network |
| `rsl_rl/runners/go2_rough_runner.py` | `Go2RoughRunner` — training loop |

---

## Architecture

### Observation dimensions

| Buffer | Shape | Contents |
|--------|-------|----------|
| `obs_buf` | `[N, 42]` | gravity(3) + cmd(3) + dof_pos(12) + dof_vel(12) + last_joint_action(12) |
| `privileged_obs_buf` | `[N, 48]` | lin_vel(3) + ang_vel(3) + obs_buf |
| actor input (policy obs) | `[N, 210]` | 5 history steps × 42 stacked newest-first |

### Action dimensions

| Signal | Shape | Description |
|--------|-------|-------------|
| Actor output | `[N, 15]` | `[:12]` residual, `[12:]` skill logits |
| `skill_cmd` | `[N, 3]` | `softmax(skill_logits)` — probability simplex |
| `prior_input` | `[N, 45]` | `obs_buf[t] ∥ skill_cmd` |
| `prior_action` | `[N, 12]` | frozen model output |
| `joint_action` | `[N, 12]` | `prior_action + residual_scale × residual` |

`residual_action_scale = 0.25` (set in config).

### Network — `ActorCriticTwoHead`

- **Backbone**: shared MLP `210 → 512 → 256 → ELU`
- **Residual head**: `Linear(256, 12)`
- **Skill head**: `Linear(256, 3)`
- **Critic**: separate MLP `48 → 512 → 256 → ELU → 1`
- **Exploration noise**: two separate `nn.Parameter` vectors — `std_residual[12]` and
  `std_skill[3]` — concatenated into a single `Normal(mean, std)` for PPO.

---

## Data Flow per Step

```
t=0  env.reset()
       └─ reset_idx() → compute_observations() → obs_buf[42]
       └─ obs_buf_history.reset(all, obs_buf) → history all zeros→current

t=k  runner: obs = env.get_observations()          # [N, 210] stacked history
             actions = policy.act(obs)              # [N, 15]

     env.step(actions):
       1. split:  residual = actions[:, :12]
                  skill_cmd = softmax(actions[:, 12:])
       2. prior:  prior_input  = cat([obs_buf_t, skill_cmd], dim=-1)  # [N,45]
                  prior_action = frozen_prior(prior_input)             # [N,12]
       3. combine: joint_action = prior_action + 0.25 × residual       # [N,12]
       4. super().step(joint_action):
            - self.actions = clip(joint_action)        # [N,12]
            - decimation loop → torques → physics
            - post_physics_step():
                compute_observations() → obs_buf_{t+1}  # [N,42]
                last_actions[:] = self.actions[:]        # [N,12]
            - obs_buf_history.reset(done_envs, obs_buf_{t+1}[done])
            - obs_buf_history.insert(obs_buf_{t+1})
            - policy_obs = history.get_obs_vec([0..4])  # [N,210]
       5. return (policy_obs, privileged_obs, rew, done, infos, ...)
```

### Timing invariant
`obs_buf` used to query the prior at step k is the observation produced by
`compute_observations()` at step k-1 — exactly the same temporal slice seen
by the actor. No look-ahead.

---

## `num_actions=15` vs `num_dof=12` Mismatch Fix

`LeggedRobot._init_buffers()` allocates `torques`, `p_gains`, `d_gains`,
`actions`, `last_actions` with `num_actions` dims.  With `num_actions=15`
the PD controller would receive a 15-element tensor for 12 joints.

**Fix in `Go2RoughEnv._init_buffers()`**: after calling `super()._init_buffers()`,
re-create those five buffers with `num_dof=12`.  The parent fills gains for
indices `0..num_dof-1` in order, so slicing `[:12]` is lossless.

`compute_randomized_gains()` is also overridden to use `num_dof` so that
`randomized_p_gains` / `randomized_d_gains` shapes stay `[N, 12]`.

After `super().step(joint_action)` (12-dim), `self.actions` is set to a
clipped 12-dim tensor by `LeggedRobot.step()`, keeping `last_actions` and
the obs consistent.

---

## `obs_buf` dimension consistency

`compute_observations()` (base class) produces:
```
privileged_obs = [lin_vel(3), ang_vel(3), gravity(3), cmd(3), dof_pos(12), dof_vel(12), actions(12)]
               = 48 dims
```
Because `num_obs (42) == num_privileged_obs (48) − 6`, the policy obs strips
the first 6 (lin+ang vel):
```
obs_buf = privileged_obs[:, 6:]   → 42 dims
```
`last_actions` (12-dim) in the obs records the actual **joint** action sent
to the PD controller, not the skill logit.

---

## Training

```bash
python legged_gym/scripts/train.py --task go2_rough --headless
```

Checkpoints are saved to `logs/go2_rough/<datetime>_<run_name>/`.

Config knobs (in `go2_rough_config.py`):

| Key | Default | Notes |
|-----|---------|-------|
| `residual_action_scale` | 0.25 | Scale of residual correction relative to prior |
| `skill_cmd_dim` | 3 | Number of gait skills (pace/trot/canter) |
| `include_history_steps` | 5 | Steps of obs history fed to actor |
| `backbone_hidden_dims` | [512, 256] | Shared backbone layers |
| `motion_prior` | `pretrained/[pos]+[lsgan]+[h2]+[project].pt` | Path relative to `LEGGED_GYM_ROOT_DIR` |

---

## Skill interpretation

The skill head outputs 3 logits passed through `softmax` before reaching the
prior.  The prior was trained with a one-hot or soft skill vector meaning:
- index 0 → pace
- index 1 → trot
- index 2 → canter

The actor can smoothly interpolate between gaits by distributing probability
mass across indices.

---

## Frozen prior contract

The prior TorchScript model must accept a `[N, 45]` float tensor
(`obs_42 ∥ skill_3`) and return `[N, 12]` joint position targets.
It is loaded with `torch.jit.load(...).eval()` and queried inside
`torch.inference_mode()` so it never accumulates gradients.
