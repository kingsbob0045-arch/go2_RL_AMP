# go2_camp_posTracking

Conditional AMP (CAMP) locomotion for Go2 with **waypoint-based navigation** and **categorical skill commands** (pace / trot / canter). The robot must travel through a 2-waypoint path while imitating reference motion captured for each gait style.

---

## File Structure

```
go2_camp_posTracking/
├── go2_camp_posTracking_env.py      # IsaacGym environment
├── go2_camp_posTracking_config.py   # Env + training hyper-parameters
└── README.md                        # This file

rsl_rl/rsl_rl/
├── algorithms/camp_posTracking_ppo.py   # PPO + LSGAN discriminator update
├── runners/camp_posTracking_runner.py   # Training loop (no curriculum)
├── datasets/camp_motion_loader.py       # Skill-conditioned reference loader
├── modules/camp_disc_project.py         # Projection discriminator
└── storage/replay_buffer_camp.py        # Circular AMP replay buffer
```

---

## Data Flow

### 1. Episode Initialisation

```
reset_idx(env_ids)
  └─ _resample_skill_cmds(env_ids)   → sample one-hot skill ∈ {pace, trot, canter}
  └─ _resample_commands(env_ids)     → sample 2 chained XY waypoints
       wp0: random direction from robot position
       wp1: within ±max_turn_angle (π/2) of wp0 leg direction
       stored in waypoints[N, K=2, 2]  (XY only)
```

### 2. Every Simulation Step

```
post_physics_step()
  ├─ refresh tensors (root_states, contacts, …)
  ├─ _update_pos_target()
  │    ├─ dist_xy < arrival_threshold → advance waypoint_idx
  │    ├─ dist_xy < lookahead_threshold → switch obs target to next wp (lookahead)
  │    ├─ at last waypoint → immediately resample new 2-waypoint path
  │    └─ pos_target_world ← waypoints[arange_N, obs_idx]   [N, 2]
  ├─ check_termination()
  ├─ compute_reward()                 → _reward_tracking_pos()
  └─ compute_observations()
```

### 3. Observation Construction

```
compute_observations()
  ├─ yaw  ← quaternion → atan2
  ├─ delta_world = pos_target_world - root_states[:, :2]   (XY, world frame)
  ├─ [dx_body, dy_body] ← rotate delta_world by -yaw       (body frame)
  ├─ commands[:, 0] = dx_body
  │  commands[:, 1] = dy_body
  │  commands[:, 2] = 0            (Z slot unused)
  ├─ super().compute_observations() → obs_buf [N, 42], privileged_obs_buf [N, 48]
  └─ cat skill_cmd_buf [N, 3]
       → obs_buf [N, 45]  /  privileged_obs_buf [N, 51]
```

**Observation layout (actor 45-dim / critic 51-dim):**

| Dims (actor) | Dims (critic) | Content |
|---|---|---|
| — | 0:3 | `base_lin_vel × 2.0` (privileged) |
| — | 3:6 | `base_ang_vel × 0.25` (privileged) |
| 0:3 | 6:9 | `projected_gravity` |
| 3:5 | 9:11 | `[dx_body × 0.5, dy_body × 0.5]` (XY to waypoint) |
| 5 | 11 | `0.0` (Z slot, unused) |
| 6:18 | 12:24 | `dof_pos` |
| 18:30 | 24:36 | `dof_vel` |
| 30:42 | 36:48 | `actions` |
| 42:45 | 48:51 | `skill_cmd` one-hot [pace, trot, canter] |

### 4. Task Reward

Implements the waypoint-direction velocity tracking from the parkour paper:

```
d_w = (p - x) / ||p - x||                  # unit direction to waypoint (world XY)
r   = clamp(v_world · d_w,  max=v_cmd[skill])
```

- `v_world`: robot XY velocity in **world frame** (not base frame — prevents turn-in-place exploit)
- `v_cmd` is skill-conditioned: `pace=1.0`, `trot=1.5`, `canter=3.0` m/s
- Reward is naturally negative when robot moves away from waypoint

### 5. AMP Reward (Imitation)

```
runner.learn() per step:
  skill_cmd ← env.get_skill_cmd_buf()              # capture before step
  amp_obs_buf [N, H=5, 43] ← env.get_amp_observation_buf()

  discriminator.predict_amp_reward(amp_obs_buf, skill_cmd, task_rewards)
    └─ d = disc(amp_obs_flat, skill_cmd)            # projection discriminator
    └─ r_imi = amp_reward_coef × clamp(1 - ¼(d-1)², 0)   # LSGAN reward
    └─ r_total = lerp × r_task + (1-lerp) × r_imi
```

**AMP observation (43-dim per frame, H=5 frames):**
`joint_pos(12) + foot_pos(12) + base_lin_vel(3) + base_ang_vel(3) + joint_vel(12) + z_pos(1)`

### 6. Discriminator Training (LSGAN)

```
camp_posTracking_ppo.update() per mini-batch:
  pol_seq, pol_skill ← amp_replay_buffer.sample()       # [B, H, 43], [B, 3]
  exp_seq            ← camp_motion_loader.sample(pol_skill)   # skill-matched expert
  pol_flat / exp_flat ← flatten + normalize              # [B, H×43]

  expert_d = disc(exp_flat, pol_skill)   # expert conditioned on policy skill
  policy_d = disc(pol_flat, pol_skill)

  amp_loss      = 0.5 × [MSE(expert_d, +1) + MSE(policy_d, -1)]   # LSGAN
  grad_pen_loss = zero-centred GP on expert data
  disc_loss     = amp_loss + grad_pen_loss
```

> **Note:** Expert data has no skill labels. We condition both expert and policy on `pol_skill` so the discriminator judges "motion quality given this skill context."

### 7. Skill-Conditioned Expert Sampling

`CAMPLoader.sample(skill_cmd)` decodes the one-hot `skill_cmd` per sample to look up pre-indexed motion trajectories, returning H-frame sequences that match the requested gait.

---

## Key Design Decisions

### Waypoint Navigation (vs. velocity commands)
The robot cannot follow arbitrary velocity directions on arbitrary terrain. Instead, 2 chained waypoints are sampled in the world frame each episode. The policy receives the **body-frame XY direction** to the current waypoint — not the absolute position — so the representation stays egocentric and bounded.

### Lookahead
When the robot is within `lookahead_threshold` (0.5 m) of waypoint k, the observation target switches to waypoint k+1. This removes the incentive to decelerate and stop exactly at each waypoint.

### Turn Constraint
wp1 is sampled within `±max_turn_angle` (π/2 = 90°) of the direction of the first leg. This prevents backtracking and keeps the route physically reasonable.

### World-Frame Velocity Reward
The reward uses `v_world · d_w` (world frame) rather than `v_base · d_cmd` (base frame). In base frame the robot can rotate in place to maximise the dot-product without making lateral progress. World frame closes this exploit.

### Skill-Conditioned Speed Cap
`v_cmd` in the reward is skill-indexed: `[pace=1.0, trot=1.5, canter=3.0]` m/s. The cap controls the maximum task reward achievable per gait. Paired with AMP's imitation reward, this steers each skill to its natural speed range without explicitly commanding velocity.

### LSGAN + Projection Discriminator
- **LSGAN** (least-squares GAN): MSE loss, stable training, no mode collapse risk of vanilla GAN
- **Projection discriminator**: skill embedding is projected into feature space and dot-producted with trunk features. More parameter-efficient than concatenation; gradient flows cleanly through both the obs branch and the skill branch
- Discriminator optimizer: Adam (not RMSprop — no WGAN)

### Lerp Schedule
`task_reward_lerp` ramps from 0.15 → 0.4 over 10 000 iterations. Early training is dominated by imitation (AMP reward), giving the policy a stable motion prior before the task reward starts pulling it toward waypoints.

### Reference State Initialisation (RSI)
85% of episode resets initialise the robot at a random frame from the reference motion dataset (matched to the sampled skill). This keeps the policy close to the demonstration manifold throughout training and accelerates convergence.

---

## Configuration Quick Reference

| Parameter | Value | Note |
|---|---|---|
| `num_envs` | 5480 | Training environments |
| `num_observations` | 45 | Actor obs dim |
| `num_privileged_obs` | 51 | Critic obs dim |
| `amp_horizon` (H) | 5 | AMP observation window (frames) |
| `amp_num_obs` | 43 | AMP obs per frame |
| `num_waypoints` | 2 | Waypoints per episode |
| `pos_target_dist_range` | [5, 15] m | Per-waypoint XY distance |
| `max_turn_angle` | π/2 | Max direction change between legs |
| `arrival_threshold` | 0.2 m | Waypoint advance distance |
| `lookahead_threshold` | 0.5 m | Lookahead switch distance |
| `tracking_vel_cmd` | [1.0, 1.5, 3.0] | Speed cap per skill [m/s] |
| `tracking_pos` scale | 2.0 | Task reward weight |
| `amp_reward_coef` | 2.0 | AMP reward coefficient |
| `amp_task_reward_lerp` | 0.4 | Final task/imi blend ratio |
| `disc_learning_rate` | 1e-3 | Discriminator Adam LR |
| `skill_insert` | project | Discriminator skill conditioning |
