# SECAMP: Skill-Conditioned Adversarial Motion Priors for Quadruped Locomotion

**SECAMP** is a multi-skill locomotion framework for quadruped robots that combines Adversarial Motion Priors (AMP) with skill-conditioned policy learning. A single policy learns to switch between natural gaits — **trot**, **pace**, and **canter** — while tracking waypoint-based navigation commands, all without hand-crafted reward shaping.

Trained entirely in [IsaacGym](https://developer.nvidia.com/isaac-gym) on the **Unitree Go2** robot, the framework supports sim-to-sim transfer to MuJoCo and direct sim-to-real deployment.

---

## Key Features

- **Multi-skill locomotion**: One policy controls trot (~1.5 m/s), pace (~1.0 m/s), and canter (~2.5 m/s) via a discrete one-hot skill code
- **Waypoint-based navigation**: Clamped position commands decouple gait from speed and enable natural deceleration at arrival
- **Dual-waypoint lookahead**: Encourages smooth cornering by pre-targeting the next waypoint before fully arriving at the current one
- **Conditional discriminator**: Projection-based architecture (inspired by cGANs) provides skill-specific imitation rewards without gradient vanishing
- **Symmetry augmentation**: Mirrored canter motions enable seamless left/right skill transitions from any gait phase
- **Asymmetric Actor-Critic**: Critic receives privileged state (ground-truth velocities) during training; deployed actor uses only on-board sensors

---

## System Overview

```
Motion Library (MoCap)          IsaacGym Environment
  trot / pace / canter    ──►   Go2 Robot + Waypoints (W1, W2)
  Skill-conditioned               │
  expert sampling                 │  Proprioceptive Obs (45D)
                                  │  Privileged Obs (51D)
                                  │  AMP Transitions (43×2D)
                                  ▼
             ┌─────────────────────────────────┐
             │   Skill-Aware Condition c        │
             │   [Skill Code (3D) | Command (3D)]│
             └─────────────────────────────────┘
                    │                  │
           ┌────────▼────────┐ ┌──────▼───────┐
           │  Actor π_θ      │ │  Critic V_ψ  │
           │  [1024,512,256] │ │  [1024,512,  │
           │  45D → 12 DoF   │ │   256]       │
           └────────┬────────┘ └──────┬───────┘
                    │                  │
             Joint Targets          Value Est.
                    │                  │
                    └──────────────────┘
                              │
                         PPO Update
                              │
           ┌──────────────────▼───────────────┐
           │      Discriminator D_φ            │
           │  Backbone + Skill Projection      │
           │  [1024, 512]  → r^I ∈ [0, 1]    │
           └──────────────────────────────────┘
```

---

## Installation

### 1. Create Conda Environment

```bash
conda create -n secamp python=3.8
conda activate secamp
```

### 2. Install PyTorch (CUDA 11.3)

```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### 3. Install IsaacGym

Download **IsaacGym Preview 3** from the [NVIDIA developer site](https://developer.nvidia.com/isaac-gym), then:

```bash
cd isaacgym/python && pip install -e .
# Verify: cd examples && python 1080_balls_of_solitude.py
```

### 4. Install This Repository

```bash
git clone <this-repo>
cd AMP_for_Hardware

# RL algorithms (PPO + SECAMP)
cd rsl_rl && pip install -e . && cd ..

# Legged gym environments
pip install -e .
```

---

## Repository Structure

```
AMP_for_Hardware/
├── legged_gym/
│   ├── envs/
│   │   ├── go2_secamp/          # SECAMP environment + config
│   │   ├── go2/                 # Standard AMP environment
│   │   └── base/                # Base legged robot class
│   └── scripts/
│       ├── train.py             # Training entry point
│       └── play.py              # Evaluation / visualization
├── rsl_rl/rsl_rl/
│   ├── algorithms/
│   │   ├── secamp_ppo.py        # SECAMP PPO algorithm
│   │   └── amp_ppo.py           # Standard AMP PPO
│   ├── modules/
│   │   ├── actor_critic.py      # Shared actor-critic network
│   │   ├── secamp_disc_project.py  # Projection discriminator
│   │   └── secamp_disc_concat.py   # Concat discriminator
│   ├── datasets/
│   │   └── secamp_motion_loader.py # Skill-conditioned data loader
│   └── runners/
│       └── go2_secamp_runner.py    # Training loop orchestration
├── datasets/
│   └── camp/                    # Reference motion clips (.json)
│       ├── trot0.json ... trot2.json
│       ├── pace0.json ... pace2.json
│       ├── canter0.json ... canter5.json  # includes mirrored
│       └── left_turn0.json, right_turn0.json ...
├── depoly_mujoco/               # MuJoCo sim-to-sim deployment
├── pretrained/                  # Pre-trained checkpoints
└── resources/                   # Go2 URDF / MJCF assets
```

---

## Training

### Train SECAMP (multi-skill waypoint tracking)

```bash
python legged_gym/scripts/train.py --task=go2_secamp --headless
```

Press `v` during training to toggle rendering on/off.

**Useful flags:**

| Flag | Description |
|------|-------------|
| `--headless` | Disable rendering for faster training |
| `--num_envs 4096` | Override number of parallel environments |
| `--resume` | Resume from last checkpoint |
| `--checkpoint 300` | Load specific iteration checkpoint |
| `--run_name my_run` | Tag this run |

Checkpoints are saved to:
```
logs/go2_secamp/<date>_<run_name>/model_<iter>.pt
```

### Train baseline AMP (single-skill velocity tracking)

```bash
python legged_gym/scripts/train.py --task=go2_amp --headless
```

---

## Evaluation

### Visualize in IsaacGym

```bash
python legged_gym/scripts/play.py --task=go2_secamp
```

### Deploy in MuJoCo (sim-to-sim)

```bash
cd depoly_mujoco
python deploy_mujoco.py
```

---

## Key Design Choices

### Skill-Aware Condition `c = [c^S, c^T]`

| Component | Type | Dimension | Description |
|-----------|------|-----------|-------------|
| `c^S` Skill Code | One-hot | 3 | Selects gait: `[trot, pace, canter]` |
| `c^T` Task Command | Continuous | 3 | Clamped displacement to waypoint (max 1 m) |

The **clamped** command formulation bounds the input space while preserving near-target distance information, enabling the policy to decelerate autonomously without external heuristics.

### Observation Spaces

| Element | Dim | Actor | Critic | Discriminator |
|---------|-----|-------|--------|---------------|
| Base Linear Velocity | 3 | | ✓ | ✓ |
| Base Angular Velocity | 3 | | ✓ | ✓ |
| Base Height | 1 | | | ✓ |
| Projected Gravity | 3 | ✓ | ✓ | |
| Clamped Task Command | 3 | ✓ | ✓ | |
| Joint Positions | 12 | ✓ | ✓ | ✓ |
| Joint Velocities | 12 | ✓ | ✓ | ✓ |
| Last Action | 12 | ✓ | ✓ | |
| Local Foot Positions | 12 | | | ✓ |
| Skill Code | 3 | ✓ | ✓ | |
| **Total** | | **45** | **51** | **43×2 + 3** |

### Reward Function

```
r_total = (1 - λ) · r^I  +  λ · r^T

r^T = min( ⟨v, d_w⟩,  v_max(skill) )     # task: progress toward waypoint
r^I = max( 0, 1 - 0.25·(D(τ) - 1)² )     # imitation: LSGAN-based AMP reward
```

### Network Architectures

| Network | Hidden Dims | Input | Output |
|---------|-------------|-------|--------|
| Actor `π_θ` | [1024, 512, 256] | 45D | 12D joint targets |
| Critic `V_ψ` | [1024, 512, 256] | 51D | scalar value |
| Discriminator `D_φ` | [1024, 512] | 89D (or projected) | scalar score |

---

## Domain Randomization

| Parameter | Range |
|-----------|-------|
| Ground Friction | [0.25, 1.75] |
| Added Base Mass | [-1.0, 1.0] kg |
| Velocity Perturbation | [-1.0, 1.0] m/s |
| Motor Gain Multiplier | [0.9, 1.1] |
| Motor Damping Multiplier | [0.9, 1.1] |

---

## Reference Motion Data

13 clips (~23 seconds total) retargeted from canine MoCap data to Go2 kinematics:

| Skill | Clips | Speed Range | Notes |
|-------|-------|-------------|-------|
| Trot | 3 | 0.86 – 2.03 m/s | |
| Pace | 3 | 0.60 – 1.14 m/s | |
| Canter | 3 (+3 mirrored) | 1.11 – 3.50 m/s | Symmetry augmented |
| Turn | 4 (L+R) | — | Mixed into all skills at 10% |

---

## Citation

If you use this codebase, please also cite the original AMP paper and legged_gym:

```bibtex
@inproceedings{escontrela2022adversarial,
  title={Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions},
  author={Escontrela, Alejandro and others},
  booktitle={IROS},
  year={2022}
}

@inproceedings{rudin2022learning,
  title={Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning},
  author={Rudin, Nikita and others},
  booktitle={CoRL},
  year={2022}
}
```

---

## Acknowledgements

Built on top of [AMP_for_Hardware](https://github.com/AlejandrEscontrela/AMP_for_Hardware) and [legged_gym](https://github.com/leggedrobotics/legged_gym). Reference motions retargeted from the MANN dataset.
