# Go2 AMP：Isaac Gym → Isaac Lab 迁移版

本目录是独立的 Isaac Lab external project。旧的 `legged_gym/` 保留为结果对照，不再被这里的训练入口导入。结构遵循 NVIDIA 的 external project 规范：`source/` 放扩展包，`scripts/` 放训练、回放和导出入口。

## 已迁移任务

| 旧任务 | Isaac Lab Gym ID | Actor 观测 | Actor 输出 |
|---|---|---:|---:|
| `go2_amp` | `Isaac-Go2-AMP-Direct-v0` | 42 | 12 |
| `go2_secamp` | `Isaac-Go2-SECAMP-Direct-v0` | 45 | 12 |
| `go2_rough` | `Isaac-Go2-Rough-Residual-Direct-v0` | 5 × 42 = 210 | 12 residual + 3 skill |

迁移保留了 200 Hz PhysX / 50 Hz policy、显式 Go2 关节顺序、AMP/RSI、SECAMP skill 与 waypoint、rough history 与 frozen prior、地形 curriculum、domain randomization，以及原有自定义 AMP/SECAMP/two-head PPO runner。Gymnasium 兼容层只转换 runner 接口，仿真状态全部来自 Isaac Lab。

## 安装

先按照 [NVIDIA Isaac Lab 官方安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Sim 和 Isaac Lab。建议使用独立 conda/uv 环境，因为仓库内的 `rsl_rl` 是带 AMP/SECAMP 扩展的 fork。

```bash
# 当前目录：Lab_my_AMP_for_hardware_V1/isaaclab
python -m pip install -e source/go2_amp_isaaclab
python -m pip install -e ../rsl_rl
```

如果 Isaac Lab 不在当前 Python 环境，将下面的 `python` 换成 `/path/to/IsaacLab/isaaclab.sh -p`。官方 `UNITREE_GO2_CFG` 使用 NVIDIA Nucleus 上的 Go2 USD，首次运行需要下载该资源。motion dataset 和 frozen prior 都从旧项目根目录解析，不依赖启动工作目录。

## 分阶段验收

```bash
python scripts/list_envs.py

python scripts/zero_agent.py --task Isaac-Go2-AMP-Direct-v0 --num_envs 16 --steps 200
python scripts/zero_agent.py --task Isaac-Go2-SECAMP-Direct-v0 --num_envs 16 --steps 200
python scripts/zero_agent.py --task Isaac-Go2-Rough-Residual-Direct-v0 --num_envs 16 --steps 200

python scripts/train.py \
  --task Isaac-Go2-AMP-Direct-v0 \
  --num_envs 64 --max_iterations 2 \
  --amp_preload_transitions 1000 --headless
```

正式训练：

```bash
python scripts/train.py --task Isaac-Go2-AMP-Direct-v0 --headless
python scripts/train.py --task Isaac-Go2-SECAMP-Direct-v0 --headless
python scripts/train.py --task Isaac-Go2-Rough-Residual-Direct-v0 --headless
```

日志写入旧项目的 `logs/go2_*_isaaclab/`，不会覆盖 Isaac Gym 的旧实验。

## 回放和导出

```bash
python scripts/play.py \
  --task Isaac-Go2-AMP-Direct-v0 \
  --checkpoint ../logs/go2_amp_isaaclab/<run>/model_50000.pt \
  --num_envs 16 --export exports/go2_amp.pt
```

### Isaac Sim 中的 PS4 手柄回放（SECAMP）

手柄直接由 `play.py` 读取，不需要 ROS 2 `joy_node`、MuJoCo 或额外 Python 包。运行前确认系统存在
`/dev/input/js0`，然后使用一个环境启动可视化回放：

```bash
python scripts/play.py \
  --task Isaac-Go2-SECAMP-Direct-v0 \
  --checkpoint ../logs/go2_secamp_isaaclab/<run>/model_50000.pt \
  --num_envs 1 --mode joystick
```

映射与旧的 `deploy_secamp.py --mode joystick` 一致：按住 **L1+R1** 才启用左摇杆；
松开即停止；**X/O/△** 分别选择 pace/trot/canter；同时按下 **L2+R2** 结束回放。
`--mode waypoint` 使用旧部署脚本相同的 48 秒 pace → trot → canter 固定演示路线，
`--mode autonomous`（默认）保留环境自身随机 waypoint 行为。
可额外传入 `--steps N` 让回放在 N 个 policy step 后自动退出，便于检查启动流程。

导出同时生成 `go2_amp.pt` 和 `go2_amp.json`。JSON 包含任务类型、观测/动作维度、关节顺序和控制周期；ROS2 部署节点会在发布任何电机目标前校验这些字段。

旧 Isaac Gym checkpoint 不建议直接用于 Isaac Lab：即使网络结构相同，接触、关节驱动和资产参数也不同。应在 Isaac Lab 中重新训练并完成 Isaac Lab → MuJoCo → 真机的逐级验证。

## ROS2 / Unitree 部署

```bash
cd ../../Lab_my_Deploy_Go2Robot_V1
colcon build --symlink-install
source install/setup.bash

# 先启动原项目的 MuJoCo simulator 和 low_level_ctrl，再启动新策略节点
ros2 run deploy_rl_policy deploy_isaaclab.py \
  --config src/deploy_rl_policy/configs/go2_isaaclab.yaml \
  --is_simulation True
```

SECAMP 和 rough residual 分别使用：

```text
src/deploy_rl_policy/configs/go2_isaaclab_secamp.yaml
src/deploy_rl_policy/configs/go2_isaaclab_rough.yaml
```

确认 MuJoCo 中的站立姿态、四腿顺序、命令方向、动作幅值、50 Hz 策略频率和急停均正确后，才能改为 `--is_simulation False`。真机前仍需关闭 sport mode，并遵循原项目状态机和安全流程。

## 文件对应关系

```text
legged_gym/envs/go2*/         -> source/.../tasks/go2_amp/go2_env*.py
legged_gym/scripts/train.py   -> scripts/train.py
legged_gym/scripts/play*.py   -> scripts/play.py
rsl_rl custom runners        -> 原地保留，由 wrappers.py 适配 Gymnasium
deploy_rl_policy/scripts/*    -> deploy_isaaclab.py（新策略专用）
```
