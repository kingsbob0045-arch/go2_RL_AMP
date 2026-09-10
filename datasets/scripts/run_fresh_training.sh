#!/usr/bin/env bash
# Train every SECAMP dataset from scratch, stopping each run as soon as the convergence
# criteria hold.
#
# Fixes this run depends on (all found by diffing against
# original_AMP_for_hardware_fromTianci, which is the legged_gym implementation this port
# reproduces):
#   * actions clipped at +/-100 instead of +/-1     go2_env{,_cfg}.py
#     Clipping at the action range made the environment blind to any sampled action beyond
#     it, so nothing opposed the entropy bonus and the policy std ran away (1 -> 38).
#   * policy std ceiling raised to a runaway guard  rsl_rl/runners/{go2_secamp,amp_on_policy}_runner.py
#   * init_at_random_ep_len actually reaches the env  wrappers.py
#     The runner assigned episode_length_buf onto the wrapper, so every environment used to
#     reset in lockstep every 1000 steps.
#   * infos['time_outs'] published for PPO bootstrapping  wrappers.py
#   * only_positive_rewards on the task reward       go2_env{,_cfg}.py
#   * IdealPD actuators instead of Isaac Lab's DC motor  go2_env_cfg.py
#     The DC motor halves available torque at the joint speeds a 3 m/s canter needs.
#   * reference state initialization on every reset  go2_env_cfg.py
#   * waypoint lookahead, removing the per-waypoint deceleration incentive  go2_env.py
#   * NaN-free quaternion slerp                     rsl_rl/utils/utils.py
#   * vectorised motion lookup                      rsl_rl/datasets/motion_loader.py
#   * larger PhysX buffers                          go2_env_cfg.py (needed above ~8k environments)
#
# 8192 environments, not 16384: PPO progress is driven by the number of gradient updates,
# and both settings do the same 5 x 4 = 20 updates per iteration.  Doubling the environments
# only doubles the cost of each update while the minibatch is already far past saturation.
set -u

cd /home/tumi6/Repo/ProjectsTest_Lingheng_Kong/go2_RL_AMP || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaacsim

MAX_ITERATIONS=${MAX_ITERATIONS:-2500}
NUM_ENVS=${NUM_ENVS:-8192}
TAG=${TAG:-v2}
MIN_ITERATIONS=${MIN_ITERATIONS:-500}
DATASETS=${DATASETS:-"dogml_gaits mocap_gaits nju_agility kine2go_gaits"}

echo "===RUN_CONFIG=== envs=$NUM_ENVS max_iterations=$MAX_ITERATIONS tag=$TAG"
echo "===RUN_START=== $(date -Is)"

for dataset in $DATASETS; do
    echo "===START_${dataset}=== $(date -Is)"
    PYTHONUNBUFFERED=1 python isaaclab/scripts/train.py \
        --task Isaac-Go2-SECAMP-Direct-v0 \
        --amp_dataset "$dataset" \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$MAX_ITERATIONS" \
        --early_stop \
        --early_stop_min_iterations "$MIN_ITERATIONS" \
        --headless \
        --run_name "${dataset}_${TAG}" \
        > "/tmp/${dataset}_${TAG}.log" 2>&1
    echo "===DONE_${dataset}=== exit=$? $(date -Is)"
    grep -aE "^=== CONVERGED|convergence monitor" "/tmp/${dataset}_${TAG}.log" | tail -2
done

echo "===ALL_DATASETS_DONE=== $(date -Is)"
