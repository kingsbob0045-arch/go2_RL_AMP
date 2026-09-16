#!/usr/bin/env bash
# Resume every SECAMP dataset from its latest checkpoint and train to convergence.
#
# Why this run lowers entropy_coef from 0.01 to 0.001
# ---------------------------------------------------
# ActorCritic stores the policy std as a raw parameter, so the Gaussian entropy is
# log(std) + const and its gradient w.r.t. the parameter is 1/std.  Adam normalises that
# gradient, which turns the entropy bonus into a near-constant *additive* increase of
# roughly one learning rate per update, independent of how large std already is.  The only
# opposing force is the surrogate loss, whose advantages are renormalised to unit variance
# every iteration, so as the task reward saturates the entropy term wins outright.
#
# The previous 2500-iteration sweep showed exactly that: std bottomed out near 0.45 around
# iteration 140 and then climbed monotonically, reaching the 4.0 guard rail on three of the
# four datasets.  Since ConvergenceMonitor requires std < 3.0 and a non-positive slope,
# those runs could never satisfy the criteria no matter how long they ran.
#
# A 120-iteration probe resuming dogml_gaits from std 2.196 measured:
#     entropy_coef   std slope/100it   reward   imi reward   episode length
#     0.01 (before)        +0.14        15.99        9.21              853
#     0.003                -0.154       18.63       10.82              922
#     0.001                -0.209       19.38       11.45              940
#     0.0                  -0.237       19.18       11.18              935
# 0.001 gives the best reward and imitation while still leaving a small exploration bonus,
# so it is the value used here.
set -u

cd /home/tumi6/Repo/ProjectsTest_Lingheng_Kong/go2_RL_AMP || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaacsim

MAX_ITERATIONS=${MAX_ITERATIONS:-6000}   # absolute target, not an extra amount
NUM_ENVS=${NUM_ENVS:-8192}
ENTROPY_COEF=${ENTROPY_COEF:-0.001}
TAG=${TAG:-v3}
MIN_ITERATIONS=${MIN_ITERATIONS:-3000}
RESUME_FROM=${RESUME_FROM:-v2}
DATASETS=${DATASETS:-"dogml_gaits mocap_gaits nju_agility kine2go_gaits"}

LOG_ROOT=logs/go2_secamp_isaaclab/dataset_compare

latest_checkpoint() {
    # Highest model_<n>.pt inside the newest run directory for this dataset.
    local run
    run=$(ls -d "${LOG_ROOT}/${1}_${RESUME_FROM}_"* 2>/dev/null | sort | tail -1)
    [ -n "$run" ] || return 1
    ls "$run"/model_*.pt 2>/dev/null |
        sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2
}

echo "===RUN_CONFIG=== envs=$NUM_ENVS target_iterations=$MAX_ITERATIONS entropy_coef=$ENTROPY_COEF tag=$TAG"
echo "===RUN_START=== $(date -Is)"

for dataset in $DATASETS; do
    checkpoint=$(latest_checkpoint "$dataset") || {
        echo "===SKIP_${dataset}=== no ${RESUME_FROM} run found"; continue; }
    echo "===START_${dataset}=== $(date -Is) resume=${checkpoint}"
    PYTHONUNBUFFERED=1 python isaaclab/scripts/train.py \
        --task Isaac-Go2-SECAMP-Direct-v0 \
        --amp_dataset "$dataset" \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$MAX_ITERATIONS" \
        --checkpoint "$checkpoint" \
        --entropy_coef "$ENTROPY_COEF" \
        --early_stop \
        --early_stop_min_iterations "$MIN_ITERATIONS" \
        --headless \
        --run_name "${dataset}_${TAG}" \
        > "/tmp/${dataset}_${TAG}.log" 2>&1
    echo "===DONE_${dataset}=== exit=$? $(date -Is)"
    grep -aE "^=== CONVERGED|convergence monitor" "/tmp/${dataset}_${TAG}.log" | tail -2
done

echo "===ALL_DATASETS_DONE=== $(date -Is)"
