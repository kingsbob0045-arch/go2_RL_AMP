#!/usr/bin/env bash
# Resume every SECAMP dataset run from its 500-iteration checkpoint at 16384 environments.
set -u

cd /home/tumi6/Repo/ProjectsTest_Lingheng_Kong/go2_RL_AMP || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaacsim

COMPARE_DIR=logs/go2_secamp_isaaclab/dataset_compare
EXTRA_ITERATIONS=${EXTRA_ITERATIONS:-2000}
TAG=${TAG:-conv16384}

for dataset in dogml_gaits nju_agility mocap_gaits kine2go_gaits; do
    baseline=$(find "$COMPARE_DIR" -mindepth 1 -maxdepth 1 -type d \
        -name "${dataset}_full_8192env_20260901*" | sort | tail -1)
    checkpoint="$baseline/model_500.pt"
    if [ ! -f "$checkpoint" ]; then
        echo "===SKIP_${dataset}=== missing $checkpoint"
        continue
    fi

    echo "===START_${dataset}=== $(date -Is)"
    echo "resuming from $checkpoint for $EXTRA_ITERATIONS iterations"
    # Unbuffered so progress is visible through the tee pipe while the run is live.
    PYTHONUNBUFFERED=1 python isaaclab/scripts/train.py \
        --task Isaac-Go2-SECAMP-Direct-v0 \
        --amp_dataset "$dataset" \
        --checkpoint "$checkpoint" \
        --num_envs 16384 \
        --max_iterations "$EXTRA_ITERATIONS" \
        --headless \
        --run_name "${dataset}_${TAG}" \
        2>&1 | tee "/tmp/${dataset}_${TAG}.log"
    echo "===DONE_${dataset}=== $(date -Is)"
done

echo "===ALL_DATASETS_DONE=== $(date -Is)"
