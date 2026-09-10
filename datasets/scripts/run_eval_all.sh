#!/usr/bin/env bash
# Evaluate every dataset's final checkpoint with the deterministic (deployed) policy.
set -u

cd /home/tumi6/Repo/ProjectsTest_Lingheng_Kong/go2_RL_AMP || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaacsim

COMPARE_DIR=logs/go2_secamp_isaaclab/dataset_compare
OUT=${OUT:-$COMPARE_DIR/eval_deterministic.json}
STEPS=${STEPS:-600}
NUM_ENVS=${NUM_ENVS:-256}
RUN_GLOB=${RUN_GLOB:-conv16384}
MODEL=${MODEL:-model_2500.pt}

rm -f "$OUT"
for dataset in mocap_gaits kine2go_gaits nju_agility dogml_gaits; do
    run=$(find "$COMPARE_DIR" -mindepth 1 -maxdepth 1 -type d -name "${dataset}_${RUN_GLOB}*" | sort | tail -1)
    checkpoint="$run/$MODEL"
    if [ ! -f "$checkpoint" ]; then
        echo "===SKIP_${dataset}=== missing $checkpoint"
        continue
    fi
    echo "===EVAL_${dataset}=== $(date -Is)"
    PYTHONUNBUFFERED=1 python isaaclab/scripts/eval_checkpoint.py \
        --checkpoint "$checkpoint" \
        --amp_dataset "$dataset" \
        --num_envs "$NUM_ENVS" \
        --steps "$STEPS" \
        --json "$OUT" \
        --headless 2>&1 | grep -avE "^Loaded .* motion"
    echo "===EVAL_DONE_${dataset}=== $(date -Is)"
done

echo "===ALL_EVAL_DONE=== $(date -Is)"
echo "results: $OUT"
