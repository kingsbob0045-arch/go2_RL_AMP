#!/usr/bin/env bash
# Train one SECAMP dataset as a sim-to-real curriculum: four stages, each warm-started from
# the previous one's checkpoint, each gated on the previous one producing a usable policy.
#
# Why staged
# ----------
# The v4 run added nine difficulty changes at once (terrain, action latency, observation
# latency, IMU bias, wider PD randomisation, per-link mass, per-link COM, joint
# friction/armature, three penalty terms).  Reward fell from 21.7 to -10.9 and 20000
# iterations produced exactly one bit of information: it broke.  A stage that fails here names
# the change that caused it, and the chain stops instead of burning the remaining hours.
#
# What changed before this script
# -------------------------------
#   * calf effort_limit 35.55 -> 45.43 Nm.  35.55 is the Go1 knee figure; the "91% calf
#     saturation" that motivated a bespoke penalty term was measured against a ceiling 22%
#     too low.  That bespoke term is gone; the four legged_gym terms replace it.
#   * reference_state_initialization_prob 1.0 -> 0.5.  At 1.0 every episode started mid-clip,
#     so the standing default pose -- the state low_level_ctrl hands the policy on the robot --
#     was the one initial state the policy never saw.
#   * The reference clip now comes from the pool matching the episode's skill, and the first
#     waypoint points along that clip's actual travel direction.  Previously the skill, the
#     clip and the heading were three independent draws, so the discriminator and the task
#     reward routinely asked for different motions at the same time.
#   * Total control latency cut from a worst case of five control periods (100 ms) to one
#     (20 ms), and the observation path left undelayed instead of double-counting the loop.
#
# Usage
# -----
#     datasets/scripts/run_curriculum_v5.sh dogml_gaits
#     NUM_ENVS=2048 datasets/scripts/run_curriculum_v5.sh nju_agility
#
# Runs unattended; roughly 9 hours per dataset at 4096 environments on one RTX 5090.
set -u

DATASET="${1:-dogml_gaits}"
TAG="${TAG:-v5}"
NUM_ENVS="${NUM_ENVS:-4096}"
SEED="${SEED:-77}"

W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) || exit 1
cd "$W" || exit 1
. ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
export PYTHONUNBUFFERED=1
export PYTHONPATH="$W/isaaclab/source/go2_amp_isaaclab:$W/rsl_rl"

RUNS="$W/logs/go2_secamp_isaaclab/dataset_compare"
EXPORTS="$W/exports/$TAG"
mkdir -p "$EXPORTS"

# stage:cumulative_iterations:episode_length_floor
# "core" starts from scratch so it gets 3000; the rest are warm starts and get 2000 each.
# The floors are deliberately loose -- they are there to stop a chain that has clearly
# collapsed, not to judge the policy.  Read the numbers yourself afterwards.
STAGES=(
    "core:3000:700"
    "latency:5000:600"
    "dynamics:7000:600"
    "terrain:9000:500"
)

echo "===CURRICULUM=== dataset=$DATASET tag=$TAG envs=$NUM_ENVS seed=$SEED start=$(date -Is)"

checkpoint=""
for entry in "${STAGES[@]}"; do
    IFS=: read -r stage total floor <<< "$entry"
    run_name="${DATASET}_${TAG}_${stage}"
    echo "===STAGE_BEGIN=== $stage total_iterations=$total floor=$floor at=$(date -Is)"

    args=(
        --task Isaac-Go2-SECAMP-Direct-v0
        --amp_dataset "$DATASET"
        --num_envs "$NUM_ENVS"
        --max_iterations "$total"
        --seed "$SEED"
        --stage "$stage"
        --headless
        --run_name "$run_name"
    )
    [ -n "$checkpoint" ] && args+=(--checkpoint "$checkpoint")
    # Only the first stage has headroom above the 2000-iteration floor, so it is the only one
    # where stopping early can save anything.  The later stages are exactly 2000 long.
    [ "$stage" = core ] && args+=(--early_stop --early_stop_min_iterations 2000)

    python isaaclab/scripts/train.py "${args[@]}"
    status=$?
    echo "===STAGE_TRAIN=== $stage exit=$status at=$(date -Is)"
    [ $status -ne 0 ] && { echo "===ABORT=== $stage training failed"; exit 1; }

    run_dir=$(ls -dt "$RUNS"/${run_name}_* 2>/dev/null | head -1)
    checkpoint="${run_dir}/model_${total}.pt"
    [ -f "$checkpoint" ] || { echo "===ABORT=== $stage produced no ${checkpoint}"; exit 1; }

    # Gate.  check_convergence.evaluate is the same report used to read the v3/v4 runs, so the
    # numbers here are directly comparable to those.
    python - "$run_dir" "$floor" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "datasets/scripts")
from check_convergence import evaluate

report = evaluate(Path(sys.argv[1]))
floor = float(sys.argv[2])
metrics = report.get("metrics", {})
length = metrics.get("episode_length", 0.0)
share = metrics.get("timeout_share", -1.0)
ended = "not logged" if share < 0 else f"{share:.0%} time limit, {1 - share:.0%} fall"
print(f"===GATE=== status={report['status']} ep_length={length} floor={floor} "
      f"ended_by='{ended}' reward={metrics.get('reward')} "
      f"imi_of_ceiling={metrics.get('imitation_fraction_of_ceiling')} "
      f"d_expert={metrics.get('d_expert')} noise_std={metrics.get('noise_std')} "
      f"value_loss={metrics.get('value_loss')}")
height = metrics.get("base_height", -1.0)
print(f"===GATE_POSTURE=== height={height} pitch={metrics.get('gravity_x')} "
      f"roll={metrics.get('gravity_y')} feet_down={metrics.get('foot_contacts')} "
      f"stance={metrics.get('stance_width')}")
for reason in report.get("divergence_reasons", []):
    print(f"===GATE_DIVERGED=== {reason}")
# A crouched shuffle survives every scalar the v4 gate looked at, so the height is part of the
# gate now.  0.22 m is well below the Go2's 0.33 m nominal stance: it rejects a collapse, not
# a policy that merely rides a little low.  -1.0 means the run predates the scalar; skip it.
crouched = 0.0 <= height < 0.22
if crouched:
    print(f"===GATE_CROUCHED=== base height {height} m below 0.22 m")
sys.exit(0 if report["status"] != "DIVERGED" and length >= floor and not crouched else 2)
PY
    gate=$?
    echo "===STAGE_GATE=== $stage exit=$gate at=$(date -Is)"
    if [ $gate -ne 0 ]; then
        echo "===ABORT=== stage '$stage' is the change that broke it; chain stopped."
        echo "  Its checkpoint is kept at $checkpoint, and the previous stage's policy is the"
        echo "  last one that passed.  Tune this stage alone rather than continuing."
        exit 2
    fi
done

# Export the final stage for MuJoCo.  play.py --export writes the sidecar metadata the deploy
# node checks; a raw checkpoint has none and is rejected.
out="${EXPORTS}/${DATASET}_${TAG}.pt"
python isaaclab/scripts/play.py \
    --task Isaac-Go2-SECAMP-Direct-v0 \
    --checkpoint "$checkpoint" \
    --num_envs 1 --steps 1 --headless \
    --export "$out"
echo "===EXPORT=== exit=$? path=$out"
echo "===CURRICULUM_DONE=== dataset=$DATASET at=$(date -Is)"
