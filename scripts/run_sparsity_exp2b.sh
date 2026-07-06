#!/bin/bash
# run_sparsity_exp2b.sh — Experiment 2B at 5 levels of depth-observation sparsity.
#
# --train-depths-data-fraction controls the fraction of non-surface TRAINING observations kept.
# Surface training obs and ALL val/test observations are always kept in full.
#
#   50% sparsity → keep 50% of non-surface training obs  (--train-depths-data-fraction 0.50)
#   75% sparsity → keep 25% of non-surface training obs  (--train-depths-data-fraction 0.25)
#   90% sparsity → keep 10% of non-surface training obs  (--train-depths-data-fraction 0.10)
#   95% sparsity → keep  5% of non-surface training obs  (--train-depths-data-fraction 0.05)
#   99% sparsity → keep  1% of non-surface training obs  (--train-depths-data-fraction 0.01)
#
# Experiments run sequentially on a single GPU.
# Each saves its own train.log, run_info.txt and split_map.png.
#
# Usage (run detached so the VM can be closed):
#   nohup bash scripts/run_sparsity_exp2b.sh > logs/sparsity_run.log 2>&1 &

set -euo pipefail

export NO_MLFLOW=1   # disable MLflow on cluster (no mlflow CLI needed)

# ── Common flags (identical across all runs) ──────────────────────────────────
# Val and test always use the full dataset — do NOT pass --train-depths-data-fraction here.
BASE_ARGS=(
    --train-fraction   0.90
    --val-fraction     0.05
    --n-val-squares    3
    --n-test-squares   3
    --seed             42
    --amp
    --val-every        5
    --checkpoint-every 50
    --epochs           2000
    --patience         100
    --num-workers      8
)

# ── Helper ────────────────────────────────────────────────────────────────────
run_exp() {
    local sparsity=$1   # label only (e.g. "50%")
    local tdf=$2        # --train-depths-data-fraction value (fraction of non-surface training obs KEPT)
    local bs=$3         # --batch-size / --infer-batch

    echo ""
    echo "============================================================"
    echo "  Sparsity        : ${sparsity}"
    echo "  Train fraction  : ${tdf} of training profiles kept"
    echo "  Batch size      : ${bs}"
    echo "  Started         : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    uv run scripts/experiment2b.py "${BASE_ARGS[@]}" \
        --train-depths-data-fraction "$tdf" \
        --batch-size    "$bs" \
        --infer-batch   "$bs"

    echo ""
    echo "  Finished : $(date '+%Y-%m-%d %H:%M:%S')"
}

# ── Make log directory ────────────────────────────────────────────────────────
mkdir -p logs

# ── Runs — batch size scaled to keep ~50 batches/epoch at each level ─────────
#
# Training obs = surface (all kept) + fraction of non-surface.
# Full training set: ~8.1M total (surface ~198K + non-surface ~7.9M).
#
#   sparsity | train-depths-data-fraction | non-surf kept | total train obs | batch-size
#   ---------+---------------+---------------+-----------------+-----------
#      50%   |     0.50      |     ~3.95 M   |     ~4.15 M     |   65536
#      75%   |     0.25      |     ~1.97 M   |     ~2.17 M     |   32768
#      90%   |     0.10      |     ~0.79 M   |     ~0.99 M     |   16384
#      95%   |     0.05      |     ~0.40 M   |     ~0.60 M     |    8192
#      99%   |     0.01      |     ~0.08 M   |     ~0.28 M     |    4096

run_exp "50%"  0.50  65536
run_exp "75%"  0.25  32768
run_exp "90%"  0.10  16384
run_exp "95%"  0.05   8192
run_exp "99%"  0.01   4096

echo ""
echo "============================================================"
echo "  All sparsity experiments completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
