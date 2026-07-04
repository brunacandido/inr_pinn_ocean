#!/bin/bash
# run_experiment3.sh — Experiment 3: INR on sparse training data.
#
# Runs five sequential training jobs with increasing levels of sparsity.
# --train-depths-data-fraction controls the fraction of non-surface TRAINING
# observations kept; surface training obs and ALL val/test obs are always kept.
#
# The val/test split is identical to Experiment 2B (disjoint squares, 90/5/5,
# seed=42) so all runs are directly comparable with each other and with E2B.
#
#   sparsity | fraction kept | non-surf kept | total train obs | batch-size
#   ---------+---------------+---------------+-----------------+-----------
#      50%   |     0.50      |     ~3.95 M   |     ~4.15 M     |   65536
#      75%   |     0.25      |     ~1.97 M   |     ~2.17 M     |   32768
#      90%   |     0.10      |     ~0.79 M   |     ~0.99 M     |   16384
#      95%   |     0.05      |     ~0.40 M   |     ~0.60 M     |    8192
#      99%   |     0.01      |     ~0.08 M   |     ~0.28 M     |    4096
#
# Usage (run detached so the VM can be closed):
#   nohup bash scripts/run_experiment3.sh > logs/experiment3.log 2>&1 &

set -euo pipefail

# ── Common flags (identical across all sparsity levels) ───────────────────────
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
run_level() {
    local sparsity=$1   # label (e.g. "50%")
    local frac=$2       # --train-depths-data-fraction value
    local bs=$3         # batch size

    echo ""
    echo "============================================================"
    echo "  Experiment 3 — Sparsity : ${sparsity}"
    echo "  Non-surface training kept: ${frac}"
    echo "  Batch size               : ${bs}"
    echo "  Started                  : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    uv run scripts/experiment3.py "${BASE_ARGS[@]}" \
        --train-depths-data-fraction "$frac" \
        --batch-size                 "$bs"   \
        --infer-batch                "$bs"

    echo "  Finished : $(date '+%Y-%m-%d %H:%M:%S')"
}

# ── Make log directory ────────────────────────────────────────────────────────
mkdir -p logs

# ── Sparsity sweep ────────────────────────────────────────────────────────────
run_level "50%"  0.50  65536
run_level "75%"  0.25  32768
run_level "90%"  0.10  16384
run_level "95%"  0.05   8192
run_level "99%"  0.01   4096

echo ""
echo "============================================================"
echo "  Experiment 3 complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
