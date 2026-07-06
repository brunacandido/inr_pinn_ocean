#!/bin/bash
# run_experiment4_sparsity.sh — Experiment 4 (PINN EOS) for each E3 sparsity level.
#
# Runs from highest sparsity to lowest (99% → 50%).
# Comment out any run_pinn line you don't want to execute.
#
# Usage:
#   1. Fill in the INR_E3_* paths below (best_model.pt from each E3 sparsity run).
#   2. Run detached:
#        nohup bash scripts/run_experiment4_sparsity.sh > logs/experiment4_sparsity.log 2>&1 &

set -euo pipefail

export NO_MLFLOW=1

# ── Paths to E3 checkpoints (UPDATE THESE) ────────────────────────────────────
INR_E3_99="results/experiment3/20260703_164037_sparsity99pct/best_model.pt"
INR_E3_95="results/experiment3/20260703_160708_sparsity95pct/best_model.pt"
INR_E3_90="results/experiment3/20260703_152821_sparsity90pct/best_model.pt"
INR_E3_75="results/experiment3/20260703_142816_sparsity75pct/best_model.pt"
INR_E3_50="results/experiment3/20260703_124420_sparsity50pct/best_model.pt"

# ── Common PINN flags ─────────────────────────────────────────────────────────
BASE_ARGS=(
    --pinn-epochs      1000
    --pinn-lr          1e-4
    --pinn-batch-size  8192
    --patience         100
    --min-delta        1e-6
    --eos-weight       0.1
    --n-colloc         8192
    --val-every        5
    --checkpoint-every 50
    --infer-batch      32768
    --amp
)

# ── Helper ────────────────────────────────────────────────────────────────────
run_pinn() {
    local label=$1
    local ckpt=$2

    if [ ! -f "$ckpt" ]; then
        echo ""
        echo "============================================================"
        echo "  SKIPPING $label — checkpoint not found:"
        echo "  $ckpt"
        echo "============================================================"
        return
    fi

    echo ""
    echo "============================================================"
    echo "  Experiment 4 — ${label}"
    echo "  INR checkpoint : ${ckpt}"
    echo "  Started        : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    uv run scripts/experiment4.py \
        --inr-checkpoint "$ckpt" \
        "${BASE_ARGS[@]}"

    echo "  Finished : $(date '+%Y-%m-%d %H:%M:%S')"
}

# ── Make log directory ────────────────────────────────────────────────────────
mkdir -p logs

# ── Runs — comment out any line to skip that sparsity level ──────────────────
run_pinn "E3 99% sparsity + PINN EOS" "$INR_E3_99"
run_pinn "E3 95% sparsity + PINN EOS" "$INR_E3_95"
run_pinn "E3 90% sparsity + PINN EOS" "$INR_E3_90"
run_pinn "E3 75% sparsity + PINN EOS" "$INR_E3_75"
run_pinn "E3 50% sparsity + PINN EOS" "$INR_E3_50"

echo ""
echo "============================================================"
echo "  All sparsity PINN runs complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
