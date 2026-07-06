#!/bin/bash
# run_experiment4_eos.sh — Experiment 4: PINN fine-tuned from INR (EOS only).
#
# Runs three PINN jobs in sequence, one per INR source experiment:
#   1. E1 + PINN EOS  — uniform split INR
#   2. E2 + PINN EOS  — contiguous split INR
#   3. E3 + PINN EOS  — sparse training INR (uses the 50% sparsity run)
#
# Each job:
#   - Reconstructs the exact same data split used by the source INR
#   - Warm-starts the PINN siren from the converged INR weights
#   - Fine-tunes with the TEOS-10 EOS physics loss (eq1_eos, weight=0.1)
#   - Produces a PDF report comparing INR vs PINN metrics
#
# USAGE:
#   1. Edit the INR_E1 / INR_E2 / INR_E3 paths below to point at your
#      actual best_model.pt files from E1 / E2B / E3.
#   2. Run (detached so the VM can be closed):
#        nohup bash scripts/run_experiment4_eos.sh > logs/experiment4.log 2>&1 &

set -euo pipefail

export NO_MLFLOW=1   # disable MLflow on cluster (no mlflow CLI needed)

# ── Paths to trained INR checkpoints (UPDATE THESE) ───────────────────────────
# Replace the placeholder paths with the actual timestamped folder names.

INR_E1="results/experiment1/REPLACE_WITH_E1_TIMESTAMP/best_model.pt"
INR_E2="results/experiment2b/REPLACE_WITH_E2B_TIMESTAMP/best_model.pt"
# For E3, pick the sparsity level you want to compare (e.g. 50%)
INR_E3="results/experiment3/REPLACE_WITH_E3_TIMESTAMP_sparsity50pct/best_model.pt"

# ── Common PINN training flags ────────────────────────────────────────────────
BASE_ARGS=(
    --pinn-epochs      1000
    --pinn-lr          1e-4
    --pinn-batch-size  32768
    --patience         100
    --min-delta        1e-6
    --eos-weight       0.1
    --n-colloc         8192
    --val-every        5
    --checkpoint-every 50
    --infer-batch      32768
    --num-workers      8
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

# ── Run all three source experiments ─────────────────────────────────────────
run_pinn "E1 + PINN EOS"  "$INR_E1"
run_pinn "E2B + PINN EOS" "$INR_E2"
run_pinn "E3 50pct + PINN EOS" "$INR_E3"

echo ""
echo "============================================================"
echo "  Experiment 4 complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
