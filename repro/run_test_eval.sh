#!/bin/bash
# Evaluate the xs_1500 model on the HySpecNet-11k hard TEST split (lossless, delta=0)
# on GPU 4. Results land in Results/linerwkv/xs_1500/.
set -u

source ~/anaconda3/etc/profile.d/conda.sh
conda activate linerwkv

ROOT="<REPO_ROOT>"   # TODO: set to your local checkout of the linerwkv repo
TAG=xs_1500
OUT=$ROOT/Results/linerwkv/$TAG
CSV=$OUT/eval_results_${TAG}_d0_clip1.csv
LOG=$OUT/eval_run.log

mkdir -p "$OUT"
cd "$ROOT/repro" || exit 1

echo "=== START $(date '+%F %T')  tag=$TAG  gpu=4  split=test/hard  delta=0  clip=1 ===" | tee -a "$LOG"

python -u compress_eval.py \
    --config "$TAG" --tag "$TAG" \
    --difficulty hard --split test \
    --num 1140 --stride 1 --start 0 \
    --delta 0 --gpu 4 --clip 1 --resume 1 2>&1 | tee -a "$LOG"

echo "=== EVAL FINISHED $(date '+%F %T') ===" | tee -a "$LOG"

python summarize_eval.py --csv "$CSV" --summary "$OUT/metrics_summary.md" \
    --run "$(basename "$OUT") on hard/test, GPU4, $(date '+%F %T')" \
    >> "$LOG" 2>&1

echo "=== ALL DONE $(date '+%F %T') ===" | tee -a "$LOG"
