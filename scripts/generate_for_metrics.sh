#!/usr/bin/env bash
# Generate balanced per-class images for FID/IS/Recall/F_β metric eval.
#
# Output layout:
#   <OUT_DIR>/class_000/images/0000.png ... 0249.png
#   <OUT_DIR>/class_001/images/0000.png ... 0249.png
#   ...
#
# Skips classes that already have N_SAMPLES_PER_CLASS images (idempotent —
# safe to re-run after a crash to fill in the gaps).
#
# Usage:
#   ./scripts/generate_for_metrics.sh \
#       <RFM_CHECKPOINT> <HAE_CHECKPOINT> <OUT_DIR> \
#       [N_SAMPLES_PER_CLASS=250] [NUM_CLASSES=200] \
#       [CFG_SCALE=1.5] [CURVATURE=-0.1]
#
# Default (50k = 250 × 200) matches CBDM / DiffROP eval; for ablations
# pass 50 (10k = 50 × 200).

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <RFM_CKPT> <HAE_CKPT> <OUT_DIR> [N_PER_CLASS] [NUM_CLASSES] [CFG_SCALE] [CURVATURE]" >&2
    exit 1
fi

RFM_CKPT="$1"
HAE_CKPT="$2"
OUT_DIR="$3"
N_PER_CLASS="${4:-250}"
NUM_CLASSES="${5:-200}"
CFG_SCALE="${6:-1.5}"
CURVATURE="${7:--0.1}"

# Locate the riemannian-fm directory relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RFM_DIR="$(realpath "$SCRIPT_DIR/../riemannian-fm")"

mkdir -p "$OUT_DIR"
cd "$RFM_DIR"

echo "================================================================"
echo "Generating for metrics:"
echo "  RFM checkpoint:    $RFM_CKPT"
echo "  HAE checkpoint:    $HAE_CKPT"
echo "  Output dir:        $OUT_DIR"
echo "  Samples per class: $N_PER_CLASS"
echo "  Num classes:       $NUM_CLASSES"
echo "  CFG scale:         $CFG_SCALE"
echo "  Curvature:         $CURVATURE"
echo "  Total images:      $((N_PER_CLASS * NUM_CLASSES))"
echo "================================================================"

START_TS=$(date +%s)
COMPLETED=0
SKIPPED=0

for ((c=0; c<NUM_CLASSES; c++)); do
    CLASS_DIR=$(printf "%s/class_%03d" "$OUT_DIR" "$c")
    IMG_DIR="$CLASS_DIR/images"

    # Skip if this class already has the expected number of images.
    if [[ -d "$IMG_DIR" ]]; then
        n_existing=$(find "$IMG_DIR" -maxdepth 1 -name '*.png' | wc -l)
        if [[ "$n_existing" -ge "$N_PER_CLASS" ]]; then
            SKIPPED=$((SKIPPED + 1))
            if (( c % 20 == 0 )); then
                echo "[skip] class $c — already has $n_existing images"
            fi
            continue
        fi
    fi

    python generate_cifar.py \
        --rfm_checkpoint "$RFM_CKPT" \
        --hae_checkpoint "$HAE_CKPT" \
        --n_samples "$N_PER_CLASS" \
        --curvature "$CURVATURE" \
        --class_id "$c" --cfg_scale "$CFG_SCALE" \
        --output_dir "$CLASS_DIR" \
        > "$CLASS_DIR.log" 2>&1
    COMPLETED=$((COMPLETED + 1))

    # Progress + ETA every 5 classes.
    if (( c % 5 == 4 )); then
        ELAPSED=$(( $(date +%s) - START_TS ))
        DONE=$((c + 1))
        AVG=$(echo "scale=2; $ELAPSED / $DONE" | bc)
        REMAINING=$(echo "scale=0; ($NUM_CLASSES - $DONE) * $AVG / 1" | bc)
        echo "[progress] $DONE/$NUM_CLASSES classes  (${ELAPSED}s elapsed, ~${REMAINING}s remaining, avg ${AVG}s/class)"
    fi
done

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "================================================================"
echo "Done. $COMPLETED generated, $SKIPPED skipped, ${ELAPSED}s total."
echo "Output: $OUT_DIR"
echo "================================================================"
