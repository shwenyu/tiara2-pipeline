#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
[[ -n "$STAGE" ]] || { echo "usage: $0 {smoke-crops|crops|tfidf|features|base-logits|train}" >&2; exit 2; }
REPO="${REPO:-/home/shouhanyu/tiara2_pipeline}"
PATCH_ROOT="${PATCH_ROOT:-$REPO/v241b_residual}"
PYTHON="${PYTHON:-/data/shouhanyu/envs/tiara2/bin/python}"
SOURCE="${SOURCE:-/ssd/shouhanyu/Tiara2/corpus_ready_v2_1_3_bpbalanced}"
ROOT="${V241_ROOT:-/ssd/shouhanyu/Tiara2_v2_4_1}"
CROPS="$ROOT/crops_continuous_v1"
TFIDF="$ROOT/tfidf_multiscale_v1"
FEATURES="$ROOT/features_continuous_multiscale_v1"
BASE_TFIDF="$REPO/tiara/models/tfidf-models-v2.2.0/k7-first-stage"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/data/shouhanyu/Tiara2_v2_3_2/models/v2.3.2/hierarchical_model.pt}"
LABELS="${LABELS:-/ssd/shouhanyu/Tiara2_v2_3_1/cache/hierarchy_k7_composite}"
BASE_LOGITS="$ROOT/base_logits_continuous_v1"
MODEL_OUT="$ROOT/residual_adapter_v1"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

case "$STAGE" in
  smoke-crops)
    "$PYTHON" "$PATCH_ROOT/tools/prepare_continuous_crops.py" \
      --source "$SOURCE" --out "$ROOT/smoke/crops_continuous_v1" \
      --seed 24101 --focus-fraction 0.5 --smoke-limit 200 --force
    ;;
  crops)
    "$PYTHON" "$PATCH_ROOT/tools/prepare_continuous_crops.py" \
      --source "$SOURCE" --out "$CROPS" \
      --seed 24101 --focus-fraction 0.5
    ;;
  tfidf)
    "$PYTHON" -m tiara.training.train_tfidf_optimized \
      "$CROPS/train" "$TFIDF" \
      --workers 56 --batch-size 2000 --fragment-len 2499 \
      --k-first 4 5 6 --k-second 4
    ;;
  features)
    "$PYTHON" "$PATCH_ROOT/tools/build_crop_features.py" \
      --crops "$CROPS" --multiscale-tfidf "$TFIDF" \
      --base-tfidf "$BASE_TFIDF" --out "$FEATURES" \
      --workers 56 --chunk 2000
    ;;
  base-logits)
    "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
      "$PATCH_ROOT/tools/build_base_logits.py" \
      --features "$FEATURES" --checkpoint "$BASE_CHECKPOINT" \
      --labels "$LABELS" --out "$BASE_LOGITS" --batch-size 2048
    ;;
  train)
    "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
      "$PATCH_ROOT/tools/train_residual.py" \
      --features "$FEATURES" --base-logits "$BASE_LOGITS" \
      --labels "$LABELS" --out "$MODEL_OUT" \
      --epochs 12 --batch-size 512 --workers 4 --lr 3e-4
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
