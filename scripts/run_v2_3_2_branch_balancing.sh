#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
[[ -n "$STAGE" ]] || {
  echo "usage: $0 {index|dry-run|train}" >&2
  exit 2
}
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-/data/shouhanyu/envs/tiara2/bin/python}"
TORCHRUN="${TORCHRUN:-/data/shouhanyu/envs/tiara2/bin/torchrun}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

ROOT="${V232_ROOT:-/data/shouhanyu/Tiara2_v2_3_2}"
METADATA="${METADATA:-/data/shouhanyu/Tiara2_v2_3_1/metadata/hierarchical_labels_v2_3_1_deduplicated.tsv}"
TRAIN_READY="${TRAIN_READY:-/ssd/shouhanyu/Tiara2/corpus_ready_v2_1_3_bpbalanced}"
FEATURES="${FEATURES:-/ssd/shouhanyu/Tiara2_v2_3_1/cache/hierarchy_k7_composite}"
TAXDUMP_DIR="${TAXDUMP_DIR:-/data/shouhanyu/Tiara2/taxonomy}"
DONOR_ROOT="${DONOR_ROOT:-$ROOT/donor_index}"
DONOR_INDEX="${DONOR_INDEX:-$DONOR_ROOT/donor_index_manifest.json}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/models/v2.3.2}"
MAX_DONOR_SHARE="${MAX_DONOR_SHARE:-0.05}"

train_args=(
  --features "$FEATURES" --out "$MODEL_ROOT"
  --profile-version 2.3.2 --sampler hierarchical
  --max-euk-oversample 4 --donor-index "$DONOR_INDEX"
  --max-accession-share "$MAX_DONOR_SHARE"
  --max-species-share "$MAX_DONOR_SHARE"
  --max-genus-share "$MAX_DONOR_SHARE"
  --require-donor-caps
  --epochs 50 --batch 1024 --lr 0.001
  --hidden 2048,1024 --dropout 0.2 --amp
)

case "$STAGE" in
  index)
    "$PYTHON" -m tiara.hierarchical.donor_index_v232 \
      --metadata "$METADATA" --train-ready "$TRAIN_READY" \
      --features "$FEATURES" --taxdump-dir "$TAXDUMP_DIR" --out "$DONOR_ROOT"
    ;;
  dry-run)
    "$PYTHON" -m tiara.hierarchical.train "${train_args[@]}" --sampling-plan-only
    ;;
  train)
    "$TORCHRUN" --standalone --nproc_per_node=8 \
      -m tiara.hierarchical.train "${train_args[@]}"
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
