#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
[[ "$STAGE" == "plan" || "$STAGE" == "train" || "$STAGE" == "compare" ]] || {
  echo "usage: $0 {plan|train|compare}" >&2
  exit 2
}

REPO="${REPO:-/home/shouhanyu/tiara2_pipeline}"
PYTHON="${PYTHON:-/data/shouhanyu/envs/tiara2/bin/python}"
TORCHRUN="${TORCHRUN:-/data/shouhanyu/envs/tiara2/bin/torchrun}"
FEATURES="${FEATURES:-/ssd/shouhanyu/Tiara2_v2_3_1/cache/hierarchy_k7_composite}"
DONOR_INDEX="${DONOR_INDEX:-/data/shouhanyu/Tiara2_v2_3_2/donor_index/donor_index_manifest.json}"
OUT="${OUT:-/ssd/shouhanyu/Tiara2_v2_4_1/audit/m6_k7_ceiling_v1/h4096_2048_seed42}"
CONTROL="${CONTROL:-$REPO/tiara/models/hierarchical-models-v2.3.2/hierarchical_model.pt}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

train_args=(
  "$REPO/v241b_residual/m6_k7_ceiling/train_k7_ceiling.py"
  --m6-ceiling --features "$FEATURES" --out "$OUT"
  --profile-version 2.3.2 --sampler hierarchical
  --max-euk-oversample 4 --donor-index "$DONOR_INDEX"
  --max-accession-share 0.05 --max-species-share 0.05 --max-genus-share 0.05
  --require-donor-caps --epochs 50 --batch 1024 --lr 0.001
  --hidden 4096,2048 --dropout 0.2 --seed 42 --amp
)

case "$STAGE" in
  plan)
    "$TORCHRUN" --standalone --nproc_per_node=8 "${train_args[@]}" --sampling-plan-only
    ;;
  train)
    "$TORCHRUN" --standalone --nproc_per_node=8 "${train_args[@]}"
    ;;
  compare)
    "$PYTHON" "$REPO/v241b_residual/m6_k7_ceiling/compare_m6.py" \
      --control "$CONTROL" --candidate "$OUT/hierarchical_model.pt" --out "$OUT/m6_comparison.json"
    ;;
esac
