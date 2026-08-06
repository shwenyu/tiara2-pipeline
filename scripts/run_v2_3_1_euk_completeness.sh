#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
[[ -n "$STAGE" ]] || {
  echo "usage: $0 {freeze|audit|delta|prepare|train|evaluate|gate|publish}" >&2
  exit 2
}
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

BASE_CORPUS="${BASE_CORPUS:-/ssd/shouhanyu/Tiara2/corpus_ready_v2_1_3_bpbalanced}"
BASE_METADATA="${BASE_METADATA:-/data/shouhanyu/Tiara2_v2_3_0/metadata/hierarchical_labels.tsv}"
BASE_FEATURES="${BASE_FEATURES:-/ssd/shouhanyu/Tiara2_v2_3_0/cache/hierarchy_k7}"
TFIDF="${TFIDF:-$REPO/tiara/models/tfidf-models-v2.2.0/k7-first-stage}"
ROOT="${V231_ROOT:-/data/shouhanyu/Tiara2_v2_3_1}"
SSD_ROOT="${V231_SSD_ROOT:-/ssd/shouhanyu/Tiara2_v2_3_1}"
AUDIT_DIR="${AUDIT_DIR:-$ROOT/audit}"
FREEZE_MANIFEST="${FREEZE_MANIFEST:-$ROOT/manifests/base_freeze_manifest.json}"
CURATED_METADATA="${CURATED_METADATA:-$AUDIT_DIR/hierarchical_labels_v2_3_1.tsv}"
AUDIT_REPORT="${AUDIT_REPORT:-$AUDIT_DIR/euk_completeness_report.json}"
DELTA_ROOT="${DELTA_ROOT:-$ROOT/euk_delta}"
DELTA_METADATA="${DELTA_METADATA:-$DELTA_ROOT/euk_delta_metadata.tsv}"
DELTA_MANIFEST="${DELTA_MANIFEST:-$DELTA_ROOT/euk_delta_manifest.json}"
COMPOSITE="${COMPOSITE:-$SSD_ROOT/cache/hierarchy_k7_composite}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/models/v2.3.1}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/evaluation}"
CONFIG="${CONFIG:-$REPO/config/config_v2_3_1_euk_completeness.yaml}"

need() {
  [[ -n "${!1:-}" ]] || {
    echo "missing required environment variable: $1" >&2
    exit 2
  }
}

case "$STAGE" in
  freeze)
    mkdir -p "$(dirname "$FREEZE_MANIFEST")"
    "$PYTHON" -m tiara.hierarchical freeze-base \
      --train-ready "$BASE_CORPUS" --tfidf "$TFIDF" \
      --metadata-tsv "$BASE_METADATA" --out "$FREEZE_MANIFEST"
    ;;
  audit)
    need SOURCE_INDEX
    need POOL_METADATA
    args=(--base-metadata "$BASE_METADATA" --out "$AUDIT_DIR"
          --source-index "$SOURCE_INDEX" --pool-metadata "$POOL_METADATA")
    [[ -n "${TAXDUMP_DIR:-}" ]] && args+=(--taxdump-dir "$TAXDUMP_DIR")
    [[ -n "${OVERRIDES_TSV:-}" ]] && args+=(--overrides-tsv "$OVERRIDES_TSV")
    "$PYTHON" -m tiara.hierarchical audit-euk "${args[@]}"
    ;;
  delta)
    need CANDIDATE_ROOT
    need CANDIDATE_METADATA
    "$PYTHON" -m tiara.hierarchical build-euk-delta \
      --base-train-ready "$BASE_CORPUS" --base-metadata "$CURATED_METADATA" \
      --candidate-root "$CANDIDATE_ROOT" --candidate-metadata "$CANDIDATE_METADATA" \
      --audit-report "$AUDIT_REPORT" --base-freeze-manifest "$FREEZE_MANIFEST" \
      --verify-base full --out "$DELTA_ROOT"
    ;;
  prepare)
    "$PYTHON" -m tiara.hierarchical prepare-v231 \
      --base-features "$BASE_FEATURES" --base-train-ready "$BASE_CORPUS" \
      --curated-base-metadata "$CURATED_METADATA" \
      --delta-root "$DELTA_ROOT" --delta-metadata "$DELTA_METADATA" \
      --tfidf "$TFIDF" --base-freeze-manifest "$FREEZE_MANIFEST" \
      --verify-base full --out "$COMPOSITE"
    ;;
  train)
    "$PYTHON" -m tiara.hierarchical train \
      --features "$COMPOSITE" --out "$MODEL_ROOT" \
      --epochs 50 --batch 1024 --lr 0.001 --hidden 2048,1024 --dropout 0.2
    ;;
  evaluate)
    need TRUTH_TSV
    need PREDICTIONS_TSV
    mkdir -p "$EVAL_ROOT"
    "$PYTHON" -m tiara.hierarchical evaluate-v231 \
      --truth "$TRUTH_TSV" --predictions "$PREDICTIONS_TSV" \
      --out "$EVAL_ROOT/evaluation.json"
    ;;
  gate)
    need BASELINE_BENCHMARK
    need CURRENT_BENCHMARK
    "$PYTHON" -m tiara.hierarchical gate-v231 \
      --completeness-report "$AUDIT_REPORT" \
      --composite-features "$COMPOSITE/composite_features.json" \
      --evaluation "$EVAL_ROOT/evaluation.json" \
      --base-metadata "$CURATED_METADATA" --delta-metadata "$DELTA_METADATA" \
      --baseline-benchmark "$BASELINE_BENCHMARK" \
      --current-benchmark "$CURRENT_BENCHMARK" \
      --checkpoint "$MODEL_ROOT/hierarchical_model.pt" \
      --max-legacy-regression-pp 0.20 --min-leaf-recall 0.05 \
      --out "$EVAL_ROOT/acceptance_gates.json"
    ;;
  publish)
    DESTINATION="${DESTINATION:-/data/shouhanyu/Tiara2/releases/v2.3.1}"
    CURRENT_LINK="${CURRENT_LINK:-/data/shouhanyu/Tiara2/releases/current}"
    "$PYTHON" -m tiara.hierarchical publish-v231 \
      --checkpoint "$MODEL_ROOT/hierarchical_model.pt" \
      --training-history "$MODEL_ROOT/training_history.json" --config "$CONFIG" \
      --base-freeze-manifest "$FREEZE_MANIFEST" \
      --completeness-report "$AUDIT_REPORT" --delta-manifest "$DELTA_MANIFEST" \
      --composite-features "$COMPOSITE/composite_features.json" \
      --evaluation "$EVAL_ROOT/evaluation.json" --gates "$EVAL_ROOT/acceptance_gates.json" \
      --destination "$DESTINATION" --current-link "$CURRENT_LINK"
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
