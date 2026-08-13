#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:?usage: run_v250.sh prepare|base-logits|smoke|train}"
REPO="${REPO:-/home/shouhanyu/tiara2_pipeline}"
PY="${PY:-/data/shouhanyu/envs/tiara2/bin/python}"
TORCHRUN="${TORCHRUN:-/data/shouhanyu/envs/tiara2/bin/torchrun}"
PATCH="$REPO/v250_rc_cnn"
ROOT="${V250_ROOT:-/ssd/shouhanyu/Tiara2_v2_5_0}"
SHORT=/data/shouhanyu/Tiara2_v2_4_0_short_expert/features_1000_2499
export PYTHONPATH="$PATCH:$REPO${PYTHONPATH:+:$PYTHONPATH}"
case "$STAGE" in
 prepare) "$PY" "$PATCH/prepare_cache.py" --train-ready /ssd/shouhanyu/Tiara2/corpus_ready_v2_1_3_bpbalanced --short-features "$SHORT" --out "$ROOT/sequence_cache_v1" ;;
 base-logits) "$TORCHRUN" --standalone --nproc_per_node=8 "$PATCH/cache_base_logits.py" --short-features "$SHORT" --checkpoint /data/shouhanyu/Tiara2_v2_3_2/models/v2.3.2/hierarchical_model.pt --out "$ROOT/base_logits_v1" ;;
 smoke) "$TORCHRUN" --standalone --nproc_per_node=8 "$PATCH/train_rc_cnn.py" --cache "$ROOT/sequence_cache_v1" --base-logits "$ROOT/base_logits_v1" --short-features "$SHORT" --out "$ROOT/smoke_seed42" --epochs 1 --batch 48 --workers 2 --smoke-steps 5 ;;
 train) "$TORCHRUN" --standalone --nproc_per_node=8 "$PATCH/train_rc_cnn.py" --cache "$ROOT/sequence_cache_v1" --base-logits "$ROOT/base_logits_v1" --short-features "$SHORT" --out "$ROOT/rc_cnn_seed42" --epochs 50 --batch 96 --workers 4 --seed 42 ;;
 *) echo "unknown stage $STAGE" >&2; exit 2;;
esac
