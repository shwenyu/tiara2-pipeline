#!/usr/bin/env bash
# Launch the fragment bp-balancing checkpoint (regroup -> train).
#
# WHAT CHANGED IN v2.1.3
# ----------------------
# This wrapper used to build the python command line itself, with its own
# hard-coded corpus paths and its own WORKERS=16 (config said 24). That drift
# is exactly how v2.1.2 ended up balancing only the train split while
# config/config.yaml looked correct. The wrapper now does NOTHING but
# pre-flight checks and hands over to:
#
#     python3 -m tiara2.cli bp-balance --config <config>
#
# Every path, split, worker count and prior comes from ONE resolved config.
# Override with the config itself, or with `--set a.b=c`, never by editing
# flags here.
#
# Usage:
#   scripts/run_sample_fragments_v2_1_2.sh                # normal run
#   FORCE=1 scripts/run_sample_fragments_v2_1_2.sh        # rebuild from scratch
#   DRY_RUN=1 scripts/run_sample_fragments_v2_1_2.sh      # print the command
#   CONFIG=config/other.yaml scripts/run_sample_fragments_v2_1_2.sh
set -euo pipefail

REPO=${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
CONFIG=${CONFIG:-$REPO/config/config.yaml}
PYTHON=${PYTHON:-python3}
cd "$REPO"

if [[ ! -f $CONFIG ]]; then
  echo "[FATAL] config not found: $CONFIG" >&2
  exit 2
fi

# --- read the resolved config; never re-derive paths by hand ---------------
read_cfg() {
  "$PYTHON" - "$CONFIG" "$1" <<'PY'
import sys
sys.path.insert(0, ".")
from tiara2 import config as C
cfg = C.load(sys.argv[1])
node = cfg
for part in sys.argv[2].split("."):
    node = node.get(part, "") if isinstance(node, dict) else ""
if isinstance(node, list):
    node = ",".join(str(x) for x in node)
print(node)
PY
}

SOURCE=$(read_cfg fragment_bp_balance.source_root)
OUTPUT=$(read_cfg fragment_bp_balance.output_root)
SPLITS=$(read_cfg fragment_bp_balance.balance_splits)
CLASSES=$(read_cfg classes)

if [[ -z $SOURCE || -z $OUTPUT ]]; then
  echo "[FATAL] fragment_bp_balance.source_root/output_root are empty in $CONFIG" >&2
  exit 2
fi

echo "[pre] config : $CONFIG"
echo "[pre] source : $SOURCE"
echo "[pre] output : $OUTPUT"
echo "[pre] splits : $SPLITS"

# --- 1) every balanced split must have all five class FASTAs ---------------
# `-e` follows symlinks, so a BROKEN symlink fails here instead of silently
# producing an empty class three hours into the run.
missing=0
IFS=',' read -r -a _splits <<< "$SPLITS"
IFS=',' read -r -a _classes <<< "$CLASSES"
for split in "${_splits[@]}"; do
  for cls in "${_classes[@]}"; do
    f="$SOURCE/$split/${cls}.fasta"
    if [[ ! -e $f ]]; then
      echo "[FATAL] missing (or broken symlink): $f" >&2
      missing=1
    fi
  done
done
[[ $missing -eq 0 ]] || exit 2

# --- 2) stale-corpus guard --------------------------------------------------
# The v2_1_breadth corpus must never feed a v2.1.3 run. Check the resolved
# target of every balanced split, not just train/eukarya.
for split in "${_splits[@]}"; do
  for cls in "${_classes[@]}"; do
    target=$(readlink -f "$SOURCE/$split/${cls}.fasta" || true)
    if [[ $target == *v2_1_breadth* ]]; then
      echo "[FATAL] $split/${cls}.fasta resolves into the OLD corpus: $target" >&2
      exit 2
    fi
  done
done

# --- 3) output must not alias the input ------------------------------------
if [[ -e $OUTPUT ]] && [[ "$(readlink -f "$OUTPUT")" == "$(readlink -f "$SOURCE")" ]]; then
  echo "[FATAL] output_root resolves to source_root: $(readlink -f "$SOURCE")" >&2
  exit 2
fi

# --- 4) no concurrent sampler ----------------------------------------------
if pgrep -f 'scripts/sample_fragments_by_bp.py' >/dev/null 2>&1; then
  echo "[FATAL] a sampling run is already in progress" >&2
  exit 2
fi

args=(-u -m tiara2.cli bp-balance --config "$CONFIG")
[[ ${FORCE:-0} == 1 ]] && args+=(--force)
[[ ${DRY_RUN:-0} == 1 ]] && args+=(--dry-run)

exec "$PYTHON" "${args[@]}"
