#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-.}"
[[ -f "$DEST/pyproject.toml" && -d "$DEST/tiara/hierarchical" ]] || {
  echo "usage: $0 /path/to/complete-tiara2-v2.3.0-repo" >&2
  exit 2
}
files=(
  tiara/hierarchical/schema.py
  tiara/hierarchical/train.py
  tiara/hierarchical/evaluate.py
  tiara/hierarchical/__main__.py
  tiara/hierarchical/completeness.py
  tiara/hierarchical/freeze.py
  tiara/hierarchical/delta.py
  tiara/hierarchical/features_v231.py
  tiara/hierarchical/gates.py
  tiara/hierarchical/release.py
  config/config_v2_3_1_euk_completeness.yaml
  scripts/run_v2_3_1_euk_completeness.sh
  tests/hierarchical/test_schema.py
  tests/hierarchical/test_v231_pipeline.py
  V2.3.1_EUK_COMPLETENESS_README.md
  V2.3.1_PATCH_MANIFEST.md
)
for rel in "${files[@]}"; do
  [[ -f "$SRC/$rel" ]] || { echo "patch file missing: $SRC/$rel" >&2; exit 2; }
  mkdir -p "$DEST/$(dirname "$rel")"
  cp "$SRC/$rel" "$DEST/$rel"
done
chmod +x "$DEST/scripts/run_v2_3_1_euk_completeness.sh"
echo "Tiara2 v2.3.1 Euk-completeness patch installed into $DEST"
