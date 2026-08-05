#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-.}"
[[ -d "$DEST/tiara/hierarchical" ]] || { echo "usage: $0 /path/to/patched-tiara-repo" >&2; exit 2; }
cp "$SRC/tiara/hierarchical/data.py" "$DEST/tiara/hierarchical/data.py"
cp "$SRC/tiara/hierarchical/__main__.py" "$DEST/tiara/hierarchical/__main__.py"
cp "$SRC/V2.3.0_HIERARCHICAL_README.md" "$DEST/"
echo "hotfix1 installed: init-metadata is now available"
