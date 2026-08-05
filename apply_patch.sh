#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-.}"
[[ -f "$DEST/pyproject.toml" && -d "$DEST/tiara" ]] || { echo "usage: $0 /path/to/tiara-repo" >&2; exit 2; }
cp -a "$SRC/tiara/." "$DEST/tiara/"
mkdir -p "$DEST/config" "$DEST/tests/hierarchical"
cp -a "$SRC/config/." "$DEST/config/"
cp -a "$SRC/tests/hierarchical/." "$DEST/tests/hierarchical/"
cp "$SRC/V2.3.0_HIERARCHICAL_README.md" "$DEST/"
echo "Tiara2 v2.3.0 hierarchical patch installed into $DEST"
