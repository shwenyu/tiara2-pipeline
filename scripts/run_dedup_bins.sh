#!/usr/bin/env bash
# Per-bin cross-split double-95 dedup driver.
# Iterates shard tree: for each partition/bin, search lower splits against
# higher-priority splits (test>validation>train), emit removed-ID JSON.
# Idempotent: per-bin .done markers allow resume after interruption.
set -Eeuo pipefail

WORK=$1; MMSEQS=$2; MIN_ID=$3; MIN_COV=$4; SENS=$5; MAXSEQS=$6; MAXACC=$7; THREADS=$8; SML=$9
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BINNED=$WORK/binned/shards
DEDUP=$WORK/dedup
mkdir -p "$DEDUP"
export MMSEQS_NUM_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" TTY=0
log(){ printf '[%(%F %T)T] %s\n' -1 "$*"; }

search_pair(){ # query_fasta target_fasta out_tsv tmp
  local q=$1 t=$2 out=$3 tmp=$4
  [[ -s "$t" && -s "$q" ]] || { : > "$out"; return; }
  "$MMSEQS" easy-search "$q" "$t" "$out.tmp" "$tmp" \
    --search-type 3 -s "$SENS" --min-seq-id "$MIN_ID" -c "$MIN_COV" --cov-mode 0 \
    --alignment-mode 3 --max-seqs "$MAXSEQS" --max-accept "$MAXACC" \
    --threads "$THREADS" --split-memory-limit "$SML" --remove-tmp-files 1 \
    --format-output 'query,target,fident,qcov,tcov' >/dev/null 2>&1
  mv "$out.tmp" "$out"
}

count_seqs(){ grep -c '^>' "$1" 2>/dev/null || echo 0; }

for partition in "$BINNED"/*; do
  [[ -d "$partition" ]] || continue
  pname=$(basename "$partition")
  for bin in "$partition"/*; do
    [[ -d "$bin" ]] || continue
    bname=$(basename "$bin")
    od=$DEDUP/$pname/$bname; mkdir -p "$od"
    marker=$od/.done
    [[ -f "$marker" ]] && { log "resume bin $pname/$bname"; continue; }
    test_fa=$bin/test.fasta; val_fa=$bin/validation.fasta; train_fa=$bin/train.fasta

    # validation -> test
    if [[ -f "$val_fa" ]]; then
      search_pair "$val_fa" "${test_fa:-/dev/null}" "$od/validation_vs_test.tsv" "$od/tmp_v"
      python "$SCRIPT_DIR/dedup_priority.py" --hits "$od/validation_vs_test.tsv" \
        --query-split validation --out "$od/removed_validation.json" \
        --min-id "$MIN_ID" --min-cov "$MIN_COV" --query-count "$(count_seqs "$val_fa")"
    fi

    # train -> test + surviving validation
    if [[ -f "$train_fa" ]]; then
      prot=$od/protected.fasta; : > "$prot"
      [[ -f "$test_fa" ]] && cat "$test_fa" >> "$prot"
      if [[ -f "$val_fa" && -f "$od/removed_validation.json" ]]; then
        python "$SCRIPT_DIR/regroup_by_metadata.py" --help >/dev/null 2>&1 || true
        python - "$val_fa" "$od/removed_validation.json" >> "$prot" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
val, rj = sys.argv[1], sys.argv[2]
removed=set(json.load(open(rj)).get('removed_ids',[]))
h=None;keep=True;buf=[]
out=sys.stdout
for line in open(val):
    if line.startswith('>'):
        fid=line[1:].split()[0]
        keep=fid not in removed
    if keep: out.write(line)
PY
      elif [[ -f "$val_fa" ]]; then
        cat "$val_fa" >> "$prot"
      fi
      search_pair "$train_fa" "$prot" "$od/train_vs_protected.tsv" "$od/tmp_t"
      python "$SCRIPT_DIR/dedup_priority.py" --hits "$od/train_vs_protected.tsv" \
        --query-split train --out "$od/removed_train.json" \
        --min-id "$MIN_ID" --min-cov "$MIN_COV" --query-count "$(count_seqs "$train_fa")"
      rm -f "$prot"
    fi
    touch "$marker"
    log "DONE bin $pname/$bname"
  done
done
log "all bins complete"
