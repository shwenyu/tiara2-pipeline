#!/usr/bin/env bash
# End-to-end smoke test with a fake mmseqs (duplicate = identical sequence).
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPTS=$ROOT/scripts
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/src"/{train,validation,test} "$T/work" "$T/results"

# fake mmseqs: easy-search emits a hit when query seq == target seq (id/cov=1).
cat > "$T/bin/mmseqs" <<'PY'
#!/usr/bin/env python3
import sys
from pathlib import Path
a=sys.argv[1:]
if a and a[0]=='version': print('fake'); raise SystemExit
if a and a[0]=='easy-search':
 q,t,out,tmp=a[1:5]
 def rd(p):
  z=[];h=None;s=[]
  for l in open(p):
   l=l.strip()
   if l.startswith('>'):
    if h:z.append((h,''.join(s)))
    h=l[1:].split()[0];s=[]
   elif l:s.append(l)
  if h:z.append((h,''.join(s)))
  return z
 T=rd(t);Path(tmp).mkdir(parents=True,exist_ok=True)
 with open(out,'w') as f:
  for qh,qs in rd(q):
   for th,ts in T:
    if qs==ts and qh!=th:
     f.write(f'{qh}\t{th}\t1\t1\t1\n');break
 raise SystemExit
raise SystemExit('unsupported: '+repr(a))
PY
chmod +x "$T/bin/mmseqs"

# Synthetic corpus, bacteria class only, discrete length 1000 & 2000.
python3 - "$T/src" <<'PY'
import sys
from pathlib import Path
r=Path(sys.argv[1])
A='A'*1000; B='C'*1000; C='G'*1000; D='T'*2000; E='A'*2000
# test defines protected sequences
(r/'test/bacteria.fasta').write_text(f'>test_1\n{A}\n>test_2\n{D}\n')
# validation: v1 dup of test A (cross-split -> remove); v2 unique;
#             v3==v4 same-split dup (must BOTH be kept)
(r/'validation/bacteria.fasta').write_text(
  f'>val_1\n{A}\n>val_2\n{B}\n>val_3\n{C}\n>val_4\n{C}\n')
# train: t1 dup of test A -> remove; t2 dup of surviving val_2(B) -> remove;
#        t3 dup of test D (len2000, different bin) -> remove; t4 unique
(r/'train/bacteria.fasta').write_text(
  f'>tr_1\n{A}\n>tr_2\n{B}\n>tr_3\n{D}\n>tr_4\n{E}\n')
PY

export PATH="$T/bin:$PATH"
cd "$SCRIPTS"
python3 bin_by_length.py --source-root "$T/src" --out "$T/work/binned" \
  --classes bacteria --partition-by class --mode discrete \
  --discrete-lengths 1000 2000 --min-cov 0.95
bash run_dedup_bins.sh "$T/work" "$T/bin/mmseqs" 0.95 0.95 2.0 100 1 2 100M
python3 regroup_by_metadata.py --source-root "$T/src" --out "$T/results" \
  --classes bacteria --summary "$T/results/regroup_summary.json" \
  --removed-json $(find "$T/work/dedup" -name 'removed_*.json')

python3 - "$T" <<'PY'
import json,sys
from pathlib import Path
T=Path(sys.argv[1])
def ids(p):
 return [l[1:].split()[0] for l in open(p) if l.startswith('>')]
test=ids(T/'results/test/bacteria.fasta')
val=ids(T/'results/validation/bacteria.fasta')
tr=ids(T/'results/train/bacteria.fasta')
assert set(test)=={'test_1','test_2'}, test
# val_1 removed (dup of test); val_3 & val_4 kept (same-split dup preserved)
assert set(val)=={'val_2','val_3','val_4'}, val
# tr_1(dupA), tr_2(dup surviving val_2 B), tr_3(dup test D) removed; tr_4 kept
assert set(tr)=={'tr_4'}, tr
print('SMOKE TEST OK: bin+cross-split-priority+regroup, same-split dup preserved, cross-bin not mis-removed')
PY
