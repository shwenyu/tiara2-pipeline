#!/usr/bin/env python3
"""Audit realized v2.4.1-B coverage and split leakage by accession proxy."""
from __future__ import annotations
import argparse,csv,json,re
from collections import Counter,defaultdict
from pathlib import Path

FILES=("bacteria","archaea","eukarya","mitochondria","plastids")
BINS=((800,999),(1000,1249),(1250,1499),(1500,1750),(1751,1999),(2000,2249),(2250,2499))
ACC=re.compile(r"^(GC[AF]_\d+(?:\.\d+)?)")

def records(path):
 header=None;n=0
 with path.open() as f:
  for line in f:
   if line.startswith(">"):
    if header is not None:yield header,n
    header=line[1:].strip();n=0
   elif header is not None:n+=len(line.strip())
  if header is not None:yield header,n
def accession(header):
 token=header.split()[0].split("|")[0];m=ACC.match(token);return m.group(1) if m else token
def length_bin(n):
 for lo,hi in BINS:
  if lo<=n<=hi:return f"{lo}-{hi}"
 return "outside"
def gini(values):
 x=sorted(values);n=len(x);s=sum(x)
 return 0.0 if not n or not s else (2*sum((i+1)*v for i,v in enumerate(x))/(n*s)-(n+1)/n)
def main():
 p=argparse.ArgumentParser();p.add_argument("--crops",required=True);p.add_argument("--out",required=True);a=p.parse_args();root=Path(a.crops);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 cells=defaultdict(lambda:{"fragments":0,"bp":0,"accessions":Counter()});split_accessions={s:set() for s in ("train","validation")};total=Counter()
 for split in ("train","validation"):
  for cls in FILES:
   for header,n in records(root/split/f"{cls}.fasta"):
    acc=accession(header);b=length_bin(n);key=(split,cls,b);z=cells[key];z["fragments"]+=1;z["bp"]+=n;z["accessions"][acc]+=1;split_accessions[split].add(acc);total[(split,cls)]+=1
 rows=[]
 for (split,cls,b),z in sorted(cells.items()):
  counts=list(z["accessions"].values());rows.append({"split":split,"class":cls,"length_bin":b,"fragments":z["fragments"],"bp":z["bp"],"unique_accessions":len(counts),"max_fragments_per_accession":max(counts),"median_fragments_per_accession":sorted(counts)[len(counts)//2],"gini_fragments_per_accession":gini(counts)})
 with (out/"coverage_cells.tsv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),delimiter="\t");w.writeheader();w.writerows(rows)
 overlap=sorted(split_accessions["train"]&split_accessions["validation"]);(out/"cross_split_accessions.txt").write_text("\n".join(overlap)+("\n" if overlap else ""))
 report={"version":"2.4.1-B","scope":"realized continuous crop corpus","lineage_resolution":"assembly accession proxy; taxonomy join pending","rows":rows,"totals":{"train":sum(v for (s,_),v in total.items() if s=="train"),"validation":sum(v for (s,_),v in total.items() if s=="validation")},"unique_accessions":{s:len(v) for s,v in split_accessions.items()},"cross_split_accessions":{"count":len(overlap),"file":"cross_split_accessions.txt"},"ready_for_taxonomy_join":True}
 (out/"coverage_audit.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({"out":str(out),"totals":report["totals"],"unique_accessions":report["unique_accessions"],"cross_split":len(overlap)},indent=2))
if __name__=="__main__":main()
