#!/usr/bin/env python3
"""Extract parent accession groups aligned to frozen short validation rows."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
FILES=("bacteria.fasta","archaea.fasta","eukarya.fasta","mitochondria.fasta","plastids.fasta")
def headers(path):
    with Path(path).open() as f:
        for line in f:
            if line.startswith(">"):yield line[1:].strip()
def sha(path):
    h=hashlib.sha256();
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--train-ready",required=True);p.add_argument("--short-features",required=True);p.add_argument("--out",required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    m=json.loads((Path(a.short_features)/"composite_features.json").read_text());shard=m["splits"]["validation"]["shards"][0];idx=np.load(shard["indices"],mmap_mode="r");codes=np.empty(len(idx),dtype=np.int64);mapping={};names=[];cursor=source=0
    for fn in FILES:
        for header in headers(Path(a.train_ready)/"validation"/fn):
            if cursor<len(idx) and source==int(idx[cursor]):
                accession=header.split()[0].split("|")[0];code=mapping.setdefault(accession,len(mapping));
                if code==len(names):names.append(accession)
                codes[cursor]=code;cursor+=1
            source+=1
    if cursor!=len(idx):raise ValueError(f"matched {cursor}, expected {len(idx)}")
    path=out/"validation_parent.i64.npy";np.save(path,codes,allow_pickle=False)
    with (out/"parent_mapping.tsv").open("w") as f:
        f.write("parent_code\taccession\n");
        for i,name in enumerate(names):f.write(f"{i}\t{name}\n")
    (out/"parent_groups.json").write_text(json.dumps({"rows":len(idx),"unique_parents":len(names),"array":str(path),"array_sha256":sha(path),"source_indices":str(Path(shard["indices"]).resolve()),"source_indices_sha256":sha(shard["indices"])},indent=2,sort_keys=True)+"\n")
    print(json.dumps({"rows":len(idx),"unique_parents":len(names),"array":str(path)},indent=2))
if __name__=="__main__":main()
