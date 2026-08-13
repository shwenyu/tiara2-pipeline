#!/usr/bin/env python3
"""Cache frozen v2.3.2 logits for the exact short-expert row indices."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

HEADS=("root","euk","prok","organelle")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--short-features",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--out",required=True); p.add_argument("--batch",type=int,default=2048); a=p.parse_args()
    rank=int(os.environ.get("RANK",0)); local=int(os.environ.get("LOCAL_RANK",0)); world=int(os.environ.get("WORLD_SIZE",1))
    if world>1: torch.cuda.set_device(local); dist.init_process_group("nccl")
    dev=torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    from tiara.hierarchical.model import HierarchicalClassifier
    manifest=json.loads((Path(a.short_features)/"composite_features.json").read_text())
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); model=HierarchicalClassifier(**ck["model"]); model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
    out=Path(a.out)
    if rank==0: out.mkdir(parents=True,exist_ok=True)
    if world>1: dist.barrier()
    result={"format":"tiara2-v250-frozen-base-logits-v1","checkpoint":str(Path(a.checkpoint).resolve()),"splits":{}}
    for split in ("train","validation"):
        shard=manifest["splits"][split]["shards"][0]; n=int(shard["rows"]); base_n=int(shard["base_rows"]); dim=int(shard["dim"])
        indices=np.load(shard["indices"],mmap_mode="r"); x=np.memmap(shard["files"]["X"],dtype=np.float32,mode="r",shape=(base_n,dim))
        sd=out/split
        if rank==0:
            sd.mkdir(exist_ok=True)
            for h,size in ck["model"]["head_sizes"].items():
                with (sd/f"{h}.f32").open("wb") as f: f.truncate(n*int(size)*4)
        if world>1: dist.barrier()
        values={h:np.memmap(sd/f"{h}.f32",dtype=np.float32,mode="r+",shape=(n,int(size))) for h,size in ck["model"]["head_sizes"].items()}
        start=n*rank//world; end=n*(rank+1)//world
        with torch.inference_mode():
            for left in range(start,end,a.batch):
                right=min(end,left+a.batch); batch=torch.from_numpy(np.asarray(x[indices[left:right]]).copy()).to(dev)
                with torch.autocast("cuda",dtype=torch.float16,enabled=dev.type=="cuda"): logits=model(batch)
                for h in HEADS: values[h][left:right]=logits[h].float().cpu().numpy()
                if rank==0 and (left-start)%(a.batch*100)==0: print(f"[{split}] rank0 {left-start:,}/{end-start:,}",flush=True)
        for v in values.values(): v.flush()
        if world>1: dist.barrier()
        result["splits"][split]={"rows":n,"head_sizes":ck["model"]["head_sizes"]}
    if rank==0: (out/"base_logits.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    if world>1: dist.destroy_process_group()

if __name__=="__main__": main()
