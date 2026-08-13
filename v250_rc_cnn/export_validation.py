#!/usr/bin/env python3
"""Export epoch-6 RC-CNN residual logits for the frozen validation rows."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from model import HEADS, RCShortResidualExpert

def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--out",required=True);p.add_argument("--batch",type=int,default=128);a=p.parse_args()
    rank=int(os.environ.get("RANK",0));local=int(os.environ.get("LOCAL_RANK",0));world=int(os.environ.get("WORLD_SIZE",1))
    if world>1:torch.cuda.set_device(local);dist.init_process_group("nccl")
    dev=torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    cache=json.loads((Path(a.cache)/"sequence_cache.json").read_text());spec=cache["splits"]["validation"];n=int(spec["rows"]);max_bp=int(spec["max_bp"])
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);model=RCShortResidualExpert(**ck["model"]);model.load_state_dict(ck["state_dict"]);model.to(dev).eval()
    out=Path(a.out)
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        for h,size in ck["model"]["head_sizes"].items():
            with (out/f"{h}.f32").open("wb") as f:f.truncate(n*int(size)*4)
    if world>1:dist.barrier()
    tokens=np.memmap(spec["tokens"],dtype=np.uint8,mode="r",shape=(n,max_bp));lengths=np.memmap(spec["lengths"],dtype=np.int16,mode="r",shape=(n,));outputs={h:np.memmap(out/f"{h}.f32",dtype=np.float32,mode="r+",shape=(n,int(size))) for h,size in ck["model"]["head_sizes"].items()}
    start=n*rank//world;end=n*(rank+1)//world
    with torch.inference_mode():
        for left in range(start,end,a.batch):
            right=min(end,left+a.batch);tok=torch.from_numpy(np.asarray(tokens[left:right]).copy()).long().to(dev);length=torch.from_numpy(np.asarray(lengths[left:right]).astype(np.int64)).to(dev)
            with torch.autocast("cuda",dtype=torch.float16,enabled=dev.type=="cuda"):delta=model(tok,length,True)
            for h in HEADS:outputs[h][left:right]=delta[h].float().cpu().numpy()
            if rank==0 and (left-start)%(a.batch*100)==0:print(f"rank0 {left-start:,}/{end-start:,}",flush=True)
    for x in outputs.values():x.flush()
    if world>1:dist.barrier()
    if rank==0:(out/"residual_logits.json").write_text(json.dumps({"format":"tiara2-v250-validation-residual-logits-v1","checkpoint":str(Path(a.checkpoint).resolve()),"checkpoint_epoch":ck["epoch"],"rows":n,"head_sizes":ck["model"]["head_sizes"],"source_cache":str(Path(a.cache).resolve())},indent=2,sort_keys=True)+"\n")
    if world>1:dist.destroy_process_group()
if __name__=="__main__":main()
