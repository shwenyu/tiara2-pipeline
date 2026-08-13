#!/usr/bin/env python3
"""DDP trainer for the v2.5.0 raw-sequence RC-CNN residual expert."""
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler

from model import HEADS, RCShortResidualExpert


class SequenceDataset(Dataset):
    def __init__(self, spec, base_dir, head_sizes):
        self.spec=spec; self.n=int(spec["rows"]); self.max_bp=int(spec["max_bp"]); self.base_dir=Path(base_dir); self.head_sizes=head_sizes; self._opened=False
    def _open(self):
        if self._opened:return
        self.tokens=np.memmap(self.spec["tokens"],dtype=np.uint8,mode="r",shape=(self.n,self.max_bp)); self.lengths=np.memmap(self.spec["lengths"],dtype=np.int16,mode="r",shape=(self.n,))
        self.labels={h:np.memmap(self.spec["labels"][h],dtype=np.int64,mode="r",shape=(self.n,)) for h in HEADS}
        self.base={h:np.memmap(self.base_dir/f"{h}.f32",dtype=np.float32,mode="r",shape=(self.n,int(self.head_sizes[h]))) for h in HEADS}; self._opened=True
    def __len__(self):return self.n
    def __getitem__(self,i):
        self._open(); return (torch.from_numpy(np.asarray(self.tokens[i]).copy()).long(),torch.tensor(int(self.lengths[i])),{h:torch.tensor(int(self.labels[h][i])) for h in HEADS},{h:torch.from_numpy(np.asarray(self.base[h][i]).copy()) for h in HEADS})


class DistributedWeightedSampler(Sampler):
    def __init__(self,weights,world,rank,seed): self.w=torch.as_tensor(weights,dtype=torch.double); self.world=world; self.rank=rank; self.seed=seed; self.epoch=0; self.total=math.ceil(len(weights)/world)*world
    def set_epoch(self,e):self.epoch=e
    def __len__(self):return self.total//self.world
    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed+self.epoch); idx=torch.multinomial(self.w,self.total,replacement=True,generator=g); return iter(idx[self.rank::self.world].tolist())


class DistributedEvalSampler(Sampler):
    def __init__(self,n,world,rank):self.n=n;self.world=world;self.rank=rank
    def __len__(self):return max(0,(self.n-self.rank+self.world-1)//self.world)
    def __iter__(self):return iter(range(self.rank,self.n,self.world))


def reduce_sum(x):
    if dist.is_initialized():dist.all_reduce(x,op=dist.ReduceOp.SUM)
    return x


def macro_f1(confusion):
    tp=torch.diag(confusion); fp=confusion.sum(0)-tp; fn=confusion.sum(1)-tp
    return float((2*tp/(2*tp+fp+fn).clamp_min(1)).mean())


def loss_and_counts(logits,targets,root_index):
    loss=F.cross_entropy(logits["root"],targets["root"]); terms=1
    for head,name,weight in (("euk","euk_nuclear",1.0),("prok","prok",0.5),("organelle","organelle",0.5)):
        mask=targets["root"]==root_index[name]
        if mask.any():loss=loss+weight*F.cross_entropy(logits[head][mask],targets[head][mask]);terms+=1
    return loss/terms


def evaluate(model,loader,dev,root_index,max_batches=None):
    model.eval(); sizes={"root":3,"euk":8,"prok":2,"organelle":2}
    base_cm={h:torch.zeros((n,n),device=dev,dtype=torch.float64) for h,n in sizes.items()};hybrid_cm={h:torch.zeros((n,n),device=dev,dtype=torch.float64) for h,n in sizes.items()}
    transfer=torch.zeros(4,device=dev,dtype=torch.float64);changed=torch.zeros(2,device=dev,dtype=torch.float64)
    with torch.inference_mode():
        for batch_index,(tok,length,y,base) in enumerate(loader,1):
            tok=tok.to(dev);length=length.to(dev);y={h:v.to(dev) for h,v in y.items()};base={h:v.to(dev) for h,v in base.items()}
            with torch.autocast("cuda",dtype=torch.float16,enabled=dev.type=="cuda"): delta=model(tok,length,True); logits={h:base[h]+delta[h] for h in HEADS}
            bp=base["root"].argmax(1); hp=logits["root"].argmax(1); target=y["root"];bc=bp==target;hc=hp==target
            transfer+=torch.stack([(bc&hc).sum(),(bc&~hc).sum(),(~bc&hc).sum(),(~bc&~hc).sum()]).double();changed+=torch.tensor([(bp!=hp).sum(),len(target)],device=dev,dtype=torch.float64)
            masks={"root":torch.ones_like(target,dtype=torch.bool),"euk":target==root_index["euk_nuclear"],"prok":target==root_index["prok"],"organelle":target==root_index["organelle"]}
            for h,n in sizes.items():
                m=masks[h]; truth=y[h][m]; bpred=base[h][m].argmax(1); hpred=logits[h][m].argmax(1)
                base_cm[h]+=torch.bincount(truth*n+bpred,minlength=n*n).reshape(n,n);hybrid_cm[h]+=torch.bincount(truth*n+hpred,minlength=n*n).reshape(n,n)
            if max_batches and batch_index>=max_batches: break
    reduce_sum(transfer);reduce_sum(changed)
    for h in sizes:reduce_sum(base_cm[h]);reduce_sum(hybrid_cm[h])
    metrics={"changed_fraction":float(changed[0]/changed[1]),"both_correct":int(transfer[0]),"base_only_correct":int(transfer[1]),"hybrid_only_correct":int(transfer[2]),"both_wrong":int(transfer[3])}
    for h in sizes:metrics[f"base_{h}_macro_f1"]=macro_f1(base_cm[h]);metrics[f"hybrid_{h}_macro_f1"]=macro_f1(hybrid_cm[h]);metrics[f"delta_{h}_macro_f1"]=metrics[f"hybrid_{h}_macro_f1"]-metrics[f"base_{h}_macro_f1"]
    metrics["selector_score"]=sum(metrics[f"hybrid_{h}_macro_f1"] for h in sizes)/len(sizes)
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",required=True);p.add_argument("--base-logits",required=True);p.add_argument("--short-features",required=True);p.add_argument("--out",required=True);p.add_argument("--epochs",type=int,default=50);p.add_argument("--batch",type=int,default=96);p.add_argument("--workers",type=int,default=4);p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--seed",type=int,default=42);p.add_argument("--smoke-steps",type=int);a=p.parse_args()
    rank=int(os.environ.get("RANK",0));local=int(os.environ.get("LOCAL_RANK",0));world=int(os.environ.get("WORLD_SIZE",1));
    if world>1:torch.cuda.set_device(local);dist.init_process_group("nccl")
    dev=torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu");random.seed(a.seed+rank);np.random.seed(a.seed+rank);torch.manual_seed(a.seed+rank)
    cache=json.loads((Path(a.cache)/"sequence_cache.json").read_text()); base_meta=json.loads((Path(a.base_logits)/"base_logits.json").read_text()); head_sizes=base_meta["splits"]["train"]["head_sizes"]
    from tiara.hierarchical.schema import HierarchySchema,EUK_WEIGHTS
    from tiara.hierarchical.sampling_v232 import build_hierarchical_sampling_plan
    schema=HierarchySchema.from_dict(cache["schema"]);root_index=schema.index("root")
    train=SequenceDataset(cache["splits"]["train"],Path(a.base_logits)/"train",head_sizes);val=SequenceDataset(cache["splits"]["validation"],Path(a.base_logits)/"validation",head_sizes); train._open()
    root=np.asarray(train.labels["root"]);euk=np.asarray(train.labels["euk"]);euk_mask=root==root_index["euk_nuclear"]
    donor=json.loads((Path(a.short_features)/"donor_index/donor_index_manifest.json").read_text()); donor_arrays={k:np.load(donor["arrays"][k],mmap_mode="r") for k in ("accession","species","genus")}
    plan=build_hierarchical_sampling_plan(root,euk,schema,EUK_WEIGHTS,4.0,donor_arrays,{k:.05 for k in donor_arrays})
    sampler=DistributedWeightedSampler(plan.sample_weights,world,rank,a.seed);vs=DistributedEvalSampler(len(val),world,rank)
    tl=DataLoader(train,batch_size=a.batch,sampler=sampler,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0);vl=DataLoader(val,batch_size=a.batch,sampler=vs,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    model=RCShortResidualExpert(head_sizes).to(dev); model=DDP(model,device_ids=[local]) if world>1 else model;opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);scaler=torch.amp.GradScaler("cuda",enabled=dev.type=="cuda")
    out=Path(a.out)
    if rank==0:out.mkdir(parents=True,exist_ok=True);(out/"sampling_plan.json").write_text(json.dumps(plan.report,indent=2,sort_keys=True)+"\n")
    history=[];best=-1
    for epoch in range(1,a.epochs+1):
        sampler.set_epoch(epoch);model.train();total=0.;seen=0;t0=time.time()
        for step,(tok,length,y,base) in enumerate(tl,1):
            tok=tok.to(dev,non_blocking=True);length=length.to(dev,non_blocking=True);y={h:v.to(dev,non_blocking=True) for h,v in y.items()};base={h:v.to(dev,non_blocking=True) for h,v in base.items()};opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=dev.type=="cuda"):delta=model(tok,length,True);logits={h:base[h]+delta[h] for h in HEADS};loss=loss_and_counts(logits,y,root_index)
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(opt);scaler.update();total+=float(loss.detach())*len(tok);seen+=len(tok)
            if rank==0 and step%100==0:print(f"epoch={epoch} step={step}/{len(tl)} loss={total/seen:.6f}",flush=True)
            if a.smoke_steps and step>=a.smoke_steps:break
        metrics=evaluate(model,vl,dev,root_index,5 if a.smoke_steps else None);metrics.update({"epoch":epoch,"train_loss":total/max(seen,1),"seconds":time.time()-t0})
        if rank==0:
            print(json.dumps(metrics,sort_keys=True),flush=True);history.append(metrics);(out/"training_history.json").write_text(json.dumps(history,indent=2)+"\n")
            state=model.module if isinstance(model,DDP) else model;score=metrics["selector_score"]
            if score>best:best=score;torch.save({"version":"2.5.0-RC-CNN","epoch":epoch,"model":{"head_sizes":head_sizes,"embed_dim":32,"channels":192,"dropout":.15},"state_dict":state.state_dict(),"schema":cache["schema"],"metrics":metrics,"data_contract":{"sequence_cache":str(Path(a.cache).resolve()),"source_short_features":str(Path(a.short_features).resolve()),"unchanged_dataset":True,"router_trained":False}},out/"rc_cnn_residual_model.pt")
        if a.smoke_steps:break
    if world>1:dist.destroy_process_group()

if __name__=="__main__":main()
