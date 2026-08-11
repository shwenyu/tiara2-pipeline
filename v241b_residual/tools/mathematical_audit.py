#!/usr/bin/env python3
"""M1-M5 audit for frozen base + residual logits on a labeled split."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from torch.utils.data import DataLoader

HEADS=("root","euk","prok","organelle"); SIZES={"root":3,"euk":8,"prok":2,"organelle":2}

def macro_f1(y,p,k):
 vals=[]
 for c in range(k):
  tp=np.sum((y==c)&(p==c));fp=np.sum((y!=c)&(p==c));fn=np.sum((y==c)&(p!=c));vals.append(2*tp/max(1,2*tp+fp+fn))
 return float(np.mean(vals))
def nll(z,y): return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]))
def best_binary_threshold(prob,truth):
 order=np.argsort(-prob); yy=truth[order].astype(np.int64);tp=np.cumsum(yy);fp=np.cumsum(1-yy);total=tp[-1] if len(tp) else 0
 f1=2*tp/np.maximum(1,2*tp+fp+(total-tp));i=int(np.argmax(f1)) if len(f1) else 0
 return {"empirical_threshold":float(prob[order[i]]) if len(prob) else None,"best_f1":float(f1[i]) if len(f1) else 0.0,"f1_half":float(f1[i]/2) if len(f1) else 0.0}
def lengths(crops):
 out=[]
 for name in ("bacteria","archaea","eukarya","mitochondria","plastids"):
  n=None
  with (Path(crops)/"validation"/f"{name}.fasta").open() as f:
   for line in f:
    if line.startswith(">"):
     if n is not None:out.append(n)
     n=0
    elif n is not None:n+=len(line.strip())
   if n is not None:out.append(n)
 return np.asarray(out,np.int32)
def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--base-logits",required=True);p.add_argument("--labels",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--trainer",required=True);p.add_argument("--crops",required=True);p.add_argument("--out",required=True);p.add_argument("--batch",type=int,default=1024);a=p.parse_args()
 spec=importlib.util.spec_from_file_location("v241b_train",a.trainer);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);model=m.ResidualAdapter();model.load_state_dict(ck["state_dict"]);model.cuda().eval();ds=m.CropDataset(a.features,a.base_logits,a.labels,"validation");loader=DataLoader(ds,batch_size=a.batch,num_workers=8,pin_memory=True)
 ys=[];bs=[];rs=[]
 with torch.inference_mode():
  for x,b,y in loader:ys.append(y.numpy());bs.append(b.numpy());r=model(x.cuda(non_blocking=True));rs.append(np.concatenate([r[h].cpu().numpy() for h in HEADS],1))
 yflat=np.concatenate(ys);bflat=np.concatenate(bs);rflat=np.concatenate(rs);lens=lengths(a.crops)
 if len(lens)!=len(yflat):raise ValueError("length/row mismatch")
 offsets={"root":(0,3),"euk":(3,11),"prok":(11,13),"organelle":(13,15)};root=yflat[:,0];bins=((800,999),(1000,1249),(1250,1499),(1500,1750),(1751,1999),(2000,2249),(2250,2499));result={"scope":"validation_not_external_benchmark","rows":len(lens),"cells":{}}
 for hidx,h in enumerate(HEADS):
  lo,hi=offsets[h];mask_head=np.ones(len(root),bool) if h=="root" else root=={"euk":0,"prok":1,"organelle":2}[h]
  for lower,upper in bins:
   mask=mask_head&(lens>=lower)&(lens<=upper)&(yflat[:,hidx]>=0);yy=yflat[mask,hidx];base=bflat[mask,lo:hi];res=rflat[mask,lo:hi]
   if not len(yy):continue
   bp=base.argmax(1);fp=(base+.125*res).argmax(1);d=float(np.mean(bp!=fp));bc=bp==yy;fc=fp==yy
   alphas=np.linspace(-.25,1,51);surface=[]
   for alpha in alphas:
    z=base+alpha*res;surface.append({"alpha":float(alpha),"nll":nll(z,yy),"macro_f1":macro_f1(yy,z.argmax(1),SIZES[h])})
   opt=minimize_scalar(lambda beta:nll(base*beta,yy),bounds=(.05,10),method="bounded");temp=1/opt.x;prob=softmax(base/temp,axis=1);thresholds={str(c):best_binary_threshold(prob[:,c],yy==c) for c in range(SIZES[h])}
   bl=logsumexp(base,axis=1)-base[np.arange(len(yy)),yy];fl=logsumexp(base+.125*res,axis=1)-(base+.125*res)[np.arange(len(yy)),yy]
   result["cells"][f"{h}/{lower}-{upper}"]={"N":int(len(yy)),"class_counts":np.bincount(yy,minlength=SIZES[h]).tolist(),"disagreement":d,"delta_min_accuracy":float(2.8*np.sqrt(d/len(yy))),"quadrants":{"both_correct":int(np.sum(bc&fc)),"base_only":int(np.sum(bc&~fc)),"residual_only":int(np.sum(~bc&fc)),"both_wrong":int(np.sum(~bc&~fc))},"base_accuracy":float(np.mean(bc)),"candidate_accuracy":float(np.mean(fc)),"base_macro_f1":macro_f1(yy,bp,SIZES[h]),"candidate_macro_f1":macro_f1(yy,fp,SIZES[h]),"loss_correlation":float(np.corrcoef(bl,fl)[0,1]),"sigma_base_over_candidate":float(np.std(bl)/max(1e-12,np.std(fl))),"alpha_surface":surface,"temperature":{"T_star":float(temp),"nll":float(opt.fun)},"one_vs_rest_thresholds":thresholds}
 Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps({"out":a.out,"rows":len(lens),"cells":len(result["cells"])},indent=2))
if __name__=="__main__":main()
