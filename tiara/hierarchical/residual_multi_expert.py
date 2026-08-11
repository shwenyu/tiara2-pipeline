#!/usr/bin/env python3
"""Unified deterministic inference for the v2.3.2 + v2.4.1-B residual expert."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tiara.hierarchical.data import fasta
from tiara.hierarchical.model import HierarchicalClassifier, probabilities
from tiara.hierarchical.schema import HierarchySchema
from tiara.src.transformations import TfidfWeighter
from tiara.training.featurize_cache import featurize_block

HEADS=("root","euk","prok","organelle")

class ResidualAdapter(nn.Module):
 def __init__(self,dim_in=5376,hidden=(1024,512),head_sizes=None,dropout=.1):
  super().__init__();head_sizes=head_sizes or {"root":3,"euk":8,"prok":2,"organelle":2};layers=[];last=dim_in
  for width in hidden:layers += [nn.Linear(last,width),nn.GELU(),nn.Dropout(dropout)];last=width
  self.encoder=nn.Sequential(*layers);self.heads=nn.ModuleDict({k:nn.Linear(last,v) for k,v in head_sizes.items()})
 def forward(self,x):z=self.encoder(x);return {k:h(z) for k,h in self.heads.items()}

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8<<20),b""):h.update(block)
 return h.hexdigest()

def classify(bundle,input_fasta,output,batch=512,device=None,min_len=1000,max_records=None):
 bp=Path(bundle).resolve();m=json.loads(bp.read_text())
 if m.get("format")!="tiara2-residual-multi-expert-v1":raise ValueError("invalid residual multi-expert manifest")
 resolve=lambda value:(Path(value) if Path(value).is_absolute() else (bp.parent/value).resolve())
 dev=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"));base_path=resolve(m["base"]["checkpoint"]);res_path=resolve(m["residual"]["checkpoint"])
 for name,path in (("base",base_path),("residual",res_path)):
  if _sha(path)!=m[name]["sha256"]:raise ValueError(f"{name} checkpoint hash mismatch")
 bc=torch.load(base_path,map_location=dev,weights_only=False);schema=HierarchySchema.from_dict(bc["schema"]);base=HierarchicalClassifier(**bc["model"]);base.load_state_dict(bc["state_dict"]);base.to(dev).eval()
 rc=torch.load(res_path,map_location=dev,weights_only=False);cfg=m["residual"]["model"];residual=ResidualAdapter(**cfg);residual.load_state_dict(rc["state_dict"]);residual.to(dev).eval()
 tf={k:TfidfWeighter.load_params(str(resolve(v))) for k,v in m["tfidf"].items()};threshold=int(m["router"]["residual_if_length_lt_bp"]);alpha=float(m["router"]["alpha"]);temperatures=m.get("temperatures");thresholds=m.get("thresholds",{})
 def flush(records,writer):
  if not records:return
  seqs=[s for _,s in records];x7=featurize_block(seqs,7,np.asarray(tf["k7"].idfs,np.float32),4**7)
  with torch.inference_mode():logits=base(torch.from_numpy(x7).to(dev));short=[i for i,s in enumerate(seqs) if len(s)<threshold]
  if short:
   parts=[featurize_block([seqs[i] for i in short],k,np.asarray(tf[f"k{k}"].idfs,np.float32),4**k) for k in (4,5,6)];xr=torch.from_numpy(np.concatenate(parts,axis=1)).to(dev)
   with torch.inference_mode():
    delta=residual(xr)
    for h in HEADS:logits[h][short]=logits[h][short]+alpha*delta[h]
  probs=probabilities(logits,temperatures);short_set=set(short)
  for i,(header,seq) in enumerate(records):
   ri=int(probs["root"][i].argmax());root=schema.profile.root[ri];rp=float(probs["root"][i,ri]);leaf=root;lp=rp;branch={"euk_nuclear":"euk","prok":"prok","organelle":"organelle"}.get(root)
   if branch:li=int(probs[branch][i].argmax());leaf=schema.classes(branch)[li];lp=float(probs[branch][i,li])
   if rp<float(thresholds.get("root",0)) or (branch and lp<float(thresholds.get(branch,0))):leaf="unknown"
   writer.writerow([header,len(seq),"base+residual" if i in short_set else "base",root,leaf,f"{rp:.8f}",f"{lp:.8f}"])
 with Path(output).open("w",newline="") as f:
  w=csv.writer(f,delimiter="\t");w.writerow(["record_id","length_bp","expert","root","leaf","root_probability","leaf_probability"]);records=[];accepted=0
  for header,seq in fasta(Path(input_fasta)):
   if len(seq)<min_len:continue
   records.append((header,seq));accepted+=1
   if len(records)>=batch:flush(records,w);records=[]
   if max_records is not None and accepted>=max_records:break
  flush(records,w)
