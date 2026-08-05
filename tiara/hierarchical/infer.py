"""Batched cascade inference for v2.3.x hierarchical checkpoints."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np,torch
from tiara.src.transformations import TfidfWeighter
from tiara.training.featurize_cache import featurize_block
from .schema import HierarchySchema
from .model import HierarchicalClassifier,probabilities
from .data import fasta

def classify(checkpoint,tfidf,input_fasta,output,batch=512,device=None,thresholds=None,temperatures=None,min_len=1000):
 dev=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"));ck=torch.load(checkpoint,map_location=dev,weights_only=False);sc=HierarchySchema.from_dict(ck["schema"]);model=HierarchicalClassifier(**ck["model"]);model.load_state_dict(ck["state_dict"]);model.to(dev).eval();tf=TfidfWeighter.load_params(tfidf);thresholds=thresholds or {}; rows=[]
 def flush(items,out):
  if not items:return
  X=featurize_block([s for _,s in items],tf.k,np.asarray(tf.idfs,dtype=np.float32),4**tf.k)
  with torch.inference_mode():pr=probabilities(model(torch.from_numpy(X).to(dev)),temperatures)
  for i,(h,_) in enumerate(items):
   ri=int(pr["root"][i].argmax());root=sc.profile.root[ri];conf=float(pr["root"][i,ri]);leaf=root;leafconf=conf
   branch={"euk_nuclear":"euk","prok":"prok","organelle":"organelle"}.get(root)
   if branch:
    j=int(pr[branch][i].argmax());leaf=sc.classes(branch)[j];leafconf=float(pr[branch][i,j])
   if conf<float(thresholds.get("root",0)) or (branch and leafconf<float(thresholds.get(branch,0))):leaf="unknown"
   out.writerow([h,root,leaf,f"{conf:.8f}",f"{leafconf:.8f}"])
 with open(output,"w",newline="") as f:
  w=csv.writer(f,delimiter="\t");w.writerow(["record_id","root","leaf","root_probability","leaf_probability"]);buf=[]
  for h,s in fasta(Path(input_fasta)):
   if len(s)<min_len:continue
   buf.append((h,s))
   if len(buf)>=batch:flush(buf,w);buf=[]
  flush(buf,w)
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--tfidf",required=True);p.add_argument("-i","--input",required=True);p.add_argument("-o","--output",required=True);p.add_argument("--batch",type=int,default=512);p.add_argument("--device");p.add_argument("--min-len",type=int,default=1000);a=p.parse_args(argv);classify(a.checkpoint,a.tfidf,a.input,a.output,a.batch,a.device,min_len=a.min_len)
if __name__=="__main__":main()
