"""End-to-end root/leaf metrics and viral-to-euk false-positive rate."""
from __future__ import annotations
import csv,json
from collections import defaultdict
def _f1(truth,pred):
 classes=sorted(set(truth)|set(pred));vals={}
 for c in classes:
  tp=sum(a==c and b==c for a,b in zip(truth,pred));fp=sum(a!=c and b==c for a,b in zip(truth,pred));fn=sum(a==c and b!=c for a,b in zip(truth,pred));vals[c]=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
 return {"macro_f1":sum(vals.values())/len(vals) if vals else 0.,"per_class_f1":vals}
def evaluate(truth_tsv,pred_tsv):
 def load(p):
  with open(p,newline="") as f:return {r["record_id"]:r for r in csv.DictReader(f,delimiter="\t")}
 t,p=load(truth_tsv),load(pred_tsv);ids=sorted(set(t)&set(p));out={"records":len(ids)}
 for level in ("root","leaf"):out[level]=_f1([t[i][level] for i in ids],[p[i][level] for i in ids])
 virus=[i for i in ids if t[i]["root"]=="virus"];out["viral_to_euk_fpr"]=sum(p[i]["root"]=="euk_nuclear" for i in virus)/len(virus) if virus else None;return out
