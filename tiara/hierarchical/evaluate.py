"""Root/leaf metrics, confusion matrices and complete cascade accuracy."""
from __future__ import annotations
import csv,json
from collections import Counter
from pathlib import Path
from .completeness import _normal_group,_record_id,_value
from .schema import EUK,PROK,ORGANELLE

def _truth(row):
 if row.get("root") and row.get("leaf"):return row["root"],row["leaf"]
 legacy=_value(row,"legacy_class").lower()
 if legacy in {"euk","eukarya","eukaryota","euk_nuclear"}:
  group=_normal_group(_value(row,"euk_group"))
  if not group:raise ValueError(f"truth Euk row {_record_id(row)!r} has no valid euk_group")
  return "euk_nuclear",group
 if legacy in {"bacteria","archaea"}:return "prok",legacy
 if legacy in {"mitochondria","mitochondrion","mito"}:return "organelle","mitochondria"
 if legacy in {"plastid","plastids","chloroplast"}:return "organelle","plastid"
 if legacy in {"virus","viral"}:return "virus","virus"
 raise ValueError(f"cannot derive truth for {_record_id(row)!r}")
def _load(path,pred=False):
 out={}
 with open(path,newline="",encoding="utf-8",errors="replace") as f:
  for row in csv.DictReader(f,delimiter="\t"):
   record=row.get("record_id","").strip() if pred else _record_id(row)
   if not record or record in out:raise ValueError(f"missing/duplicate record_id: {record!r}")
   if pred:
    if not row.get("root") or not row.get("leaf"):raise ValueError("prediction TSV requires root and leaf")
    out[record]=(row["root"],row["leaf"])
   else:out[record]=_truth(row)
 return out
def _metrics(truth,pred,expected):
 extra=sorted((set(truth)|set(pred))-set(expected));order=list(expected)+extra;idx={x:i for i,x in enumerate(order)};m=[[0]*len(order) for _ in order]
 for a,b in zip(truth,pred):m[idx[a]][idx[b]]+=1
 per={};vals=[]
 for name in expected:
  i=idx[name];tp=m[i][i];fp=sum(m[r][i] for r in range(len(order)) if r!=i);fn=sum(m[i][c] for c in range(len(order)) if c!=i);support=sum(m[i]);p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.;f=2*p*r/(p+r) if p+r else 0.;per[name]={"precision":p,"recall":r,"f1":f,"support":support};vals.append(f)
 total=len(truth);correct=sum(m[idx[x]][idx[x]] for x in expected)
 return {"records":total,"accuracy":correct/total if total else 0.,"macro_f1":sum(vals)/len(vals) if vals else 0.,"per_class":per,"class_order":order,"confusion_matrix":m,"unexpected_labels":extra}
def evaluate(truth_tsv,pred_tsv,out_path=None):
 t=_load(truth_tsv);p=_load(pred_tsv,True);ids=sorted(set(t)&set(p))
 if not ids:raise ValueError("truth and predictions share no record_id")
 rt=[t[i][0] for i in ids];rp=[p[i][0] for i in ids];root_classes=tuple(x for x in ("euk_nuclear","prok","organelle","virus") if x in set(rt));out={"records":len(ids),"truth_records":len(t),"prediction_records":len(p),"missing_predictions":len(set(t)-set(p)),"predictions_without_truth":len(set(p)-set(t)),"root":_metrics(rt,rp,root_classes)}
 for head,(root,classes) in {"euk":("euk_nuclear",EUK),"prok":("prok",PROK),"organelle":("organelle",ORGANELLE)}.items():
  sub=[i for i in ids if t[i][0]==root];out[head]=_metrics([t[i][1] for i in sub],[p[i][1] for i in sub],classes)
 correct=sum(t[i]==p[i] for i in ids);out["cascade"]={"correct":correct,"records":len(ids),"accuracy":correct/len(ids)};out["truth_root_counts"]=dict(Counter(rt));out["prediction_root_counts"]=dict(Counter(rp))
 if out_path:Path(out_path).parent.mkdir(parents=True,exist_ok=True);Path(out_path).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 return out
__all__=["evaluate"]
