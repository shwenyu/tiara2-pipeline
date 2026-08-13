#!/usr/bin/env python3
"""Grouped-OOF search for conservative head-specific local corrections."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
HEADS=("root","euk","prok","organelle")
SIZES={"root":3,"euk":8,"prok":2,"organelle":2}
ALPHAS=(0.0625,0.125,0.25,0.5,1.0)

def softmax(x):
    y=x-x.max(1,keepdims=True);e=np.exp(y);return e/e.sum(1,keepdims=True)
def macro_f1(y,p,n):
    cm=np.bincount(y*n+p,minlength=n*n).reshape(n,n);tp=np.diag(cm);den=2*tp+cm.sum(0)-tp+cm.sum(1)-tp;return float(np.mean(np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)))
def rules(features):
    yield "all",np.ones(len(features["entropy"]),bool)
    yield "disagree",features["disagree"]
    for q in (.5,.7,.8,.9,.95):
        t=np.quantile(features["entropy"],q);yield f"entropy>={q}",features["entropy"]>=t
        t=np.quantile(features["resnorm"],q);yield f"resnorm>={q}",features["resnorm"]>=t
    for q1 in (.7,.8,.9):
        e=np.quantile(features["entropy"],q1)
        for q2 in (.5,.7,.8):
            r=np.quantile(features["resnorm"],q2);yield f"entropy>={q1}&resnorm>={q2}",(features["entropy"]>=e)&(features["resnorm"]>=r)
def fold_id(groups,k=5):return np.asarray([(int(x)*2654435761)%k for x in groups],dtype=np.int8)
def evaluate_choice(base,resid,y,active,alpha,n):
    pred=base.argmax(1);hp=(base+alpha*resid).argmax(1);final=np.where(active,hp,pred);bc=pred==y;hc=final==y
    return {"macro_f1":macro_f1(y,final,n),"base_macro_f1":macro_f1(y,pred,n),"base_only_correct":int(np.sum(bc&~hc)),"hybrid_only_correct":int(np.sum(~bc&hc)),"activation_rate":float(active.mean())}
def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",required=True);p.add_argument("--base-logits",required=True);p.add_argument("--residual-logits",required=True);p.add_argument("--groups",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    cache=json.loads((Path(a.cache)/"sequence_cache.json").read_text());spec=cache["splits"]["validation"];nrows=int(spec["rows"]);length=np.memmap(spec["lengths"],dtype=np.int16,mode="r",shape=(nrows,));groups=np.load(a.groups,mmap_mode="r");folds=fold_id(groups);schema=cache["schema"];root=np.memmap(spec["labels"]["root"],dtype=np.int64,mode="r",shape=(nrows,));root_names=schema["root"]
    report={"format":"tiara2-v250-grouped-oof-local-router-audit-v1","folds":5,"alphas":list(ALPHAS),"heads":{}}
    for h in HEADS:
        size=SIZES[h];y=np.memmap(spec["labels"][h],dtype=np.int64,mode="r",shape=(nrows,));base=np.memmap(Path(a.base_logits)/"validation"/f"{h}.f32",dtype=np.float32,mode="r",shape=(nrows,size));resid=np.memmap(Path(a.residual_logits)/f"{h}.f32",dtype=np.float32,mode="r",shape=(nrows,size))
        applicable=np.ones(nrows,bool) if h=="root" else root==root_names.index("euk_nuclear" if h=="euk" else h)
        ids=np.flatnonzero(applicable);b=np.asarray(base[ids]);r=np.asarray(resid[ids]);target=np.asarray(y[ids]);probs=softmax(b);sortedp=np.sort(probs,axis=1);feat={"entropy":-(probs*np.log(probs+1e-12)).sum(1),"margin":sortedp[:,-1]-sortedp[:,-2],"resnorm":np.linalg.norm(r,axis=1),"disagree":b.argmax(1)!=(b+r).argmax(1),"length":np.asarray(length[ids])};ff=folds[ids];oof=np.zeros(len(ids),dtype=np.int64);active_all=np.zeros(len(ids),bool);fold_reports=[]
        for f in range(5):
            tr=ff!=f;te=ff==f;best=None
            train_feat={k:v[tr] for k,v in feat.items()}
            for alpha in ALPHAS:
                for name,act in rules(train_feat):
                    ev=evaluate_choice(b[tr],r[tr],target[tr],act,alpha,size);score=ev["macro_f1"]-2*max(0,ev["base_only_correct"]-ev["hybrid_only_correct"])/max(1,tr.sum())
                    if best is None or score>best[0]:best=(score,alpha,name,ev)
            _,alpha,name,train_ev=best
            # Recreate the chosen quantile rule using train thresholds, then apply to held-out rows.
            if name=="all":act=np.ones(te.sum(),bool)
            elif name=="disagree":act=feat["disagree"][te]
            else:
                parts=name.split("&");act=np.ones(te.sum(),bool)
                for part in parts:
                    key,q=part.split(">=");threshold=np.quantile(train_feat[key],float(q));act&=feat[key][te]>=threshold
            bp=b[te].argmax(1);hp=(b[te]+alpha*r[te]).argmax(1);oof[te]=np.where(act,hp,bp);active_all[te]=act
            ev={"macro_f1":macro_f1(target[te],oof[te],size),"base_macro_f1":macro_f1(target[te],bp,size),"base_only_correct":int(np.sum((bp==target[te])&(oof[te]!=target[te]))),"hybrid_only_correct":int(np.sum((bp!=target[te])&(oof[te]==target[te]))),"activation_rate":float(act.mean())};fold_reports.append({"fold":f,"alpha":alpha,"rule":name,"train":train_ev,"heldout":ev})
        bp=b.argmax(1);overall={"base_macro_f1":macro_f1(target,bp,size),"oof_macro_f1":macro_f1(target,oof,size),"delta_macro_f1":macro_f1(target,oof,size)-macro_f1(target,bp,size),"base_only_correct":int(np.sum((bp==target)&(oof!=target))),"hybrid_only_correct":int(np.sum((bp!=target)&(oof==target))),"activation_rate":float(active_all.mean()),"rows":len(ids)}
        report["heads"][h]={"overall":overall,"folds":fold_reports}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/"local_router_audit.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({h:v["overall"] for h,v in report["heads"].items()},indent=2,sort_keys=True))
if __name__=="__main__":main()
