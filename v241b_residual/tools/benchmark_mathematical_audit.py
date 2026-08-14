#!/usr/bin/env python3
"""Run M1-M5 on the 16 frozen binary benchmark per-contig cells."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp,softmax
from sklearn.metrics import average_precision_score

def binary_logits(z):return np.column_stack((logsumexp(z[:,1:],axis=1),z[:,0]))
def nll(z,y):return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]))
def best_f1(prob,eligible,y):
 score=np.where(eligible,prob,-1.0);order=np.argsort(-score);yy=y[order].astype(np.int64);tp=np.cumsum(yy);fp=np.cumsum(1-yy);pos=int(tp[-1]);f1=2*tp/np.maximum(1,2*tp+fp+(pos-tp));i=int(np.argmax(f1));pred=eligible&(prob>=score[order[i]])
 return float(f1[i]),float(score[order[i]]),pred
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--alpha-min",type=float,default=-.25);p.add_argument("--alpha-max",type=float,default=1.0);p.add_argument("--alpha-steps",type=int,default=51);a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);manifest=json.load((root/"benchmark_manifest.json").open());alphas=np.linspace(a.alpha_min,a.alpha_max,a.alpha_steps);report={"schema_version":"tiara2-benchmark-m1-m5-v1","source_manifest_sha256":sha(root/"benchmark_manifest.json"),"scope":"external_benchmark_diagnostic_no_selection","cells":{}}
 for name,meta in sorted(manifest["cells"].items()):
  path=root/meta["file"]
  if sha(path)!=meta["sha256"]:raise ValueError(f"hash mismatch {name}")
  with np.load(path,allow_pickle=False) as d:base=d["base_logits"].astype(np.float64);res=d["residual_logits"].astype(np.float64);y=d["y_true"].astype(np.int64);length=int(d["length_bp"][0]);lineage=d["lineage_id"]
  surface=[];preds={}
  for alpha in alphas:
   z=base+alpha*res;b=binary_logits(z);prob=softmax(z,axis=1)[:,0];eligible=z.argmax(1)==0;f1,t,pred=best_f1(prob,eligible,y);surface.append({"alpha":float(alpha),"binary_nll":nll(b,y),"best_f1":f1,"threshold":t,"auprc_raw":float(average_precision_score(y,prob)),"auprc_decision_score":float(average_precision_score(y,np.where(eligible,prob,0.0)))})
   if abs(alpha-.125)<1e-9:preds["candidate"]=pred
  b0=binary_logits(base);p0=softmax(base,axis=1)[:,0];f0,t0,pred0=best_f1(p0,base.argmax(1)==0,y);preds["base"]=pred0
  zc=base+.125*res;bc=binary_logits(zc);pc=softmax(zc,axis=1)[:,0];fc,tc,predc=best_f1(pc,zc.argmax(1)==0,y);preds["candidate"]=predc
  br=pred0==y;cr=predc==y;dis=float(np.mean(pred0!=predc));q={"both_correct":int(np.sum(br&cr)),"base_only":int(np.sum(br&~cr)),"candidate_only":int(np.sum(~br&cr)),"both_wrong":int(np.sum(~br&~cr))}
  bl=logsumexp(b0,axis=1)-b0[np.arange(len(y)),y];cl=logsumexp(bc,axis=1)-bc[np.arange(len(y)),y]
  opt=minimize_scalar(lambda beta:nll(b0*beta,y),bounds=(.05,10),method="bounded");T=1/opt.x;cal=softmax(b0/T,axis=1)[:,1];cf,ct,_=best_f1(cal,base.argmax(1)==0,y)
  by_lineage={}
  for g in np.unique(lineage):
   m=lineage==g;by_lineage[str(g)]={"N":int(m.sum()),"positive":int(y[m].sum()),"base_correct":int(br[m].sum()),"candidate_correct":int(cr[m].sum()),"base_only":int(np.sum(br[m]&~cr[m])),"candidate_only":int(np.sum(~br[m]&cr[m]))}
  report["cells"][name]={"file":meta["file"],"sha256":meta["sha256"],"N":len(y),"positive":int(y.sum()),"length_bp":length,"disagreement":dis,"delta_min_accuracy":float(2.8*np.sqrt(dis/len(y))),"base":{"best_f1":f0,"threshold":t0,"binary_nll":nll(b0,y),"auprc_raw":float(average_precision_score(y,p0)),"auprc_decision_score":float(average_precision_score(y,np.where(base.argmax(1)==0,p0,0.0)))},"candidate":{"best_f1":fc,"threshold":tc,"binary_nll":nll(bc,y),"auprc_raw":float(average_precision_score(y,pc)),"auprc_decision_score":float(average_precision_score(y,np.where(zc.argmax(1)==0,pc,0.0)))},"quadrants":q,"oracle_accuracy":float(1-q["both_wrong"]/len(y)),"loss_correlation":float(np.corrcoef(bl,cl)[0,1]),"sigma_base_over_candidate":float(np.std(bl)/max(1e-12,np.std(cl))),"alpha_surface":surface,"base_calibration":{"T_star":float(T),"calibrated_nll":float(opt.fun),"best_f1":cf,"empirical_threshold":ct,"f1_half_diagnostic":cf/2},"lineages":by_lineage}
 (out/"benchmark_m1_m5.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 rows=[]
 for name,v in report["cells"].items():
  bn=min(v["alpha_surface"],key=lambda x:x["binary_nll"]);bf=max(v["alpha_surface"],key=lambda x:x["best_f1"]);q=v["quadrants"]
  rows.append({"cell":name,"N":v["N"],"positive":v["positive"],"disagreement":v["disagreement"],"delta_min_accuracy_pp":100*v["delta_min_accuracy"],"base_f1":v["base"]["best_f1"],"candidate_f1":v["candidate"]["best_f1"],"delta_f1_pp":100*(v["candidate"]["best_f1"]-v["base"]["best_f1"]),"base_auprc_raw":v["base"]["auprc_raw"],"candidate_auprc_raw":v["candidate"]["auprc_raw"],"delta_auprc_raw_pp":100*(v["candidate"]["auprc_raw"]-v["base"]["auprc_raw"]),"base_auprc_decision":v["base"]["auprc_decision_score"],"candidate_auprc_decision":v["candidate"]["auprc_decision_score"],"delta_auprc_decision_pp":100*(v["candidate"]["auprc_decision_score"]-v["base"]["auprc_decision_score"]),"base_only":q["base_only"],"candidate_only":q["candidate_only"],"oracle_accuracy":v["oracle_accuracy"],"best_alpha_nll":bn["alpha"],"best_alpha_f1":bf["alpha"],"T_star":v["base_calibration"]["T_star"],"loss_correlation":v["loss_correlation"]})
 with (out/"benchmark_m1_m5_summary.tsv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),delimiter="\t");w.writeheader();w.writerows(rows)
 print(json.dumps({"out":str(out),"cells":len(rows),"json_sha256":sha(out/"benchmark_m1_m5.json"),"summary_sha256":sha(out/"benchmark_m1_m5_summary.tsv")},indent=2))
if __name__=="__main__":main()
