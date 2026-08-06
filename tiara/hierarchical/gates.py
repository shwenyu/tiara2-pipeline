"""Strict v2.3.1 acceptance gates."""
from __future__ import annotations
import csv,json,sqlite3,tempfile
from pathlib import Path
from .completeness import _record_id,_value
from .schema import EUK
def _gate(name,passed,**details):return {"name":name,"passed":bool(passed),"details":details}
def _score(data):
 for obj in [data]+[data[k] for k in ("summary","canonical","metrics","benchmark") if isinstance(data.get(k),dict)]:
  for key in ("four_task_average_f1","four_task_mean_f1","average_f1","mean_f1","canonical_average_f1"):
   if key in obj:
    value=float(obj[key]);return value*100 if abs(value)<=1.5 else value
 raise ValueError("benchmark JSON lacks average F1")
def audit_combined_leakage(base_metadata,delta_metadata,work_db=None):
 temporary=work_db is None
 if temporary:
  f=tempfile.NamedTemporaryFile(suffix=".sqlite",delete=False);work_db=f.name;f.close()
 p=Path(work_db)
 if p.exists():p.unlink()
 db=sqlite3.connect(p);db.executescript("CREATE TABLE ids(id TEXT PRIMARY KEY) WITHOUT ROWID;CREATE TABLE keys(kind TEXT,value TEXT,split TEXT,PRIMARY KEY(kind,value)) WITHOUT ROWID;");dups=0;over={k:0 for k in ("species_taxid","split_group_id","duplicate_cluster_id")};examples={k:[] for k in over};rows=0
 try:
  for path in (base_metadata,delta_metadata):
   with open(path,newline="",encoding="utf-8",errors="replace") as f:
    for row in csv.DictReader(f,delimiter="\t"):
     rows+=1;record=_record_id(row);split=_value(row,"split")
     try:db.execute("INSERT INTO ids VALUES(?)",(record,))
     except sqlite3.IntegrityError:dups+=1
     for kind in over:
      value=_value(row,kind)
      if not value:continue
      prev=db.execute("SELECT split FROM keys WHERE kind=? AND value=?",(kind,value)).fetchone()
      if prev is None:db.execute("INSERT INTO keys VALUES(?,?,?)",(kind,value,split))
      elif prev[0]!=split:over[kind]+=1
  db.commit()
 finally:
  db.close()
  if temporary:
   for suffix in ("","-wal","-shm"):
    try:Path(str(p)+suffix).unlink()
    except FileNotFoundError:pass
 return {"rows":rows,"duplicate_record_ids":dups,"overlap_counts":over,"examples":examples,"ok":dups==0 and not any(over.values())}
def run_gates(completeness_report,composite_features,evaluation_json,base_metadata,delta_metadata,*,baseline_benchmark=None,current_benchmark=None,checkpoint=None,max_legacy_regression_pp=.20,min_leaf_recall=.05,out_path=None):
 c=json.loads(Path(completeness_report).read_text());f=json.loads(Path(composite_features).read_text());e=json.loads(Path(evaluation_json).read_text());gates=[];unresolved=int(c.get("rows",{}).get("unresolved_euk",-1));gates.append(_gate("all_euk_labels_resolved",unresolved==0,unresolved=unresolved));combined=f.get("combined_euk_counts",{});missing={s:[g for g in EUK if int(combined.get(s,{}).get(g,0))==0] for s in ("train","validation")};gates.append(_gate("eight_euk_leaves_nonzero_train_validation",not missing["train"] and not missing["validation"],missing=missing,counts=combined));expected={"hidden":[2048,1024],"dropout":.2,"learning_rate":.001,"epochs":50,"batch_size":1024,"optimizer":"AdamW","sampler":"v2.3.0 root WeightedRandomSampler","checkpoint_selector":"root_macro_f1"};contract=f.get("frozen_training_contract",{});drift={k:{"expected":v,"actual":contract.get(k)} for k,v in expected.items() if contract.get(k)!=v};gates.append(_gate("v230_training_contract_frozen",int(f.get("k",-1))==7 and not drift,k=f.get("k"),drift=drift));leak=audit_combined_leakage(base_metadata,delta_metadata);gates.append(_gate("no_species_split_dedup_leakage",leak["ok"],**leak));per=e.get("euk",{}).get("per_class",{});unsupported=[x for x in EUK if int(per.get(x,{}).get("support",0))==0];near=[x for x in EUK if float(per.get(x,{}).get("recall",0))<min_leaf_recall];gates.append(_gate("euk_leaf_metrics_interpretable",not unsupported and not near,unsupported=unsupported,recall_below_threshold=near));cascade=e.get("cascade",{});gates.append(_gate("cascade_accuracy_reported",int(cascade.get("records",0))>0 and "accuracy" in cascade,cascade=cascade))
 if baseline_benchmark and current_benchmark:
  base=_score(json.loads(Path(baseline_benchmark).read_text()));cur=_score(json.loads(Path(current_benchmark).read_text()));delta=cur-base;gates.append(_gate("canonical_benchmark_noninferior",delta>=-max_legacy_regression_pp,baseline_percent=base,current_percent=cur,delta_pp=delta))
 else:gates.append(_gate("canonical_benchmark_noninferior",False,error="benchmark JSON missing"))
 if checkpoint:
  import torch
  ck=torch.load(checkpoint,map_location="cpu",weights_only=False);m=ck.get("model",{});ok=ck.get("version")=="2.3.1" and m.get("hidden")==[2048,1024] and float(m.get("dropout",-1))==.2 and ck.get("checkpoint_selector")=="root_macro_f1";gates.append(_gate("checkpoint_contract",ok,version=ck.get("version"),model=m))
 else:gates.append(_gate("checkpoint_contract",False,error="checkpoint missing"))
 result={"version":"2.3.1","passed":all(x["passed"] for x in gates),"passed_count":sum(x["passed"] for x in gates),"total_count":len(gates),"gates":gates}
 if out_path:Path(out_path).parent.mkdir(parents=True,exist_ok=True);Path(out_path).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 return result
__all__=["run_gates","audit_combined_leakage"]
