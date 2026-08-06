"""Reuse v2.3.0 base features and featurize only the v2.3.1 Euk delta."""
from __future__ import annotations
import csv,json,sqlite3
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .completeness import _is_euk_row,_normal_group,_record_id,_value
from .data import FILES,fasta,rid
from .freeze import sha256_file,verify_freeze
from .schema import EUK,schema
HEADS=("root","euk","prok","organelle")
def _now():return datetime.now(timezone.utc).isoformat()
def _path(root,split,manifest,head):
 value=manifest.get("files",{}).get(head);candidate=Path(value) if value else root/split/("X.f32" if head=="X" else f"{head}.i64")
 if candidate.is_file():return candidate.resolve()
 if not candidate.is_absolute() and (root/candidate).is_file():return (root/candidate).resolve()
 fallback=root/split/("X.f32" if head=="X" else f"{head}.i64")
 if fallback.is_file():return fallback.resolve()
 raise FileNotFoundError(candidate)
def _label_db(path,metadata):
 if path.exists():path.unlink()
 db=sqlite3.connect(path);db.execute("CREATE TABLE labels(record_id TEXT PRIMARY KEY,split TEXT,euk_group TEXT) WITHOUT ROWID");n=0
 with open(metadata,newline="",encoding="utf-8",errors="replace") as f:
  for row in csv.DictReader(f,delimiter="\t"):
   if not _is_euk_row(row):continue
   record=_record_id(row);split=_value(row,"split");group=_normal_group(_value(row,"euk_group"))
   if not record or split not in {"train","validation"} or not group:raise ValueError(f"invalid curated Euk row {record!r}")
   try:db.execute("INSERT INTO labels VALUES(?,?,?)",(record,split,group))
   except sqlite3.IntegrityError as exc:raise ValueError(f"duplicate Euk metadata ID {record}") from exc
   n+=1
 db.commit();return db,{"indexed_euk_rows":n}
def _class_n(freeze,root,split,cls,name):
 entry=freeze.get("splits",{}).get(split,{}).get(cls) if freeze else None
 if entry:return int(entry["records"])
 p=root/split/name;return sum(1 for _ in fasta(p)) if p.is_file() else 0
def _relabel(base_root,base_manifest,train_ready,db,out,freeze):
 shards={};stats={};idx=schema("2.3.1").index("euk")
 for split in ("train","validation"):
  manifest=base_manifest["splits"][split];rows=int(manifest["rows"]);dim=int(manifest["dim"]);d=out/"base_label_overrides"/split;d.mkdir(parents=True,exist_ok=True);label_path=d/"euk.i64";labels=np.memmap(label_path,"int64","w+",shape=(rows,));labels[:]=-1;pos=0;counts=Counter();found=0
  for cls,name in FILES.items():
   if cls=="virus":continue
   n=_class_n(freeze,train_ready,split,cls,name)
   if cls!="eukarya":pos+=n;continue
   seen=0
   for header,_ in fasta(train_ready/split/name):
    record=rid(header);row=db.execute("SELECT split,euk_group FROM labels WHERE record_id=?",(record,)).fetchone()
    if row is None or row[0]!=split:raise ValueError(f"curated metadata missing/split mismatch {record}")
    labels[pos]=idx[row[1]];counts[row[1]]+=1;pos+=1;seen+=1;found+=1
   if seen!=n:raise ValueError(f"freeze row mismatch {split}/eukarya: {n} vs {seen}")
  if pos!=rows:raise ValueError(f"base feature order mismatch {split}: {pos} vs {rows}")
  labels.flush();del labels
  files={"X":str(_path(base_root,split,manifest,"X")),"root":str(_path(base_root,split,manifest,"root")),"euk":str(label_path.resolve()),"prok":str(_path(base_root,split,manifest,"prok")),"organelle":str(_path(base_root,split,manifest,"organelle"))};shards[split]={"name":"frozen_base_v2.3.0","rows":rows,"dim":dim,"files":files,"immutable_features":True,"euk_labels_rebuilt_only":True};stats[split]={"rows":rows,"resolved_euk_rows":found,"euk_counts":{g:counts[g] for g in EUK},"euk_label_sha256":sha256_file(label_path)}
 return shards,stats
def _delta_meta(path):
 out={};counts=Counter()
 with open(path,newline="",encoding="utf-8",errors="replace") as f:
  for row in csv.DictReader(f,delimiter="\t"):
   record=_record_id(row);split=_value(row,"split");group=_normal_group(_value(row,"euk_group"))
   if not record or split not in {"train","validation"} or not group or record in out:raise ValueError(f"invalid/duplicate delta row {record!r}")
   out[record]=(split,group);counts[(split,group)]+=1
 return out,counts
def _prepare_delta(delta_root,delta_metadata,tfidf,out,chunk):
 from tiara.src.transformations import TfidfWeighter
 from tiara.training.featurize_cache import featurize_block
 records,expected=_delta_meta(delta_metadata);tf=TfidfWeighter.load_params(str(tfidf));k=int(tf.k)
 if k!=7:raise ValueError(f"v2.3.1 freezes k=7, got {k}")
 dim=4**k;idf=np.asarray(tf.idfs,dtype=np.float32);sc=schema("2.3.1");root_value=sc.index("root")["euk_nuclear"];eidx=sc.index("euk");shards={};stats={}
 for split in ("train","validation"):
  path=delta_root/split/"eukarya.fasta"
  if not path.is_file():raise FileNotFoundError(path)
  n=sum(expected[(split,g)] for g in EUK);d=out/"delta_features"/split;d.mkdir(parents=True,exist_ok=True);id_path=d/"record_ids.txt";files={"X":str((d/"X.f32").resolve()),**{h:str((d/f"{h}.i64").resolve()) for h in HEADS}}
  if n==0:
   for name in ("X.f32",*(f"{h}.i64" for h in HEADS),"record_ids.txt"):(d/name).touch()
   shard={"name":"euk_delta_v2.3.1","rows":0,"dim":dim,"files":files,"record_ids":str(id_path.resolve())};(d/"manifest.json").write_text(json.dumps(shard,indent=2)+"\n");shards[split]=shard;stats[split]={"rows":0,"euk_counts":{g:0 for g in EUK},"feature_sha256":sha256_file(d/"X.f32")};continue
  X=np.memmap(d/"X.f32","float32","w+",shape=(n,dim));ys={h:np.memmap(d/f"{h}.i64","int64","w+",shape=(n,)) for h in HEADS};ys["root"][:]=root_value;ys["euk"][:]=-1;ys["prok"][:]=-1;ys["organelle"][:]=-1;pos=0;counts=Counter();seqs=[];labs=[];ids=[]
  def flush(handle):
   nonlocal pos,seqs,labs,ids
   if not seqs:return
   z=featurize_block(seqs,k,idf,dim);end=pos+len(seqs);X[pos:end]=z;ys["euk"][pos:end]=np.asarray(labs,dtype=np.int64)
   for x in ids:handle.write(x+"\n")
   pos=end;seqs=[];labs=[];ids=[]
  with id_path.open("w") as handle:
   for header,seq in fasta(path):
    record=rid(header);meta=records.get(record)
    if meta is None or meta[0]!=split:raise ValueError(f"delta metadata missing/split mismatch {record}")
    seqs.append(seq);labs.append(eidx[meta[1]]);ids.append(record);counts[meta[1]]+=1
    if len(seqs)>=chunk:flush(handle)
   flush(handle)
  if pos!=n:raise ValueError(f"delta count mismatch {split}: {n} vs {pos}")
  X.flush();[v.flush() for v in ys.values()];del X,ys;shard={"name":"euk_delta_v2.3.1","rows":n,"dim":dim,"files":files,"record_ids":str(id_path.resolve())};(d/"manifest.json").write_text(json.dumps(shard,indent=2)+"\n");shards[split]=shard;stats[split]={"rows":n,"euk_counts":{g:counts[g] for g in EUK},"feature_sha256":sha256_file(d/"X.f32")}
 return shards,stats
def prepare_v231_features(base_features,base_train_ready,curated_base_metadata,delta_root,delta_metadata,tfidf,out_dir,*,base_freeze_manifest=None,verify_base="full",chunk=2048):
 base_root=Path(base_features);base_manifest_path=base_root/"hierarchy_features.json";base_manifest=json.loads(base_manifest_path.read_text());sc=schema("2.3.1")
 if int(base_manifest.get("k",-1))!=7:raise ValueError("v2.3.1 requires k7 base cache")
 for h in HEADS:
  if tuple(base_manifest["schema"][h])!=tuple(sc.classes(h)):raise ValueError(f"schema drift {h}")
 model_sha=sha256_file(Path(tfidf)/"model.npy")
 if base_manifest.get("tfidf_sha256")!=model_sha:raise ValueError("TF-IDF differs from base feature manifest")
 freeze=None;verification={"ok":None,"mode":"not_requested"}
 if base_freeze_manifest:
  verification=verify_freeze(base_freeze_manifest,verify_base)
  if not verification["ok"]:raise ValueError("base freeze failed")
  freeze=json.loads(Path(base_freeze_manifest).read_text())
 out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);dbpath=out/".labels.sqlite";db,index_stats=_label_db(dbpath,curated_base_metadata)
 try:base_shards,base_stats=_relabel(base_root,base_manifest,Path(base_train_ready),db,out,freeze)
 finally:
  db.close()
  for suffix in ("","-wal","-shm"):
   try:Path(str(dbpath)+suffix).unlink()
   except FileNotFoundError:pass
 delta_shards,delta_stats=_prepare_delta(Path(delta_root),delta_metadata,tfidf,out,chunk);splits={s:{"rows":base_shards[s]["rows"]+delta_shards[s]["rows"],"dim":base_shards[s]["dim"],"shards":[base_shards[s]]+([delta_shards[s]] if delta_shards[s]["rows"] else [])} for s in ("train","validation")};combined={s:{g:base_stats[s]["euk_counts"][g]+delta_stats[s]["euk_counts"][g] for g in EUK} for s in ("train","validation")};missing={s:[g for g in EUK if combined[s][g]==0] for s in combined}
 manifest={"schema_version":1,"version":"2.3.1","format":"tiara-hierarchical-composite-features","generated_at":_now(),"schema":sc.to_dict(),"k":7,"tfidf":str(Path(tfidf).resolve()),"tfidf_sha256":model_sha,"frozen_training_contract":{"hidden":[2048,1024],"dropout":.2,"learning_rate":.001,"epochs":50,"batch_size":1024,"optimizer":"AdamW","sampler":"v2.3.0 root WeightedRandomSampler","checkpoint_selector":"root_macro_f1"},"base":{"feature_manifest":str(base_manifest_path.resolve()),"freeze_manifest":base_freeze_manifest,"verification":verification,"label_index":index_stats,"stats":base_stats},"delta":{"root":str(Path(delta_root).resolve()),"metadata":str(Path(delta_metadata).resolve()),"stats":delta_stats},"splits":splits,"combined_euk_counts":combined,"missing_euk_classes":missing,"ready":not missing["train"] and not missing["validation"]};(out/"composite_features.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");return manifest
__all__=["prepare_v231_features"]
