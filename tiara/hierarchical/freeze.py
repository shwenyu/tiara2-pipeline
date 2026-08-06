"""Freeze and verify the immutable v2.3.0 base contract."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from .data import FILES,fasta,rid

def sha256_file(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def _now():return datetime.now(timezone.utc).isoformat()
def _regular(path):
 p=Path(path);s=p.stat();return {"path":str(p.resolve()),"bytes":s.st_size,"mtime_ns":s.st_mtime_ns,"sha256":sha256_file(p)}
def _fasta(path):
 p=Path(path);ids=hashlib.sha256();n=bp=0
 for header,seq in fasta(p):ids.update(rid(header).encode());ids.update(b"\n");n+=1;bp+=len(seq)
 out=_regular(p);out.update({"records":n,"bp":bp,"ordered_record_id_sha256":ids.hexdigest()});return out
def freeze_base(train_ready,tfidf,metadata_tsv,out_path):
 root=Path(train_ready);meta=Path(metadata_tsv);tf=Path(tfidf)
 if not root.is_dir():raise FileNotFoundError(root)
 if not meta.is_file():raise FileNotFoundError(meta)
 for p in (tf/"model.npy",tf/"params.txt"):
  if not p.is_file():raise FileNotFoundError(p)
 splits={}
 for split in ("train","validation","test"):
  if not (root/split).is_dir():continue
  classes={}
  for cls,name in FILES.items():
   if cls=="virus":continue
   p=root/split/name
   if p.is_file():classes[cls]=_fasta(p)
  splits[split]=classes
 manifest={"schema_version":1,"contract":"tiara2-v2.3.0-frozen-base","generated_at":_now(),"immutable":True,"train_ready":str(root.resolve()),"splits":splits,"metadata":_regular(meta),"tfidf":{"root":str(tf.resolve()),"model":_regular(tf/"model.npy"),"params":_regular(tf/"params.txt")}}
 out=Path(out_path);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");return manifest
def verify_freeze(manifest_path,mode="quick"):
 if mode not in {"quick","full"}:raise ValueError("mode must be quick or full")
 m=json.loads(Path(manifest_path).read_text());errors=[];checked=0
 def check(entry,label):
  nonlocal checked
  p=Path(entry["path"]);checked+=1
  if not p.is_file():errors.append(f"missing {label}: {p}");return False
  if p.stat().st_size!=int(entry["bytes"]):errors.append(f"size changed for {label}: {p}");return False
  if mode=="full" and sha256_file(p)!=entry["sha256"]:errors.append(f"sha256 changed for {label}: {p}");return False
  return True
 for split,classes in m.get("splits",{}).items():
  for cls,entry in classes.items():
   if check(entry,f"{split}/{cls}") and mode=="full":
    ids=hashlib.sha256();n=0
    for h,_ in fasta(entry["path"]):ids.update(rid(h).encode());ids.update(b"\n");n+=1
    if n!=int(entry["records"]):errors.append(f"record count changed for {split}/{cls}")
    if ids.hexdigest()!=entry["ordered_record_id_sha256"]:errors.append(f"record order/IDs changed for {split}/{cls}")
 check(m["metadata"],"metadata");check(m["tfidf"]["model"],"tfidf/model.npy");check(m["tfidf"]["params"],"tfidf/params.txt")
 return {"ok":not errors,"mode":mode,"checked_files":checked,"errors":errors}
__all__=["freeze_base","verify_freeze","sha256_file"]
