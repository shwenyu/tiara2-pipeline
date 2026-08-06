"""Build an Euk-only v2.3.1 delta without mutating the frozen base."""
from __future__ import annotations
import csv,hashlib,json,sqlite3
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from .completeness import _accession,_normal_group,_record_id,_value
from .data import FILES,fasta,rid
from .freeze import sha256_file,verify_freeze
from .schema import EUK

def _now():return datetime.now(timezone.utc).isoformat()
def _seq_digest(seq):return hashlib.blake2b(seq.upper().encode("ascii","ignore"),digest_size=16).digest()
def _db(path):
 db=sqlite3.connect(path);db.execute("PRAGMA journal_mode=WAL");db.execute("PRAGMA synchronous=NORMAL");db.executescript("""
 CREATE TABLE base_ids(record_id TEXT PRIMARY KEY) WITHOUT ROWID;
 CREATE TABLE sequence_hashes(digest BLOB PRIMARY KEY) WITHOUT ROWID;
 CREATE TABLE split_keys(kind TEXT,value TEXT,split TEXT,PRIMARY KEY(kind,value)) WITHOUT ROWID;
 CREATE TABLE candidate_meta(record_id TEXT PRIMARY KEY,accession TEXT,split TEXT,euk_group TEXT,species_taxid TEXT,split_group_id TEXT,duplicate_cluster_id TEXT) WITHOUT ROWID;
 """);return db
def _split_key(db,kind,value,split,strict):
 if not value or not split:return True
 row=db.execute("SELECT split FROM split_keys WHERE kind=? AND value=?",(kind,value)).fetchone()
 if row is None:db.execute("INSERT INTO split_keys VALUES(?,?,?)",(kind,value,split));return True
 if row[0]==split:return True
 if strict:raise ValueError(f"base leaks {kind}={value} across {row[0]} and {split}")
 return False
def _index_base_meta(db,path):
 n=euk=0
 with open(path,newline="",encoding="utf-8",errors="replace") as f:
  for row in csv.DictReader(f,delimiter="\t"):
   record=_record_id(row)
   if not record:raise ValueError("base metadata row has no record_id")
   try:db.execute("INSERT INTO base_ids VALUES(?)",(record,))
   except sqlite3.IntegrityError as exc:raise ValueError(f"duplicate base record_id {record}") from exc
   split=_value(row,"split")
   for kind in ("species_taxid","split_group_id","duplicate_cluster_id"):_split_key(db,kind,_value(row,kind),split,True)
   n+=1;euk+=int(_normal_group(_value(row,"euk_group"))!="")
   if n%100000==0:db.commit()
 db.commit();return {"rows":n,"euk_rows":euk}
def _index_candidate_meta(db,path):
 n=unresolved=0
 with open(path,newline="",encoding="utf-8",errors="replace") as f:
  for row in csv.DictReader(f,delimiter="\t"):
   record=_record_id(row);group=_normal_group(_value(row,"euk_group"))
   if not record:raise ValueError("candidate metadata row has no record_id")
   unresolved+=int(not group)
   try:db.execute("INSERT INTO candidate_meta VALUES(?,?,?,?,?,?,?)",(record,_accession(row),_value(row,"split"),group,_value(row,"species_taxid"),_value(row,"split_group_id"),_value(row,"duplicate_cluster_id")))
   except sqlite3.IntegrityError as exc:raise ValueError(f"duplicate candidate record_id {record}") from exc
   n+=1
   if n%100000==0:db.commit()
 db.commit();return {"rows":n,"unresolved_euk":unresolved}
def _index_base_seq(db,root):
 n=dups=0;by=Counter()
 for split in ("train","validation"):
  for cls,name in FILES.items():
   if cls=="virus":continue
   p=Path(root)/split/name
   if not p.is_file():continue
   for header,seq in fasta(p):
    db.execute("INSERT OR IGNORE INTO base_ids VALUES(?)",(rid(header),));before=db.total_changes;db.execute("INSERT OR IGNORE INTO sequence_hashes VALUES(?)",(_seq_digest(seq),));dups+=int(db.total_changes==before);n+=1;by[split]+=1
    if n%100000==0:db.commit()
 db.commit();return {"records":n,"within_base_duplicate_sequences":dups,"by_split":dict(by)}
def _fastas(root,split):
 base=Path(root)/split;canonical=base/"eukarya.fasta"
 if canonical.is_file():return [canonical]
 out=[]
 for group in EUK:
  for ext in (".fasta",".fa",".fna"):
   p=base/f"{group}{ext}"
   if p.is_file():out.append(p)
 return out
def _requirements(audit,target):
 if target:
  invalid=set(target)-set(EUK)
  if invalid:raise ValueError(f"invalid target groups {sorted(invalid)}")
  return {s:set(target) for s in ("train","validation")}
 if audit:
  data=json.loads(Path(audit).read_text());return {s:set(data.get("missing_by_split",{}).get(s,[])) for s in ("train","validation")}
 return {s:set(EUK) for s in ("train","validation")}
def build_euk_delta(base_train_ready,base_metadata,candidate_root,candidate_metadata,out_dir,*,audit_report=None,base_freeze_manifest=None,verify_base="quick",target_groups=None,min_per_leaf_split=1,keep_index=False):
 verification={"ok":None,"mode":"not_requested"}
 if base_freeze_manifest:
  verification=verify_freeze(base_freeze_manifest,verify_base)
  if not verification["ok"]:raise ValueError("base freeze failed: "+"; ".join(verification["errors"]))
 req=_requirements(audit_report,target_groups);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
 for s in ("train","validation"):(out/s).mkdir(parents=True,exist_ok=True)
 dbpath=out/".euk_delta.sqlite"
 if dbpath.exists():dbpath.unlink()
 db=_db(dbpath);base_meta_stats=_index_base_meta(db,base_metadata);cand_meta_stats=_index_candidate_meta(db,candidate_metadata);base_seq_stats=_index_base_seq(db,base_train_ready)
 metadata=out/"euk_delta_metadata.tsv";fields=["record_id","accession","split","legacy_class","species_taxid","split_group_id","duplicate_cluster_id","euk_group","source_fasta","source_header","sequence_blake2b128"];accepted=defaultdict(Counter);rejected=Counter();inputs=[];handles={s:(out/s/"eukarya.fasta").open("w") for s in ("train","validation")}
 try:
  with metadata.open("w",newline="",encoding="utf-8") as mf:
   writer=csv.DictWriter(mf,fieldnames=fields,delimiter="\t");writer.writeheader()
   for split in ("train","validation"):
    paths=_fastas(candidate_root,split)
    if req[split] and not paths:raise FileNotFoundError(f"no candidate FASTA under {Path(candidate_root)/split}")
    for p in paths:
     inputs.append({"path":str(p.resolve()),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
     for header,seq in fasta(p):
      record=rid(header);row=db.execute("SELECT accession,split,euk_group,species_taxid,split_group_id,duplicate_cluster_id FROM candidate_meta WHERE record_id=?",(record,)).fetchone()
      if row is None:rejected["missing_metadata"]+=1;continue
      acc,meta_split,group,taxid,split_group,dup_cluster=row
      if not group:rejected["unresolved_euk_group"]+=1;continue
      if group not in req[split]:rejected["not_required_for_split"]+=1;continue
      if meta_split and meta_split!=split:rejected["metadata_split_mismatch"]+=1;continue
      if not taxid:rejected["missing_species_taxid"]+=1;continue
      if db.execute("SELECT 1 FROM base_ids WHERE record_id=?",(record,)).fetchone():rejected["duplicate_record_id_to_base"]+=1;continue
      digest=_seq_digest(seq)
      if db.execute("SELECT 1 FROM sequence_hashes WHERE digest=?",(digest,)).fetchone():rejected["duplicate_sequence_to_base_or_delta"]+=1;continue
      keys=(("species_taxid",taxid),("split_group_id",split_group),("duplicate_cluster_id",dup_cluster))
      if any(not _split_key(db,k,v,split,False) for k,v in keys):rejected["cross_split_group_overlap"]+=1;continue
      handles[split].write(f">{header}\n{seq}\n");writer.writerow({"record_id":record,"accession":acc or record.split("|",1)[0],"split":split,"legacy_class":"eukarya","species_taxid":taxid,"split_group_id":split_group,"duplicate_cluster_id":dup_cluster,"euk_group":group,"source_fasta":str(p.resolve()),"source_header":header.replace("\t"," "),"sequence_blake2b128":digest.hex()});db.execute("INSERT INTO base_ids VALUES(?)",(record,));db.execute("INSERT INTO sequence_hashes VALUES(?)",(digest,));accepted[split][group]+=1
  db.commit()
 finally:
  for h in handles.values():h.close()
  db.close()
 unmet={s:{g:min_per_leaf_split-accepted[s][g] for g in sorted(req[s]) if accepted[s][g]<min_per_leaf_split} for s in ("train","validation")};outputs=[]
 for s in ("train","validation"):
  p=out/s/"eukarya.fasta";outputs.append({"path":str(p.resolve()),"bytes":p.stat().st_size,"sha256":sha256_file(p),"records":sum(accepted[s].values())})
 outputs.append({"path":str(metadata.resolve()),"bytes":metadata.stat().st_size,"sha256":sha256_file(metadata)})
 manifest={"schema_version":1,"version":"2.3.1","stage":"euk_delta","generated_at":_now(),"policy":{"base_is_immutable":True,"sequence_dedup":"blake2b-128","cross_split_keys":["species_taxid","split_group_id","duplicate_cluster_id"],"min_per_leaf_split":min_per_leaf_split},"inputs":{"base_train_ready":str(Path(base_train_ready).resolve()),"base_metadata":str(Path(base_metadata).resolve()),"candidate_root":str(Path(candidate_root).resolve()),"candidate_metadata":str(Path(candidate_metadata).resolve()),"audit_report":audit_report,"base_freeze_manifest":base_freeze_manifest,"candidate_fastas":inputs},"base_verification":verification,"index_stats":{"base_metadata":base_meta_stats,"candidate_metadata":cand_meta_stats,"base_sequences":base_seq_stats},"requirements":{s:sorted(req[s]) for s in req},"accepted":{s:{g:accepted[s][g] for g in EUK} for s in ("train","validation")},"rejected":dict(rejected),"unmet":unmet,"outputs":outputs,"ready":not any(unmet.values())};(out/"euk_delta_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 if not keep_index:
  for suffix in ("","-wal","-shm"):
   try:Path(str(dbpath)+suffix).unlink()
   except FileNotFoundError:pass
 return manifest
__all__=["build_euk_delta"]
