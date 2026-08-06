"""Atomic gate-controlled publisher for v2.3.1."""
from __future__ import annotations
import hashlib,json,os,shutil,time
from datetime import datetime,timezone
from pathlib import Path
def _sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def _checksums(root):
 entries=[]
 for p in sorted(x for x in root.rglob("*") if x.is_file() and x.name!="SHA256SUMS"):
  rel=p.relative_to(root).as_posix();entries.append({"path":rel,"bytes":p.stat().st_size,"sha256":_sha(p)})
 (root/"SHA256SUMS").write_text("".join(f"{x['sha256']}  {x['path']}\n" for x in entries));return entries
def _link(target,link):
 link.parent.mkdir(parents=True,exist_ok=True);tmp=link.with_name(f".{link.name}.tmp.{os.getpid()}")
 try:tmp.unlink()
 except FileNotFoundError:pass
 tmp.symlink_to(target.resolve(),target_is_directory=True);os.replace(tmp,link)
def publish_release(checkpoint,training_history,config,base_freeze_manifest,completeness_report,delta_manifest,composite_features,evaluation_json,gates_json,destination,*,current_link=None,extra_files=None,backup=True):
 gates=json.loads(Path(gates_json).read_text())
 if gates.get("version")!="2.3.1" or not gates.get("passed"):raise ValueError("refusing publish: gates failed")
 import torch
 ck=torch.load(checkpoint,map_location="cpu",weights_only=False);model=ck.get("model",{})
 if ck.get("version")!="2.3.1" or model.get("hidden")!=[2048,1024] or float(model.get("dropout",-1))!=.2 or ck.get("checkpoint_selector")!="root_macro_f1":raise ValueError("checkpoint contract failed")
 required={"hierarchical_model.pt":checkpoint,"training_history.json":training_history,"config_v2_3_1_euk_completeness.yaml":config,"base_freeze_manifest.json":base_freeze_manifest,"euk_completeness_report.json":completeness_report,"euk_delta_manifest.json":delta_manifest,"composite_features.json":composite_features,"evaluation.json":evaluation_json,"acceptance_gates.json":gates_json}
 for name,p in required.items():
  if not Path(p).is_file():raise FileNotFoundError(f"missing {name}: {p}")
 dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True);staged=dest.with_name(f".{dest.name}.staging.{os.getpid()}")
 if staged.exists():shutil.rmtree(staged)
 staged.mkdir();backup_path=None
 try:
  for name,p in required.items():shutil.copy2(p,staged/name)
  extras=[]
  for p in extra_files or ():
   src=Path(p);shutil.copy2(src,staged/src.name);extras.append(src.name)
  manifest={"schema_version":1,"version":"2.3.1","generated_at":datetime.now(timezone.utc).isoformat(),"checkpoint":{"schema":ck.get("schema"),"model":model,"best_root_macro_f1":ck.get("best_root_macro_f1"),"checkpoint_selector":ck.get("checkpoint_selector")},"frozen_primary_variable":"Euk data completeness","required_files":sorted(required),"extra_files":sorted(extras),"gates":gates};(staged/"model_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");files=_checksums(staged)
  if dest.exists():
   if not backup:raise FileExistsError(dest)
   backup_path=dest.with_name(f"{dest.name}.backup_{time.strftime('%Y%m%d_%H%M%S')}");os.replace(dest,backup_path)
  os.replace(staged,dest)
  if current_link:_link(dest,Path(current_link))
 finally:
  if staged.exists():shutil.rmtree(staged,ignore_errors=True)
 return {"published":True,"version":"2.3.1","destination":str(dest.resolve()),"backup":str(backup_path.resolve()) if backup_path else None,"current_link":current_link,"files":files}
__all__=["publish_release"]
