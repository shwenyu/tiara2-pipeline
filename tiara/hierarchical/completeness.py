"""v2.3.1 Euk completeness audit and metadata curation."""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from .schema import EUK

FIELD_ALIASES = {
 "record_id": ("record_id","frag_id","id","sequence_id"),
 "accession": ("accession","assembly_accession","genome_accession"),
 "species_taxid": ("species_taxid","taxid","organism_taxid"),
 "lineage": ("lineage","taxonomic_lineage","taxonomy"),
 "organism": ("organism","organism_name","species_name"),
 "euk_group": ("euk_group","clade","stage2_class","leaf"),
 "legacy_class": ("legacy_class","fine_class","class"),
 "label": ("label",), "supergroup": ("supergroup","sg"),
 "source_group": ("source_group","group","division"), "split": ("split",),
 "split_group_id": ("split_group_id",),
 "duplicate_cluster_id": ("duplicate_cluster_id","dedup_cluster_id","duplicate_cluster"),
}
GROUP_ALIASES = {"fungi":"fungi","opisthokonta-fungi":"fungi","land_plant":"land_plant","plant":"land_plant","embryophyta":"land_plant","algae":"algae","alga":"algae","metazoa_vertebrate":"metazoa_vertebrate","vertebrate":"metazoa_vertebrate","host-mammalia":"metazoa_vertebrate","metazoa_invertebrate":"metazoa_invertebrate","invertebrate":"metazoa_invertebrate","alveolata":"alveolata","stramenopiles":"stramenopiles","other_protist":"other_protist","protist":"other_protist"}

def _now(): return datetime.now(timezone.utc).isoformat()
def _sha256(path):
 if not path or not Path(path).is_file(): return None
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()
def _value(row, canonical):
 for name in FIELD_ALIASES.get(canonical,(canonical,)):
  value=row.get(name)
  if value is not None and str(value).strip(): return str(value).strip()
 return ""
def _normal_group(value):
 key=str(value or "").strip().lower().replace(" ","_")
 return GROUP_ALIASES.get(key,key if key in EUK else "")
def _record_id(row): return _value(row,"record_id")
def _accession(row):
 value=_value(row,"accession")
 return value or (_record_id(row).split("|",1)[0] if _record_id(row) else "")
def _is_euk_row(row):
 if _normal_group(_value(row,"euk_group")): return True
 legacy=_value(row,"legacy_class").lower(); label=_value(row,"label").lower(); sg=_value(row,"supergroup").lower()
 return legacy in {"euk","eukarya","eukaryota","euk_nuclear"} or label in {"euk","eukarya","eukaryota","euk_nuclear"} or "euk" in legacy or sg.startswith(("opisthokonta","archaeplastida","protist","sar"))
def _open_tsv(path):
 handle=Path(path).open(newline="",encoding="utf-8",errors="replace")
 return csv.DictReader(handle,delimiter="\t"),handle

def _load_overrides(path):
 if not path:return {}
 out={}; reader,h=_open_tsv(path)
 try:
  for row in reader:
   key=(row.get("key") or _record_id(row) or _accession(row)).strip(); group=_normal_group(_value(row,"euk_group"))
   if not key or not group: raise ValueError("override TSV requires key and valid euk_group")
   if key in out and out[key]!=group: raise ValueError(f"conflicting override for {key}")
   out[key]=group
 finally:h.close()
 return out

def _load_sources(path):
 if not path:return {},0
 out={}; duplicates=0; reader,h=_open_tsv(path)
 try:
  for row in reader:
   acc=_accession(row)
   if not acc:continue
   if acc in out:
    duplicates+=1
    for k,v in row.items():
     if v and not out[acc].get(k):out[acc][k]=v
   else:out[acc]=dict(row)
 finally:h.close()
 return out,duplicates

class Resolver:
 def __init__(self,taxdump_dir=None,overrides=None):
  self.overrides=overrides or {}; self.taxonomy=None
  from tiara2.taxonomy import Taxonomy
  self.lineage_taxonomy=Taxonomy.from_lineages()
  if taxdump_dir:
   from tiara2.taxonomy import find_taxdump
   root=find_taxdump(taxdump_dir)
   if root is None:raise FileNotFoundError(f"taxdump not found under {taxdump_dir}")
   self.taxonomy=Taxonomy.from_taxdump(root)
 def resolve(self,row,source=None):
  source=source or {}; record=_record_id(row); acc=_accession(row) or _accession(source)
  for key in (record,acc):
   if key and key in self.overrides:return self.overrides[key],"manual_override"
  taxid=_value(row,"species_taxid") or _value(source,"species_taxid")
  lineage=_value(row,"lineage") or _value(source,"lineage"); organism=_value(row,"organism") or _value(source,"organism")
  if taxid and self.taxonomy:
   info=self.taxonomy.info(taxid,lineage=lineage,organism=organism)
   if info.clade in EUK:return info.clade,"taxdump"
  if lineage:
   info=self.lineage_taxonomy.info(taxid or 0,lineage=lineage,organism=organism)
   if info.clade in EUK:return info.clade,"lineage"
  group=(_value(row,"source_group") or _value(source,"source_group")).lower()
  group_map={"fungi":"fungi","plant":"land_plant","protozoa":"other_protist","invertebrate":"metazoa_invertebrate","vertebrate_other":"metazoa_vertebrate","vertebrate_mammalian":"metazoa_vertebrate"}
  if group in group_map:return group_map[group],"source_group"
  sg=(_value(row,"supergroup") or _value(source,"supergroup")).lower()
  sg_map={"opisthokonta-fungi":"fungi","host-mammalia":"metazoa_vertebrate","opisthokonta-metazoa-vertebrate":"metazoa_vertebrate","opisthokonta-metazoa-invertebrate":"metazoa_invertebrate","archaeplastida-landplant":"land_plant","archaeplastida-alga":"algae","sar-alveolata":"alveolata","sar-stramenopiles":"stramenopiles"}
  if sg in sg_map:return sg_map[sg],"supergroup"
  existing=_normal_group(_value(row,"euk_group") or _value(source,"euk_group"))
  return (existing,"existing_label") if existing else ("","unresolved")

def _stream_counts(path,resolver,sources,unique=False):
 counts=Counter(); unresolved=0; seen=defaultdict(set)
 if not path:return counts,unresolved
 reader,h=_open_tsv(path)
 try:
  for row in reader:
   acc=_accession(row); group,_=resolver.resolve(row,sources.get(acc,{}))
   if not group:
    unresolved+=int(_is_euk_row(row));continue
   if unique:
    if acc and acc not in seen[group]:seen[group].add(acc);counts[group]+=1
   else:counts[group]+=1
 finally:h.close()
 return counts,unresolved

def audit_euk_completeness(base_metadata,out_dir,*,source_index=None,pool_metadata=None,taxdump_dir=None,overrides_tsv=None):
 out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
 overrides=_load_overrides(overrides_tsv); sources,source_dups=_load_sources(source_index); resolver=Resolver(taxdump_dir,overrides)
 source_counts=Counter(); source_seen=defaultdict(set)
 for acc,row in sources.items():
  group,_=resolver.resolve(row,row)
  if group and acc not in source_seen[group]:source_seen[group].add(acc);source_counts[group]+=1
 pool_counts,pool_unresolved=_stream_counts(pool_metadata,resolver,sources)
 curated=out/"hierarchical_labels_v2_3_1.tsv"; record_audit=out/"euk_record_audit.tsv"
 before=Counter();after=Counter();by_split=defaultdict(Counter);actions=Counter();methods=Counter();group_splits={k:{} for k in ("species_taxid","split_group_id","duplicate_cluster_id")};overlaps=Counter();examples=defaultdict(list)
 reader,h=_open_tsv(base_metadata);base_fields=list(reader.fieldnames or []);extra=["accession","lineage","euk_group","label_status","euk_group_source"];fields=base_fields+[x for x in extra if x not in base_fields]
 total=euk_rows=unresolved=0
 try:
  with curated.open("w",newline="",encoding="utf-8") as co,record_audit.open("w",newline="",encoding="utf-8") as ao:
   cw=csv.DictWriter(co,fieldnames=fields,delimiter="\t",extrasaction="ignore");aw=csv.DictWriter(ao,fieldnames=["record_id","accession","split","species_taxid","lineage","current_euk_group","resolved_euk_group","resolution_source","label_action"],delimiter="\t");cw.writeheader();aw.writeheader()
   for row in reader:
    total+=1;outrow=dict(row)
    if not _is_euk_row(row):cw.writerow(outrow);continue
    euk_rows+=1;acc=_accession(row);source=sources.get(acc,{});current=_normal_group(_value(row,"euk_group"));resolved,method=resolver.resolve(row,source);split=_value(row,"split");taxid=_value(row,"species_taxid") or _value(source,"species_taxid");lineage=_value(row,"lineage") or _value(source,"lineage")
    if current:before[current]+=1
    if not resolved:action="unresolved";unresolved+=1
    elif not current:action="label_error_missing"
    elif current!=resolved:action="label_error_wrong"
    else:action="unchanged"
    actions[action]+=1;methods[method]+=1
    if resolved:after[resolved]+=1;by_split[split][resolved]+=1
    outrow.update({"accession":acc,"species_taxid":taxid,"lineage":lineage,"euk_group":resolved,"label_status":"resolved" if resolved else "unresolved_euk","euk_group_source":method});cw.writerow(outrow)
    aw.writerow({"record_id":_record_id(row),"accession":acc,"split":split,"species_taxid":taxid,"lineage":lineage,"current_euk_group":current,"resolved_euk_group":resolved,"resolution_source":method,"label_action":action})
    for kind in group_splits:
     value=_value(outrow,kind)
     if value:
      prev=group_splits[kind].setdefault(value,split)
      if prev and split and prev!=split:overlaps[kind]+=1
 finally:h.close()
 coverage=[]
 with (out/"euk_coverage.tsv").open("w",newline="") as f:
  names=["euk_group","base_before","base_after","base_train","base_validation","existing_pool_fragments","source_genomes","diagnosis","needs_delta_train","needs_delta_validation"];w=csv.DictWriter(f,fieldnames=names,delimiter="\t");w.writeheader()
  for group in EUK:
   tr=by_split["train"][group];va=by_split["validation"][group]
   if tr and va:diag="label_error_recovered" if before[group]==0 and after[group]>0 else "complete"
   elif pool_counts[group]>0:diag="bp_balance_not_selected"
   elif source_counts[group]>0:diag="source_fragments_missing"
   else:diag="source_missing"
   row={"euk_group":group,"base_before":before[group],"base_after":after[group],"base_train":tr,"base_validation":va,"existing_pool_fragments":pool_counts[group],"source_genomes":source_counts[group],"diagnosis":diag,"needs_delta_train":int(tr==0),"needs_delta_validation":int(va==0)};w.writerow(row);coverage.append(row)
 missing={s:[g for g in EUK if by_split[s][g]==0] for s in ("train","validation")}
 report_path=out/"euk_completeness_report.json"
 report={"schema_version":1,"version":"2.3.1","generated_at":_now(),"inputs":{"base_metadata":str(base_metadata),"base_metadata_sha256":_sha256(base_metadata),"source_index":source_index,"source_index_sha256":_sha256(source_index),"pool_metadata":pool_metadata,"pool_metadata_sha256":_sha256(pool_metadata)},"outputs":{"curated_metadata":str(curated),"record_audit":str(record_audit),"coverage":str(out/"euk_coverage.tsv"),"report":str(report_path)},"rows":{"total":total,"euk":euk_rows,"unresolved_euk":unresolved,"pool_unresolved_euk":pool_unresolved,"duplicate_source_rows_merged":source_dups},"base_before":{g:before[g] for g in EUK},"base_after":{g:after[g] for g in EUK},"base_by_split":{s:{g:by_split[s][g] for g in EUK} for s in ("train","validation")},"missing_by_split":missing,"label_actions":dict(actions),"resolution_methods":dict(methods),"leakage":{"overlap_counts":dict(overlaps),"examples":dict(examples)},"coverage":coverage}
 report["training_ready"]=unresolved==0 and not missing["train"] and not missing["validation"] and not any(overlaps.values());report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");return report

__all__=["audit_euk_completeness","Resolver","_normal_group","_record_id","_accession","_value","_is_euk_row"]
