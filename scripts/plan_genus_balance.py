#!/usr/bin/env python3
"""Genome-level abundance balancing across all five Tiara2 classes.

The bias being corrected is repeated sequencing of popular species. Selection:
  1) one best accession per (scope, training_class, species_taxid)
  2) at most N different species per (scope, training_class, genus_taxid)

``scope`` is explicit:
  * global    -- one cap across the whole corpus (v2.1 behaviour)
  * per_split -- an independent cap in train/validation/test (v2.1.2)
No fragments are sampled and target_bp is intentionally blank.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict, Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent; REPO=HERE.parent
for p in (str(HERE), str(REPO)):
    if p not in sys.path: sys.path.insert(0,p)
from progress import add_progress_args, counter_from_args
from tiara2.taxonomy import Taxonomy
from select_genomes import quality_key

FIVE=("bacteria","archaea","eukarya","mitochondria","plastids")
ACC_KEYS=("assembly_accession","entity_id","sequence_accession","accession")
PLAN_COLS=("assembly_accession","split","training_class","taxid","species_taxid",
           "species_name","genus_taxid","genus_name","species_rank",
           "genus_rank","organism_name","is_anchor","target_bp","target_fragments")
DECISION_COLS=PLAN_COLS+("selected","decision",)


def first(r, keys):
    for k in keys:
        v=str(r.get(k,"") or "").strip()
        if v: return v
    return ""


def read_tsv(path):
    with open(path,encoding="utf-8",newline="") as fh:
        r=csv.DictReader(fh,delimiter="\t"); return list(r),list(r.fieldnames or [])


def write_tsv(path,rows,cols):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=cols,delimiter="\t",extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def read_anchors(path):
    if not path: return set()
    out=set()
    for line in Path(path).read_text().splitlines():
        x=line.split("#",1)[0].strip()
        if x: out.add(x.split("\t",1)[0])
    return out

def read_splits(path):
    """Return exact/base accession -> split from index/split_assignments.tsv."""
    if not path: return {},{}
    rows,_=read_tsv(path); exact={}; bases={}
    for r in rows:
        acc=first(r,("entity_id","assembly_accession","sequence_accession","accession"))
        split=first(r,("split","epoch"))
        if acc and split:
            exact[acc]=split; bases.setdefault(base_acc(acc),split)
    return exact,bases


def base_acc(x):
    x=str(x or ""); return x.rsplit(".",1)[0] if "." in x else x


def rank_ids(tax, taxid, accession, organism):
    try: tid=int(taxid)
    except (TypeError,ValueError): tid=0
    info=tax.info(tid,organism=organism)
    species=0
    if tid:
        for node,rank in tax.ancestors(tid):
            if rank=="species": species=node; break
    # Conservative fallback: unresolved taxa never collapse together.
    species_key=f"t{species}" if species else f"acc:{accession}"
    species_name=tax.names.get(species, organism or accession) if species else (organism or accession)
    genus=info.ranks.get("genus",0) if info.ok else 0
    genus_key=f"t{genus}" if genus else (info.genus_key or f"acc:{accession}")
    genus_name=tax.names.get(genus,info.genus_name or organism or accession) if genus else (info.genus_name or organism or accession)
    return species,species_key,species_name,genus,genus_key,genus_name,info.ok


def main(argv=None):
    ap=argparse.ArgumentParser(description="One genome per species; cap species per genus in all five classes")
    ap.add_argument("--candidates",required=True)
    ap.add_argument("--taxdump",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--report",default="")
    ap.add_argument("--decisions",default="")
    ap.add_argument("--anchors",default="")
    ap.add_argument("--split-assignments",default="",
                    help="split_assignments.tsv used when --scope=per_split")
    ap.add_argument("--scope",choices=("global","per_split"),default="global",
                    help="global=v2.1; per_split=independent train/validation/test cap (v2.1.2)")
    ap.add_argument("--max-species-per-genus",type=int,default=4)
    ap.add_argument("--dry-run",action="store_true")
    add_progress_args(ap); a=ap.parse_args(argv)
    if a.max_species_per_genus<1: raise SystemExit("--max-species-per-genus must be >= 1")
    rows,_=read_tsv(a.candidates); tax=Taxonomy.from_taxdump(a.taxdump)
    anchors=read_anchors(a.anchors); anchor_bases={base_acc(x) for x in anchors}
    split_exact,split_base=read_splits(a.split_assignments)
    annotated=[]; counter=counter_from_args(a,"genus-balance",total=len(rows),noun="row")
    for src in rows:
        counter.count(); acc=first(src,ACC_KEYS); k=first(src,("training_class",))
        if not acc or k not in FIVE: continue
        r=dict(src); r["assembly_accession"]=acc
        resolved_split=(split_exact.get(acc) or split_base.get(base_acc(acc)) or
                        first(r,("split","epoch")))
        if a.scope=="per_split" and not resolved_split:
            raise SystemExit(
                f"--scope=per_split but no split was resolved for {acc}; "
                "provide --split-assignments or a split/epoch column")
        split=resolved_split if a.scope=="per_split" else "all"
        r["split"]=split
        organism=first(r,("organism_name","definition")); taxid=first(r,("taxid","species_taxid"))
        sp,spk,spn,ge,gek,gen,ok=rank_ids(tax,taxid,acc,organism)
        r.update({"_species_key":spk,"species_taxid":str(sp or ""),"species_name":spn,
                  "_genus_key":gek,"genus_taxid":str(ge or ""),"genus_name":gen,
                  "_tax_ok":ok,"is_anchor":"1" if base_acc(acc) in anchor_bases else "0"})
        annotated.append(r)
    counter.finish()

    # Stage 1: one best genome/sequence per species inside each class.
    by_species=defaultdict(list)
    for r in annotated: by_species[(r["split"],r["training_class"],r["_species_key"])].append(r)
    species_winners=[]; decisions=[]
    for (_split,_k,_s),group in sorted(by_species.items()):
        group.sort(key=lambda r:(0 if r["is_anchor"]=="1" else 1,quality_key(r)))
        for i,r in enumerate(group):
            r["species_rank"]=i+1
            if i==0: species_winners.append(r)
            else:
                d=dict(r); d.update({"genus_rank":"","selected":"0","decision":"duplicate_within_species"}); decisions.append(d)

    # Stage 2: cap the number of distinct species represented by each genus.
    by_genus=defaultdict(list)
    for r in species_winners: by_genus[(r["split"],r["training_class"],r["_genus_key"])].append(r)
    selected=[]
    for (_split,_k,_g),group in sorted(by_genus.items()):
        group.sort(key=lambda r:(0 if r["is_anchor"]=="1" else 1,quality_key(r)))
        for i,r in enumerate(group):
            r["genus_rank"]=i+1
            d=dict(r)
            if i<a.max_species_per_genus:
                r["target_bp"]=""; r["target_fragments"]=""; selected.append(r)
                d.update({"selected":"1","decision":"selected"})
            else: d.update({"selected":"0","decision":"dropped_by_genus_cap"})
            decisions.append(d)

    per_class={}
    for k in FIVE:
        inp=[r for r in annotated if r["training_class"]==k]
        dec=[r for r in decisions if r["training_class"]==k]
        per_class[k]={
            "input_accessions":len(inp),
            "unique_species":len({r["_species_key"] for r in inp}),
            "duplicate_assemblies_removed":sum(r["decision"]=="duplicate_within_species" for r in dec),
            "unique_genera":len({r["_genus_key"] for r in inp}),
            "genomes_removed_by_genus_cap":sum(r["decision"]=="dropped_by_genus_cap" for r in dec),
            "selected_genomes":sum(r["decision"]=="selected" for r in dec),
            "unresolved_taxonomy":sum(not r["_tax_ok"] for r in inp),
        }
    per_split_class={}
    for split in sorted({r["split"] for r in annotated}):
        per_split_class[split]={}
        for k in FIVE:
            inp=[r for r in annotated if r["split"]==split and r["training_class"]==k]
            dec=[r for r in decisions if r["split"]==split and r["training_class"]==k]
            per_split_class[split][k]={
                "input_accessions":len(inp),
                "unique_species":len({r["_species_key"] for r in inp}),
                "unique_genera":len({r["_genus_key"] for r in inp}),
                "selected_genomes":sum(r["decision"]=="selected" for r in dec),
            }
    selected_bases={base_acc(r["assembly_accession"]) for r in selected}
    report={"strategy":"one_best_genome_per_species_then_genus_cap",
            "scope":a.scope,
            "grouping":("split,training_class,genus" if a.scope=="per_split"
                        else "training_class,genus"),
            "max_species_per_genus":a.max_species_per_genus,"taxonomy_backend":"taxdump",
            "input_rows":len(annotated),"selected_rows":len(selected),"per_class":per_class,
            "per_split_class":per_split_class,
            "anchors_requested":len(anchor_bases),"anchors_selected":len(anchor_bases & selected_bases),
            "anchors_deferred_or_absent":sorted(anchor_bases-selected_bases)}
    if not a.dry_run:
        write_tsv(a.out,selected,PLAN_COLS)
        decpath=a.decisions or str(Path(a.out).with_name("genus_balance_decisions.tsv"))
        write_tsv(decpath,decisions,DECISION_COLS)
        repath=a.report or str(Path(a.out).with_name("genus_balance_report.json"))
        Path(repath).write_text(json.dumps(report,indent=2))
    print(f"genus-level genome balance (scope={a.scope})")
    for k,v in per_class.items():
        print(f"  {k:14s}: input {v['input_accessions']:,} -> selected {v['selected_genomes']:,}; "
              f"species duplicates {v['duplicate_assemblies_removed']:,}; genus-cap drops {v['genomes_removed_by_genus_cap']:,}")
    print(f"  TOTAL         : {len(annotated):,} -> {len(selected):,}")
    if a.dry_run: print("  DRY RUN: no files written")
    else: print(f"  selection -> {a.out}")
    return 0

if __name__=="__main__": sys.exit(main())
