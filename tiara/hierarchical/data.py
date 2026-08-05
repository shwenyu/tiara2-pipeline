"""Build v2.3.x hierarchical feature memmaps from the frozen v2.1.3 corpus."""
from __future__ import annotations
import argparse,csv,json,hashlib,re
from pathlib import Path
import numpy as np
from .schema import schema
from .labels import labels_for
FILES={"bacteria":"bacteria.fasta","archaea":"archaea.fasta","eukarya":"eukarya.fasta","mitochondria":"mitochondria.fasta","plastids":"plastids.fasta","virus":"virus.fasta"}
_META_TOKEN=re.compile(r"(?:^|[|;\s])(?P<key>species_taxid|taxid|host_species_taxid|split_group_id|virus_cluster_id)=(?P<value>[^|;\s]+)",re.I)

def fasta(path):
    head=None; parts=[]
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if head is not None: yield head,"".join(parts).upper()
                head=line[1:].strip(); parts=[]
            else: parts.append(line.strip())
        if head is not None: yield head,"".join(parts).upper()

def rid(header): return header.split()[0]
def header_meta(header):
    out={m.group("key").lower():m.group("value") for m in _META_TOKEN.finditer(header or "")}
    if "species_taxid" not in out and "taxid" in out:out["species_taxid"]=out["taxid"]
    try:out["euk_group"]=labels_for("eukarya",header,policy="error")["euk"]
    except ValueError:out["euk_group"]=""
    return out

def init_metadata(train_ready,out_path,virus_dir=None,include_virus=False):
    """Create the metadata TSV that audit/prepare consume.

    This is intentionally a separate, explicit step: unresolved euk labels are
    written blank and must be curated instead of silently becoming
    other_protist.
    """
    out=Path(out_path);out.parent.mkdir(parents=True,exist_ok=True)
    fields=["record_id","split","legacy_class","species_taxid","host_species_taxid","split_group_id","virus_cluster_id","euk_group","label_status"]
    counts={"rows":0,"unresolved_euk":0}
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter="\t");w.writeheader()
        for split in ("train","validation"):
            base=Path(train_ready)/split
            for cls,name in FILES.items():
                if cls=="virus" and not include_virus:continue
                path=(Path(virus_dir)/split/name if cls=="virus" and virus_dir else base/name)
                if not path.is_file():continue
                for header,_seq in fasta(path):
                    md=header_meta(header); unresolved=cls=="eukarya" and not md.get("euk_group")
                    row={k:"" for k in fields};row.update(md);row.update({"record_id":rid(header),"split":split,"legacy_class":cls,"label_status":"unresolved_euk" if unresolved else "resolved"})
                    w.writerow(row);counts["rows"]+=1;counts["unresolved_euk"]+=int(unresolved)
    report={"path":str(out),**counts,"ready":counts["unresolved_euk"]==0}
    (out.with_suffix(out.suffix+".report.json")).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report
def load_meta(path):
    if not path:return {}
    with open(path,newline="") as f:
        rows=csv.DictReader(f,delimiter="\t"); return {r.get("record_id") or r.get("id") or r.get("accession"):r for r in rows}
def _sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def prepare(train_ready,out_dir,tfidf_folder,version="2.3.0",metadata_tsv=None,virus_dir=None,unknown_euk="error",chunk=2048):
    # Keep metadata init/audit usable in a lightweight base environment.  The
    # scientific stack is required only when feature preparation actually runs.
    from tiara.training.featurize_cache import featurize_block
    from tiara.src.transformations import TfidfWeighter
    sc=schema(version); roots=sc.index("root"); idx={h:sc.index(h) for h in ("euk","prok","organelle")}; meta=load_meta(metadata_tsv)
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); tf=TfidfWeighter.load_params(str(tfidf_folder)); k=int(tf.k); dim=4**k
    summary={"version":version,"schema":sc.to_dict(),"k":k,"tfidf":str(tfidf_folder),"tfidf_sha256":_sha(Path(tfidf_folder)/"model.npy"),"splits":{}}
    for split in ("train","validation"):
        entries=[]; base=Path(train_ready)/split
        for cls,name in FILES.items():
            if cls=="virus" and not sc.profile.virus_enabled:continue
            p=(Path(virus_dir)/split/name if cls=="virus" and virus_dir else base/name)
            if p.is_file():entries.append((cls,p))
        n=sum(1 for _,p in entries for _ in fasta(p)); sd=out/split; sd.mkdir(exist_ok=True)
        X=np.memmap(sd/"X.f32",dtype="float32",mode="w+",shape=(n,dim)); ys={h:np.memmap(sd/f"{h}.i64",dtype="int64",mode="w+",shape=(n,)) for h in ("root","euk","prok","organelle")}
        pos=0; counts={}
        for cls,p in entries:
            buf=[]; labs=[]
            def flush():
                nonlocal pos,buf,labs
                if not buf:return
                z=featurize_block(buf,k,np.asarray(tf.idfs,dtype=np.float32),dim); m=len(buf); X[pos:pos+m]=z
                for j,lab in enumerate(labs):
                    ys["root"][pos+j]=roots[lab["root"]]
                    for h in ("euk","prok","organelle"):ys[h][pos+j]=-1 if lab[h] is None else idx[h][lab[h]]
                    key=lab["root"]+":"+str(lab.get("euk") or lab.get("prok") or lab.get("organelle") or "-"); counts[key]=counts.get(key,0)+1
                pos+=m;buf=[];labs=[]
            for header,seq in fasta(p):
                row=meta.get(rid(header),{}); lab=labels_for(cls,header,row.get("euk_group"),unknown_euk); buf.append(seq);labs.append(lab)
                if len(buf)>=chunk:flush()
            flush()
        X.flush();[v.flush() for v in ys.values()]
        sm={"rows":n,"dim":dim,"counts":counts,"files":{h:str(sd/("X.f32" if h=="X" else f"{h}.i64")) for h in ("X","root","euk","prok","organelle")}}
        (sd/"manifest.json").write_text(json.dumps(sm,indent=2,sort_keys=True)+"\n");summary["splits"][split]=sm
    (out/"hierarchy_features.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");return summary

def audit_tsv(path):
    path=Path(path)
    if not path.is_file():
        return {"ok":False,"error":"metadata_missing","path":str(path),"next_command":"python -m tiara.hierarchical init-metadata --train-ready /ssd/shouhanyu/Tiara2/corpus_ready_v2_1_3_bpbalanced --out "+str(path)}
    with open(path,newline="") as f: rows=list(csv.DictReader(f,delimiter="\t"))
    fields=[x for x in ("species_taxid","split_group_id","virus_cluster_id") if x in (rows[0] if rows else {})]; problems={}
    for field in fields:
        seen={}
        for r in rows:
            key=r.get(field); split=r.get("split")
            if key and split:seen.setdefault(key,set()).add(split)
        bad={k:sorted(v) for k,v in seen.items() if len(v)>1};problems[field]=bad
    unresolved=sum(1 for r in rows if r.get("legacy_class")=="eukarya" and not r.get("euk_group"))
    return {"ok":not any(problems.values()) and unresolved==0,"rows":len(rows),"unresolved_euk":unresolved,"overlaps":{k:len(v) for k,v in problems.items()},"details":problems}
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--train-ready",required=True);p.add_argument("--out",required=True);p.add_argument("--tfidf",required=True);p.add_argument("--version",default="2.3.0");p.add_argument("--metadata-tsv");p.add_argument("--virus-dir");p.add_argument("--unknown-euk",choices=["error","skip","other_protist"],default="error");a=p.parse_args(argv);prepare(a.train_ready,a.out,a.tfidf,a.version,a.metadata_tsv,a.virus_dir,a.unknown_euk)
if __name__=="__main__":main()
