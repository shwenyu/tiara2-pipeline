from __future__ import annotations
import argparse,json

def main():
 p=argparse.ArgumentParser(prog="python -m tiara.hierarchical");s=p.add_subparsers(dest="cmd",required=True)
 a=s.add_parser("init-metadata");a.add_argument("--train-ready",required=True);a.add_argument("--out",required=True);a.add_argument("--virus-dir");a.add_argument("--include-virus",action="store_true")
 a=s.add_parser("prepare");a.add_argument("--train-ready",required=True);a.add_argument("--out",required=True);a.add_argument("--tfidf",required=True);a.add_argument("--version",default="2.3.0");a.add_argument("--metadata-tsv");a.add_argument("--virus-dir");a.add_argument("--unknown-euk",choices=["error","other_protist"],default="error")
 a=s.add_parser("train");a.add_argument("--features",required=True);a.add_argument("--out",required=True);a.add_argument("--epochs",type=int,default=50);a.add_argument("--batch",type=int,default=1024);a.add_argument("--lr",type=float,default=1e-3);a.add_argument("--hidden",default="2048,1024");a.add_argument("--dropout",type=float,default=.2);a.add_argument("--device")
 a=s.add_parser("classify");a.add_argument("--checkpoint",required=True);a.add_argument("--tfidf",required=True);a.add_argument("-i","--input",required=True);a.add_argument("-o","--output",required=True);a.add_argument("--batch",type=int,default=512);a.add_argument("--device");a.add_argument("--min-len",type=int,default=1000)
 a=s.add_parser("audit");a.add_argument("--metadata-tsv",required=True);a.add_argument("--out")
 x=p.parse_args()
 if x.cmd=="init-metadata":
  from .data import init_metadata;r=init_metadata(x.train_ready,x.out,x.virus_dir,x.include_virus);print(json.dumps(r,indent=2,sort_keys=True));raise SystemExit(0 if r["ready"] else 2)
 elif x.cmd=="prepare":
  from .data import prepare;prepare(x.train_ready,x.out,x.tfidf,x.version,x.metadata_tsv,x.virus_dir,x.unknown_euk)
 elif x.cmd=="train":
  from .train import train;train(x.features,x.out,x.epochs,x.batch,x.lr,tuple(map(int,x.hidden.split(','))),x.dropout,x.device)
 elif x.cmd=="classify":
  from .infer import classify;classify(x.checkpoint,x.tfidf,x.input,x.output,x.batch,x.device,min_len=x.min_len)
 else:
  from .data import audit_tsv;r=audit_tsv(x.metadata_tsv);text=json.dumps(r,indent=2,sort_keys=True);print(text);x.out and open(x.out,"w").write(text+"\n");raise SystemExit(0 if r["ok"] else 2)
if __name__=="__main__":main()
