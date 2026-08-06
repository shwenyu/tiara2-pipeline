from __future__ import annotations
import argparse,json

def show(x):print(json.dumps(x,indent=2,sort_keys=True))
def targets(x):return [v.strip() for v in x.split(',') if v.strip()] if x else None
def main(argv=None):
 p=argparse.ArgumentParser(prog="python -m tiara.hierarchical");s=p.add_subparsers(dest="cmd",required=True)
 a=s.add_parser("init-metadata");a.add_argument("--train-ready",required=True);a.add_argument("--out",required=True);a.add_argument("--virus-dir");a.add_argument("--include-virus",action="store_true")
 a=s.add_parser("audit");a.add_argument("--metadata-tsv",required=True);a.add_argument("--out")
 a=s.add_parser("prepare");a.add_argument("--train-ready",required=True);a.add_argument("--out",required=True);a.add_argument("--tfidf",required=True);a.add_argument("--version",default="2.3.0");a.add_argument("--metadata-tsv");a.add_argument("--virus-dir");a.add_argument("--unknown-euk",choices=["error","other_protist"],default="error")
 a=s.add_parser("train");a.add_argument("--features",required=True);a.add_argument("--out",required=True);a.add_argument("--epochs",type=int,default=50);a.add_argument("--batch",type=int,default=1024);a.add_argument("--lr",type=float,default=1e-3);a.add_argument("--hidden",default="2048,1024");a.add_argument("--dropout",type=float,default=.2);a.add_argument("--device")
 a=s.add_parser("classify");a.add_argument("--checkpoint",required=True);a.add_argument("--tfidf",required=True);a.add_argument("-i","--input",required=True);a.add_argument("-o","--output",required=True);a.add_argument("--batch",type=int,default=512);a.add_argument("--device");a.add_argument("--min-len",type=int,default=1000)
 a=s.add_parser("freeze-base");a.add_argument("--train-ready",required=True);a.add_argument("--tfidf",required=True);a.add_argument("--metadata-tsv",required=True);a.add_argument("--out",required=True)
 a=s.add_parser("verify-freeze");a.add_argument("--manifest",required=True);a.add_argument("--mode",choices=["quick","full"],default="full")
 a=s.add_parser("audit-euk");a.add_argument("--base-metadata",required=True);a.add_argument("--out",required=True);a.add_argument("--source-index");a.add_argument("--pool-metadata");a.add_argument("--taxdump-dir");a.add_argument("--overrides-tsv")
 a=s.add_parser("build-euk-delta");a.add_argument("--base-train-ready",required=True);a.add_argument("--base-metadata",required=True);a.add_argument("--candidate-root",required=True);a.add_argument("--candidate-metadata",required=True);a.add_argument("--out",required=True);a.add_argument("--audit-report");a.add_argument("--base-freeze-manifest");a.add_argument("--verify-base",choices=["quick","full"],default="full");a.add_argument("--target-groups");a.add_argument("--min-per-leaf-split",type=int,default=1);a.add_argument("--keep-index",action="store_true")
 a=s.add_parser("prepare-v231");a.add_argument("--base-features",required=True);a.add_argument("--base-train-ready",required=True);a.add_argument("--curated-base-metadata",required=True);a.add_argument("--delta-root",required=True);a.add_argument("--delta-metadata",required=True);a.add_argument("--tfidf",required=True);a.add_argument("--out",required=True);a.add_argument("--base-freeze-manifest");a.add_argument("--verify-base",choices=["quick","full"],default="full");a.add_argument("--chunk",type=int,default=2048)
 a=s.add_parser("evaluate-v231");a.add_argument("--truth",required=True);a.add_argument("--predictions",required=True);a.add_argument("--out",required=True)
 a=s.add_parser("gate-v231");a.add_argument("--completeness-report",required=True);a.add_argument("--composite-features",required=True);a.add_argument("--evaluation",required=True);a.add_argument("--base-metadata",required=True);a.add_argument("--delta-metadata",required=True);a.add_argument("--baseline-benchmark",required=True);a.add_argument("--current-benchmark",required=True);a.add_argument("--checkpoint",required=True);a.add_argument("--max-legacy-regression-pp",type=float,default=.20);a.add_argument("--min-leaf-recall",type=float,default=.05);a.add_argument("--out",required=True)
 a=s.add_parser("publish-v231");a.add_argument("--checkpoint",required=True);a.add_argument("--training-history",required=True);a.add_argument("--config",required=True);a.add_argument("--base-freeze-manifest",required=True);a.add_argument("--completeness-report",required=True);a.add_argument("--delta-manifest",required=True);a.add_argument("--composite-features",required=True);a.add_argument("--evaluation",required=True);a.add_argument("--gates",required=True);a.add_argument("--destination",required=True);a.add_argument("--current-link");a.add_argument("--extra-file",action="append",default=[]);a.add_argument("--no-backup",action="store_true")
 x=p.parse_args(argv)
 if x.cmd=="init-metadata":
  from .data import init_metadata;r=init_metadata(x.train_ready,x.out,x.virus_dir,x.include_virus);show(r);raise SystemExit(0 if r["ready"] else 2)
 if x.cmd=="audit":
  from .data import audit_tsv;r=audit_tsv(x.metadata_tsv);show(r);raise SystemExit(0 if r["ok"] else 2)
 if x.cmd=="prepare":
  from .data import prepare;show(prepare(x.train_ready,x.out,x.tfidf,x.version,x.metadata_tsv,x.virus_dir,x.unknown_euk));return
 if x.cmd=="train":
  from .train import train;show({"best_root_macro_f1":train(x.features,x.out,x.epochs,x.batch,x.lr,tuple(map(int,x.hidden.split(','))),x.dropout,x.device)});return
 if x.cmd=="classify":
  from .infer import classify;classify(x.checkpoint,x.tfidf,x.input,x.output,x.batch,x.device,min_len=x.min_len);return
 if x.cmd=="freeze-base":
  from .freeze import freeze_base;show(freeze_base(x.train_ready,x.tfidf,x.metadata_tsv,x.out));return
 if x.cmd=="verify-freeze":
  from .freeze import verify_freeze;r=verify_freeze(x.manifest,x.mode);show(r);raise SystemExit(0 if r["ok"] else 2)
 if x.cmd=="audit-euk":
  from .completeness import audit_euk_completeness;show(audit_euk_completeness(x.base_metadata,x.out,source_index=x.source_index,pool_metadata=x.pool_metadata,taxdump_dir=x.taxdump_dir,overrides_tsv=x.overrides_tsv));return
 if x.cmd=="build-euk-delta":
  from .delta import build_euk_delta;r=build_euk_delta(x.base_train_ready,x.base_metadata,x.candidate_root,x.candidate_metadata,x.out,audit_report=x.audit_report,base_freeze_manifest=x.base_freeze_manifest,verify_base=x.verify_base,target_groups=targets(x.target_groups),min_per_leaf_split=x.min_per_leaf_split,keep_index=x.keep_index);show(r);raise SystemExit(0 if r["ready"] else 2)
 if x.cmd=="prepare-v231":
  from .features_v231 import prepare_v231_features;r=prepare_v231_features(x.base_features,x.base_train_ready,x.curated_base_metadata,x.delta_root,x.delta_metadata,x.tfidf,x.out,base_freeze_manifest=x.base_freeze_manifest,verify_base=x.verify_base,chunk=x.chunk);show(r);raise SystemExit(0 if r["ready"] else 2)
 if x.cmd=="evaluate-v231":
  from .evaluate import evaluate;show(evaluate(x.truth,x.predictions,x.out));return
 if x.cmd=="gate-v231":
  from .gates import run_gates;r=run_gates(x.completeness_report,x.composite_features,x.evaluation,x.base_metadata,x.delta_metadata,baseline_benchmark=x.baseline_benchmark,current_benchmark=x.current_benchmark,checkpoint=x.checkpoint,max_legacy_regression_pp=x.max_legacy_regression_pp,min_leaf_recall=x.min_leaf_recall,out_path=x.out);show(r);raise SystemExit(0 if r["passed"] else 2)
 if x.cmd=="publish-v231":
  from .release import publish_release;show(publish_release(x.checkpoint,x.training_history,x.config,x.base_freeze_manifest,x.completeness_report,x.delta_manifest,x.composite_features,x.evaluation,x.gates,x.destination,current_link=x.current_link,extra_files=x.extra_file,backup=not x.no_backup));return
if __name__=="__main__":main()
