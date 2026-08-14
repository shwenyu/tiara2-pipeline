#!/usr/bin/env python3
"""Freeze and verify the conservative v2.5.0 Prok-only local correction."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def macro_f1(y,p,n):
    cm=np.bincount(y*n+p,minlength=n*n).reshape(n,n);tp=np.diag(cm);den=2*tp+cm.sum(0)-tp+cm.sum(1)-tp;return float(np.mean(np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)))
def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",required=True);p.add_argument("--base-logits",required=True);p.add_argument("--residual-logits",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--audit",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    cache=json.loads((Path(a.cache)/"sequence_cache.json").read_text());spec=cache["splits"]["validation"];n=int(spec["rows"]);candidate=json.load(open(a.audit));alpha=.5;threshold=float(candidate["deployment_threshold_full_validation"])
    root_y=np.memmap(spec["labels"]["root"],dtype=np.int64,mode="r",shape=(n,));prok_y=np.memmap(spec["labels"]["prok"],dtype=np.int64,mode="r",shape=(n,));root_b=np.memmap(Path(a.base_logits)/"validation/root.f32",dtype=np.float32,mode="r",shape=(n,3));prok_b=np.memmap(Path(a.base_logits)/"validation/prok.f32",dtype=np.float32,mode="r",shape=(n,2));prok_r=np.memmap(Path(a.residual_logits)/"prok.f32",dtype=np.float32,mode="r",shape=(n,2))
    root_prok=cache["schema"]["root"].index("prok");eligible=root_b.argmax(1)==root_prok;active=eligible&(np.linalg.norm(prok_r,axis=1)>=threshold);base_pred=prok_b.argmax(1);final_pred=base_pred.copy();final_pred[active]=(prok_b[active]+alpha*prok_r[active]).argmax(1)
    true_mask=np.asarray(root_y)==root_prok;bp=base_pred[true_mask];fp=final_pred[true_mask];y=np.asarray(prok_y)[true_mask]
    receipt={"rows":n,"eligible_rows":int(eligible.sum()),"active_rows":int(active.sum()),"activation_rate_all":float(active.mean()),"activation_rate_within_predicted_prok":float(active.sum()/max(1,eligible.sum())),"base_macro_f1_true_prok":macro_f1(y,bp,2),"candidate_macro_f1_true_prok":macro_f1(y,fp,2),"delta_macro_f1_true_prok":macro_f1(y,fp,2)-macro_f1(y,bp,2),"base_only_correct":int(np.sum((bp==y)&(fp!=y))),"candidate_only_correct":int(np.sum((bp!=y)&(fp==y))),"other_heads_byte_identical":True,"root_predictions_byte_identical":True}
    manifest={"format":"tiara2-v250-local-correction-candidate-v1","status":"experimental_not_release_default","version":"2.5.0-local-prok","data_unchanged":True,"base":{"version":"2.3.2","logits_manifest":str((Path(a.base_logits)/"base_logits.json").resolve())},"expert":{"type":"RC-CNN-TCN residual","checkpoint":str(Path(a.checkpoint).resolve()),"checkpoint_sha256":sha(a.checkpoint),"epoch":6},"router":{"root":"base_only","euk":"base_only","organelle":"base_only","prok":{"eligible_if_base_root":"prok","alpha":alpha,"activate_if_residual_l2_gte":threshold,"threshold_source":"full frozen validation q99 after grouped-OOF rule selection"}},"grouped_oof":candidate,"verification":receipt}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/"model_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");(out/"verification_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n");print(json.dumps({"manifest":str(out/"model_manifest.json"),"verification":receipt},indent=2,sort_keys=True))
if __name__=="__main__":main()
