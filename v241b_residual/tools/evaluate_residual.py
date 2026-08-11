#!/usr/bin/env python3
"""Compare frozen-base and v2.4.1-B residual predictions on validation."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

HEADS=("root","euk","prok","organelle")
SIZES={"root":3,"euk":8,"prok":2,"organelle":2}

def scores(cm):
    tp=np.diag(cm).astype(float); precision=tp/np.maximum(1,cm.sum(0)); recall=tp/np.maximum(1,cm.sum(1))
    f1=2*precision*recall/np.maximum(1e-12,precision+recall)
    return {"accuracy":float(tp.sum()/max(1,cm.sum())),"macro_f1":float(f1.mean()),"rows":int(cm.sum()),"confusion":cm.tolist()}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--features",required=True); p.add_argument("--base-logits",required=True)
    p.add_argument("--labels",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--trainer",required=True)
    p.add_argument("--out",required=True); p.add_argument("--batch-size",type=int,default=1024)
    p.add_argument("--alphas",default="0,0.125,0.25,0.5,0.75,1"); a=p.parse_args()
    spec=importlib.util.spec_from_file_location("v241b_train",a.trainer); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); model=mod.ResidualAdapter(); model.load_state_dict(ck["state_dict"]); model.cuda().eval()
    ds=mod.CropDataset(a.features,a.base_logits,a.labels,"validation")
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True)
    alphas=[float(x) for x in a.alphas.split(",")]; cms={str(alpha):{h:np.zeros((SIZES[h],SIZES[h]),dtype=np.int64) for h in HEADS} for alpha in alphas}; delta_abs={h:[0.0,0] for h in HEADS}
    with torch.inference_mode():
      for x,bflat,yflat in loader:
        x=x.cuda(non_blocking=True); b={h:v.cpu().numpy() for h,v in mod.unpack_base(bflat).items()}; y=mod.unpack_y(yflat); d={h:v.float().cpu().numpy() for h,v in model(x).items()}
        root=y["root"].numpy(); masks={"root":np.ones(len(root),bool),"euk":root==0,"prok":root==1,"organelle":root==2}
        for h in HEADS:
          truth=y[h].numpy(); mask=masks[h] & (truth>=0)
          for alpha in alphas:
            pred=(b[h]+alpha*d[h]).argmax(1); np.add.at(cms[str(alpha)][h],(truth[mask],pred[mask]),1)
          delta_abs[h][0]+=float(np.abs(d[h]).sum()); delta_abs[h][1]+=d[h].size
    result={"version":"2.4.1-B","checkpoint":str(Path(a.checkpoint).resolve()),"checkpoint_epoch":ck["epoch"],
            "metrics_by_alpha":{alpha:{h:scores(cm) for h,cm in heads.items()} for alpha,heads in cms.items()},
            "mean_abs_delta":{h:s/max(1,n) for h,(s,n) in delta_abs.items()}}
    Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
