"""Shared encoder, branch heads and masked hierarchical loss."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class HierarchicalClassifier(nn.Module):
    def __init__(self, dim_in, head_sizes, hidden=(2048,1024), dropout=.2):
        super().__init__(); layers=[]; last=int(dim_in)
        for width in hidden:
            layers += [nn.Linear(last,int(width)),nn.ReLU(inplace=True),nn.Dropout(float(dropout))]; last=int(width)
        self.encoder=nn.Sequential(*layers)
        self.heads=nn.ModuleDict({name:nn.Linear(last,int(size)) for name,size in head_sizes.items()})
    def forward(self,x):
        z=self.encoder(x); return {name:head(z) for name,head in self.heads.items()}

class MaskedHierarchicalLoss(nn.Module):
    def __init__(self, root_index, weights=None, ignore_index=-1, label_smoothing=0.0):
        super().__init__(); self.root_index=dict(root_index); self.weights={"root":1.,"euk":1.,"prok":.5,"organelle":.5,**(weights or {})}; self.ignore_index=ignore_index; self.smoothing=float(label_smoothing)
    def forward(self, logits, targets):
        terms={}; terms["root"]=F.cross_entropy(logits["root"],targets["root"],label_smoothing=self.smoothing)
        mapping={"euk":"euk_nuclear","prok":"prok","organelle":"organelle"}
        for head,root_name in mapping.items():
            if head not in logits: continue
            mask=(targets["root"]==self.root_index[root_name]) & (targets[head]!=self.ignore_index)
            if bool(mask.any()): terms[head]=F.cross_entropy(logits[head][mask],targets[head][mask],label_smoothing=self.smoothing)
        total=sum(self.weights[k]*v for k,v in terms.items())
        return total, {k:float(v.detach()) for k,v in terms.items()}

def probabilities(logits, temperatures=None):
    temperatures=temperatures or {}; return {k:torch.softmax(v/float(temperatures.get(k,1.0)),dim=1) for k,v in logits.items()}
