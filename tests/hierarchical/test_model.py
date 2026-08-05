import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
try: import torch
except Exception: torch=None
@unittest.skipIf(torch is None,'torch absent')
class T(unittest.TestCase):
 def test_forward_and_masked_loss(self):
  from tiara.hierarchical.model import HierarchicalClassifier,MaskedHierarchicalLoss
  m=HierarchicalClassifier(16,{'root':3,'euk':8,'prok':2,'organelle':2},hidden=(8,4));x=torch.randn(6,16);z=m(x);self.assertEqual(tuple(z['root'].shape),(6,3))
  y={'root':torch.tensor([0,1,2,0,1,2]),'euk':torch.tensor([0,-1,-1,1,-1,-1]),'prok':torch.tensor([-1,0,-1,-1,1,-1]),'organelle':torch.tensor([-1,-1,0,-1,-1,1])};loss,parts=MaskedHierarchicalLoss({'euk_nuclear':0,'prok':1,'organelle':2})(z,y);self.assertTrue(torch.isfinite(loss));self.assertEqual(set(parts),{'root','euk','prok','organelle'})
if __name__=='__main__':unittest.main(verbosity=2)
