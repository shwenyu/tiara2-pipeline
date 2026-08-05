#!/usr/bin/env python3
from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tiara2 import baseline


def write(p, text='x'):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text); return p


def cfg(root):
    repo=root/'repo'; base=root/'cold'; fast=root/'fast'
    nnet=repo/'tiara/models/nnet-models-v2.2.0'; tfidf=repo/'tiara/models/tfidf-models-v2.2.0'
    write(nnet/'first_k-7_hidden_1-1_hidden_2-none_lr-0.1_dropout-0.1_epochs-1.pkl')
    write(tfidf/'k7-first-stage/model.npy')
    tr=fast/'train_ready'; write(tr/'train/eukarya.fasta','>a\nACGT\n'); write(tr/'validation/eukarya.fasta','>b\nACGT\n')
    c={'base':str(base),'fast_base':str(fast),'model_tag':'v2.2.0','version_tag':'v2_2_0','corpus_tag':'c',
       'results_root':str(base/'results'),'log_dir':str(base/'logs/pipeline'),'checkpoint_root':str(base/'checkpoints'),
       'work_root':str(fast/'work'),'source_ready':str(fast/'source'),'corpus_ready':str(fast/'corpus'),
       'train':{'train_ready':str(tr),'log_dir':str(base/'logs/train'),'feature_cache':str(fast/'cache'),
                'seq_pack':str(fast/'pack'),'flat_data':str(fast/'flat'),'out_models':str(base/'models_src'),
                'tfidf_dir':str(fast/'tfidf_src')},
       'publish':{'nnet_dest':str(nnet),'tfidf_dest':str(tfidf),'report':{}},
       'versioning':{},'baseline_freeze':{'training_splits':['train','validation']}}
    write(fast/'cache/x', 'x'*20); write(fast/'pack/x','x'*10); write(fast/'source/x','raw')
    return c,repo

class TestBaseline(unittest.TestCase):
    def test_freeze_and_safe_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); c,repo=cfg(root); out=root/'cold/baselines/v2.2.0'
            frozen=baseline.freeze(c,repo,freeze_dir=out)
            self.assertTrue((out/baseline.FREEZE_FILE).is_file())
            self.assertFalse(frozen['metadata_only'])
            plan=baseline.cleanup_plan(c,repo,out,profile='safe')
            keys={x['key'] for x in plan['items']}
            self.assertIn('feature_cache',keys); self.assertIn('seq_pack',keys)
            self.assertNotIn('source_ready',keys)
            baseline.apply_cleanup(plan,apply=True,yes=True)
            self.assertFalse(Path(c['train']['feature_cache']).exists())
            self.assertTrue(Path(c['train']['train_ready']).exists())
            self.assertTrue((out/'cleanup_receipt.json').is_file())
    def test_upstream_needs_double_ack(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); c,repo=cfg(root); out=root/'cold/baselines/v2.2.0'
            baseline.freeze(c,repo,freeze_dir=out)
            plan=baseline.cleanup_plan(c,repo,out,profile='reproducible',release_upstream=True)
            with self.assertRaises(baseline.BaselineError):
                baseline.apply_cleanup(plan,apply=True,yes=True)
            baseline.apply_cleanup(plan,apply=True,yes=True,acknowledge_upstream_loss=True)
            self.assertFalse(Path(c['source_ready']).exists())
            self.assertTrue(Path(c['train']['train_ready']).exists())
    def test_refuses_missing_models(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); c,repo=cfg(root); Path(c['publish']['nnet_dest']).unlink() if False else None
            import shutil; shutil.rmtree(c['publish']['nnet_dest'])
            with self.assertRaises(baseline.BaselineError):
                baseline.freeze(c,repo,freeze_dir=root/'freeze')

if __name__=='__main__': unittest.main(verbosity=2)
