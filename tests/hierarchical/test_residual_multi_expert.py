import json
from pathlib import Path

def test_v241b_bundle_contract():
 root=Path(__file__).resolve().parents[2]
 manifest=json.loads((root/"tiara/models/hierarchical-models-v2.4.1-residual/model_manifest.json").read_text())
 assert manifest["format"]=="tiara2-residual-multi-expert-v1"
 assert manifest["router"]=={"type":"deterministic_length_residual","residual_if_length_lt_bp":2500,"alpha":0.125}
 assert set(manifest["tfidf"])=={"k4","k5","k6","k7"}
