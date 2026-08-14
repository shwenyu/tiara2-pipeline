# Tiara2 v2.6.1 BioSignal candidate

This portable bundle keeps the frozen v2.5.1 hierarchical model as the generalist and adds one lightweight 60-feature ExtraTrees Euk expert. `python -m tiara.hierarchical classify --bundle model_manifest.json` computes both signals and applies a length-conditioned, Euk-only soft residual automatically.

The router can only add Euk evidence; it never demotes a base Euk prediction. Parameters were frozen on clean validation before the strict and sens-like-unseen internal panels were read. The requested clean-validation improvement gate was not met, so this is an external-benchmark candidate rather than a final release.

Example:

```bash
python -m tiara.hierarchical classify \
  --bundle tiara/models/hierarchical-models-v2.6.1-biosignal/model_manifest.json \
  --input contigs.fasta --output predictions.tsv
```
