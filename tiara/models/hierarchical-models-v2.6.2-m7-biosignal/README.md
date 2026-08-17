# Tiara2 v2.6.2 M7 + BioSignal candidate

This bundle keeps the published v2.4.1-M7 checkpoint as the generalist and leaf classifier, then adds the v2.6.1 lightweight 60-feature ExtraTrees Euk expert. The standard `python -m tiara.hierarchical classify --bundle ...` interface applies a conservative length-conditioned residual automatically. `--bundle` accepts either this directory or its `model_manifest.json` file.

The router can only promote a non-Euk M7 prediction to Euk; it never demotes M7 Euk predictions. All leaf predictions, including Fungi, are produced by the M7 branch heads. Gate parameters were frozen before the strict and sens-like-unseen internal panels were evaluated.

```bash
python -m tiara.hierarchical classify \
  --bundle tiara/models/hierarchical-models-v2.6.2-m7-biosignal \
  --input contigs.fasta --output predictions.tsv
```

Run the command inside the Tiara2 environment (on Zhenglab: `conda run -p /data/shouhanyu/envs/tiara2 ...`) and set `PYTHONPATH` to the repository root when invoking it outside an installed checkout.

Both frozen internal panels improve root macro-F1, Euk recall and Fungi recall versus M7. External benchmark status remains pending.
