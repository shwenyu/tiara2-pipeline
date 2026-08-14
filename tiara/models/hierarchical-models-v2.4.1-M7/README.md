# Tiara2 v2.4.1-M7 benchmark candidate

This directory contains the validation-selected k7 hierarchical checkpoint from
the preregistered M7 three-seed replication. All seeds completed 50 epochs and
passed the root/leaf validation gate. Seed 44, epoch 45 was selected by the
preregistered maximin objective across root, Euk, Prok, and organelle macro-F1
deltas. The external benchmark was not used for training or selection.

Run the standard hierarchical interface:

```bash
python -m tiara.hierarchical classify \
  --checkpoint tiara/models/hierarchical-models-v2.4.1-M7/multiobjective_model.pt \
  --tfidf tiara/models/tfidf-models-v2.2.0/k7-first-stage \
  --input query.fasta \
  --output predictions.tsv
```

`model_manifest.json` records the selected seed, all three validation results,
the release SHA256, and the pending external-benchmark gate.
