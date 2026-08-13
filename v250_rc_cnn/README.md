# Tiara2 v2.5.0 RC-CNN residual experiment

This directory contains the algorithm-only v2.5.0 experiment. It does not
change, resplit, or augment the v2.3.2/v2.4.0 training data. The raw-sequence
cache is aligned exactly to the frozen v2.4.0 short-expert row indices for
1000–2499 bp fragments.

## Architecture

- frozen v2.3.2 k7 TF-IDF hierarchical generalist;
- shared-weight forward/reverse-complement multiscale CNN + dilated TCN;
- zero-initialized residual heads for root, Euk, Prok, and organelle logits;
- optional conservative Prok-only local correction selected by grouped OOF.

The first seed completed 50 epochs on 8 RTX 4090 GPUs. Epoch 6 maximized the
mean validation macro-F1 across the four heads. A parent-accession grouped
five-fold audit did not support enabling the Euk, root, or organelle residual
heads. The frozen experimental router therefore:

1. leaves root, Euk, and organelle outputs byte-identical to the base path;
2. considers the residual only when the base root prediction is Prok;
3. applies alpha 0.5 only above the frozen residual-L2 threshold.

This candidate is not a release default. It is retained as a reproducible
algorithm result and as evidence that the current RC-CNN signal is mainly in
the Prok branch.

## Reproduction

```bash
bash v250_rc_cnn/run_v250.sh prepare
bash v250_rc_cnn/run_v250.sh base-logits
bash v250_rc_cnn/run_v250.sh smoke
bash v250_rc_cnn/run_v250.sh train
```

After training, use `export_validation.py`, `extract_parent_groups.py`, and
`audit_local_router.py` to reproduce the grouped-OOF audit. The `release/`
folder contains the selected checkpoint and experimental router manifest.

Large sequence caches, base logits, residual-logit exports, and training logs
are intentionally excluded from Git.
