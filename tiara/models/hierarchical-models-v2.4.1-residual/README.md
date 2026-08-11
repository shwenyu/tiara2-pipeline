# Tiara2 v2.4.1-B residual multi-expert candidate

This bundle keeps the v2.3.2 k7 model frozen and applies a k4+k5+k6 residual
adapter only to sequences shorter than 2500 bp. The unified classifier selects
the route automatically from sequence length.

```bash
python -m tiara.hierarchical classify \
  --bundle tiara/models/hierarchical-models-v2.4.1-residual/model_manifest.json \
  --input query.fasta --output predictions.tsv
```

The selected fixed blend is `base_logits + 0.125 * residual_logits`. At 2500
bp and above, the output is the frozen v2.3.2 base. The candidate passed strict
zero-initialization equivalence and validation non-inferiority across all four
hierarchical heads. Its external four-task benchmark remains pending.
