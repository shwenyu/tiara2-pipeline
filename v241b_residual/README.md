# Tiara2 v2.4.1-B — continuous multi-scale residual expert

This experiment freezes the v2.3.2 k7 generalist and changes only the short
representation/model form:

- deterministic continuous crops in `800–2499 bp`;
- explicit extra coverage for `1250–1750 bp`;
- a new `k4+k5+k6` first-stage TF–IDF family trained on those crops;
- a zero-initialized residual adapter over frozen v2.3.2 logits;
- no learned gate (`alpha=1` below 2500 bp, `alpha=0` at/above 2500 bp).

The crop stage preserves the frozen train/validation split and original record
ID as the first FASTA header token. It never changes the v2.3 corpus.

```bash
bash scripts/run_v241b.sh crops
bash scripts/run_v241b.sh tfidf
```
