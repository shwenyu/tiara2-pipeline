# Tiara2 v2.4.0 experimental multi-expert

This bundle presents the frozen v2.3.2 long model and the v2.4.0-B short model
through one inference interface. The caller supplies one FASTA; routing is
automatic and predictions are restored to input order.

- sequence length `<2500 bp`: short expert
- sequence length `>=2500 bp`: frozen v2.3.2 long expert

```bash
python -m tiara.hierarchical classify \
  --bundle tiara/models/hierarchical-models-v2.4.0-multi-expert/model_manifest.json \
  -i contigs.fasta \
  -o predictions.tsv
```

The manifest pins both checkpoint hashes and the shared k7 TF-IDF path. The
router verifies hashes before inference, batches each expert separately, and
emits a single TSV with `length_bp` and `expert` provenance columns.

This release is experimental. The frozen short validation comparison is
included, but the external four-task short benchmark is still pending. It must
not replace the v2.3.2 current release until its benchmark gates pass.
