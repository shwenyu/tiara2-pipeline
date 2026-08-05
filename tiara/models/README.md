# `tiara/models/` — published trained params (the closed loop lands here)

This directory is the **in-package home** for trained parameters. It is where the
`publish` stage copies params after training finishes:

```
tiara/models/
├── nnet-models-<model_tag>/     # first_*.pkl (len(k_first)) + second_*.pkl (len(k_second)) + training_manifest.json + SHA256SUMS
└── tfidf-models-<model_tag>/    # one k<k>-first/second-stage folder per configured k / (model.npy + params.txt) + tfidf_manifest.json + SHA256SUMS
```

Why here:

- **Closed loop.** Code lives at `~/` (this repo); runs/data live at `/data`.
  Training writes to `/data/.../models_<tag>_optimizedHP_gpu` and
  `/data/.../tfidf_<tag>`. The `publish` stage snapshots those back into this
  folder so the repo at `~/` carries a runnable copy.
- **Classic tiara layout.** `tiara.src.classification` loads models from
  `tiara/models/...`, so inference works straight from here.
- **Future packaging.** When `tiara2` is packaged, add this folder to
  `package-data` and inference points directly at the in-package params — no
  `/data` dependency at deploy time.

Generated params are intentionally NOT committed by the tooling (no git
add/commit/push). Large `.pkl` / `.npy` files are usually git-ignored or tracked
via LFS — decide per your repo policy.

Manual one-offs are still available:
`scripts/copy_v2_models_to_git.sh`, `scripts/copy_v2_tfidf_to_git.sh`.
The pipeline path is `tiara2 run --only publish` (see `config.publish`).
