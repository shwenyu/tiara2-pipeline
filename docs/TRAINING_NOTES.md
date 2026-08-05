# Training integration + optimization review

Your `train_tfidf_optimized.py`, `hyperparameter_search_gpu.py` and
`train_models_gpu.py` are already well parallelized. Below is (1) how they now
plug into the framework and (2) concrete, low-risk optimization ideas ranked by
payoff. Nothing here has been silently applied to your algorithm code -- these
are proposals with pointers.

## How training now fits the framework

- The three scripts are the **vendored `tiara/training/` package**, unchanged.
- The `train` stage reproduces the exact `05_train_v2b` order
  (flat -> TF-IDF -> HP search first/second x k -> final NNet) but:
  - reads all knobs from the single `config.yaml` (`train.*`),
  - calls them via `python -m tiara.training.*` with the repo put on
    `PYTHONPATH` **relative to the package** (no `$HOME/tiara-v2` hardcode),
  - goes through the `ModelBackend` contract so a future algorithm swap does not
    touch the pipeline.
- Resume is preserved: each script keeps its own manifest/`.partial.json`/model
  file resume, and the stage adds its input-fingerprint skip on top.

## Optimization opportunities (ranked)

### 1. Cache k-mer features + parallelize + one-pass-multi-k  ✅ DONE
Implemented in `tiara/training/featurize_cache.py` and wired through the
`ModelBackend.build_features` verb + the `train` stage. Three optimizations run
together:
- **Reuse**: features are written to the persistent `train.feature_cache`
  (default `{base}/feature_cache_{input_tag}`), keyed by (stage, k, split) plus
  an input fingerprint (per-file size). `--resume`/restart never recomputes a
  valid (stage, k, split).
- **Parallel**: each FASTA is byte-range sharded into `hp_feat_workers*4` pieces
  and featurized across `gpu.hp_feat_workers` (default 16) processes, spreading
  the multi-TB read+parse+count over many CPUs.
- **One pass, all k**: the `train` stage calls `build_features` once per stage,
  building every k (first 4/5/6, second 4/5/6/7) in a SINGLE pass over the data;
  the per-k `hp_search` then hits the cache instantly.

Costs: peak build RAM ~ `hp_feat_workers * hp_feat_chunk * 4**max_k * 4B`; cache
disk is the sum over k of `n * 4**k * 4B` (large for k6 first-stage). The cache
lives on /data and is safe to delete after training (rebuilt on demand). Tune
`gpu.hp_feat_workers` / `gpu.hp_feat_chunk` to trade speed for load.

> Still open: have `train_models_gpu` (final NNet training) read the SAME cache
> instead of recomputing features for the winning k. The cache format
> (`<feature_cache>/<stage>/k<k>/{train,val}_{X.f32,y.i64}` + `meta.json`) is
> ready to be consumed there too.

### 2. Overlap feature extraction (CPU) with GPU training
Feature extraction is CPU/numba; NNet fit is GPU. Right now they are sequential
per candidate. A single-slot producer thread that pre-builds the NEXT k's memmap
while the current k trains would hide most extraction latency. Low risk because
the two use disjoint resources.

### 3. Use the shared `GpuScheduler` instead of per-script schedulers
`tiara2/resources.py::GpuScheduler` is a portable, unit-tested reimplementation
of the bash/embedded packing logic (seed idle cards first, then pack under
shared caps, honor `max_tasks_per_gpu`). Migrating the two training scripts to
import it removes ~200 lines of duplicated scheduling and gives one place to
tune packing. Do this incrementally (search first, then final).

### 4. Pin + prefetch host->device transfers
For the dense TF-IDF matrices, `pin_memory=True` on the loader plus
`non_blocking=True` copies overlaps H2D with compute. Small, safe win on the
4090s given batch sizes of 512/4096/8192.

### 5. Consider bf16/tf32 for the MLP
`torch.backends.cuda.matmul.allow_tf32 = True` (Ampere+) is essentially free for
these 2048-wide MLPs and typically 1.2-1.5x. bf16 autocast is also safe for
NLLLoss+Softmax heads; keep an fp32 master copy. Gated behind a config flag so
the v1.1 baseline stays bit-comparable.

### 6. TF-IDF: single pass over gzip with a shared record iterator
`train_tfidf_optimized` already does one pass per stage; if disk is the limiter
for the 4 TiB corpus, a shared reader feeding both stages' document-frequency
accumulators in one physical pass halves gzip decode. Higher effort; only worth
it if TF-IDF wall-time is material vs the NNet stages.

## What to measure first
Add `--debug` timing around each stage (the stage logger already timestamps).
Start with #1 and #3: highest payoff, lowest risk, and both are enabled by
interfaces already in the framework.
