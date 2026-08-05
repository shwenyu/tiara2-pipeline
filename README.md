# Tiara2 v2.2.0 training pipeline

Tiara2 的可复现数据整理、TF-IDF、GPU 超参数搜索、最终训练、质量闸门与模型发布流程。

**v2.2.0 是基于已通过 benchmark 的 v2.1.3 做的单因素 k-range 实验。** 数据、采样、网络搜索空间、训练预算和模型塌缩闸门保持不变，只修改两组 k：

```yaml
train:
  k_first:  [5, 6, 7]
  k_second: [5, 6, 7, 8]
```

详细变更见 [`V2.2.0_CHANGES.md`](V2.2.0_CHANGES.md)。

---

## 1. 实验边界

### 唯一实验变量

| Stage | v2.1.3 | v2.2.0 |
|---|---|---|
| First-stage k | `[4, 5, 6]` | `[5, 6, 7]` |
| Second-stage k | `[4, 5, 6, 7]` | `[5, 6, 7, 8]` |
| 模型数量 | 3 + 4 | 3 + 4（不变） |

选择依据：v2.1.3 HP 中 k4 在两个 stage 都是最弱候选；first k5/k6 最好，second 从 k4 到 k7 持续提升。保持 3/4 个模型数量不变，可把 benchmark 差异主要归因到 k-range，而不是模型数量。

### 明确保持不变

- 输入语料：`/ssd/shouhanyu/Tiara2/corpus_ready_v2_1_3_bpbalanced`，逐字节复用。
- `train` 与 `validation` 均已 bp-balanced；`test` 保持原始镜像。
- `final_include_validation: false`。
- `hp_val_balance: equal`、`class_weight: balanced`。
- HP / final fit / publish 三层 collapse gate。
- HP 预算、网络架构网格、epochs、batch、row cap 均不变。
- 不重新执行 acquire / curate / chop / dedup / regroup / bp-balance。

> v2.2.0 不得把 `fragment_bp_balance.output_root` 改成由 `version_tag` 自动生成的新目录；否则会重采样语料，破坏 k-only 实验归因。

---

## 2. 版本和路径

```yaml
corpus_tag:  v2_1_2
version_tag: v2_2_0
model_tag:   v2.2.0

fragment_bp_balance:
  output_root: /ssd/shouhanyu/Tiara2/corpus_ready_v2_1_3_bpbalanced

train:
  train_ready: /ssd/shouhanyu/Tiara2/corpus_ready_v2_1_3_bpbalanced
  tfidf_dir: /ssd/shouhanyu/Tiara2/tfidf_v2_2_0
  feature_cache: /ssd/shouhanyu/Tiara2/feature_cache_v2_2_0
  seq_pack: /ssd/shouhanyu/Tiara2/seqpack_v2_2_0
  out_models: /data/shouhanyu/Tiara2/models_v2.2.0_optimizedHP_gpu
```

必须新建 v2.2.0 的 TF-IDF、feature cache 和模型目录。k 变化会改变 `4**k` 特征维度，因此禁止把 v2.1.3 的 TF-IDF/cache 软链接给 v2.2.0。

---

## 3. 动态 k 已贯穿完整链路

v2.1.3 虽然在 config 中暴露了 `k_first/k_second`，但部分叶子脚本仍硬编码旧列表。v2.2.0 已改为单一配置源：

1. `train_tfidf_optimized.py` 接收 `--k-first/--k-second`，并把列表写入 signature。
2. feature cache 和每个 k 的 HP search 使用 config 列表。
3. `train_models_gpu.py` 使用同一列表加载 HP JSON、cache、TF-IDF 并创建 final jobs。
4. publish 从 `len(train.k_first/k_second)` 自动计算模型数量。
5. config validator 拒绝空列表、重复值、非整数、乱序和超出 `[1,8]` 的 k，并检查 publish count mirror。

因此之后可以改变每组 k 的数量，但必须只改 `config/config.yaml` 的列表和对应 count mirror；训练、校验与发布不再需要手动改硬编码循环。

---

## 4. 资源预估

特征维度为 `4**k`：

| k | 维度 |
|---:|---:|
| 5 | 1,024 |
| 6 | 4,096 |
| 7 | 16,384 |
| 8 | 65,536 |

根据 v2.1.3 实测规模估算：

- first k7 train+validation cache 约 **184 GiB**；
- second k8 train+validation cache 约 **62 GiB**；
- v2.2.0 总 feature cache 预计 **310–330 GiB**。

运行前检查 `/ssd` 至少保留 config 中 `storage.min_fast_free_gb` 要求的空间。当前不建议 first-stage k8。

---

## 5. 运行前检查

```bash
cd ~/tiara2_pipeline
source /data/shouhanyu/envs/tiara2/bin/activate

# 解析并校验所有模板、k 列表和路径
python3 - <<'PY'
from tiara2 import config
c = config.load('config/config.yaml')
print('version:', c['version_tag'], c['model_tag'])
print('input:', c['train']['train_ready'])
print('k:', c['train']['k_first'], c['train']['k_second'])
print('tfidf:', c['train']['tfidf_dir'])
print('cache:', c['train']['feature_cache'])
PY

# train/validation 必须是真目录，test 应保持镜像；不得出现断链
python3 -m tiara2.cli status --config config/config.yaml

# 先看完整命令，不产生文件
python3 -m tiara2.cli run --config config/config.yaml --only train --dry-run
```

预期关键输出：

```text
version: v2_2_0 v2.2.0
input: /ssd/shouhanyu/Tiara2/corpus_ready_v2_1_3_bpbalanced
k: [5, 6, 7] [5, 6, 7, 8]
tfidf: /ssd/shouhanyu/Tiara2/tfidf_v2_2_0
cache: /ssd/shouhanyu/Tiara2/feature_cache_v2_2_0
```

如果 input 显示为 `corpus_ready_v2_2_0_bpbalanced`，不要继续运行——这表示配置意外准备重采样语料，实验不再是 k-only。

---

## 6. 训练与发布

```bash
# 只运行训练阶段；复用 v2.1.3 bp-balanced corpus
python3 -m tiara2.cli run \
  --config config/config.yaml \
  --only train

# train 完成且三层质量闸门通过后发布
python3 -m tiara2.cli run \
  --config config/config.yaml \
  --only publish
```

训练顺序：

```text
TF-IDF (k5/6/7 first; k5/6/7/8 second)
  → shared seqpack
  → feature cache per stage×k
  → HP search per stage×k
  → first-stage HP gate
  → final model training + collapse probe
  → publish quality gate
  → atomic publish + SHA256SUMS + run report
```

---

## 7. 必须通过的质量门槛

- First-stage 最佳 HP `mean_f1 >= 0.85`。
- HP validation 使用等类抽样。
- 任何候选或 final model 单类预测占比不得超过 `0.98`。
- collapsed 候选不能获胜；全部 collapsed 时训练直接失败。
- final collapse probe 失败时不得写 `.pkl`。
- publish 前必须存在完整 HP JSON 与 `training_manifest.json`。
- `publish.tfidf_src` 必须等于 `train.tfidf_dir`。
- 发布模型数必须等于 k 列表长度。

这些门槛是 v2.1.2 `failed_model_collapse` 的永久回归保护，不得为赶进度而关闭。

---

## 8. 改变 k 数量的方法（训练侧）

代码现在支持不同数量，例如精简实验：

```yaml
train:
  k_first:  [5, 6]
  k_second: [5, 6, 7]

publish:
  first_count: 2
  second_count: 3
```

注意：改变数量会同时改变计算量和可用模型数。如果目标是分析“哪个 k 更好”，应保持数量不变；如果目标是压缩训练成本，可以改变数量，但应作为独立实验版本归档。

---

## 9. 推理（统一调用层）

推理不再需要手工拼模型路径。模型集合从 `training_manifest.json` +
`tfidf_manifest.json` + 已发布目录里**发现**，代码中没有任何 k 列表硬编码。

```bash
# 只看会用哪套模型，不跑推理
python -m tiara2.cli classify --emit-manifest infer_v2.2.0.json

# 正常推理
python -m tiara2.cli classify -i contigs.fa -o out.tsv --probabilities --threads 24

# 锁定 k / 阈值
python -m tiara2.cli classify -i contigs.fa --k-first 6 --k-second 8 \
    --prob-cutoff-first 0.938553

# 用固定清单复现一次 benchmark
python -m tiara2.cli classify --manifest infer_v2.2.0.json -i contigs.fa -o out.tsv
```

三件事必须知道：

1. **k=8 需要 `bow.MAX_K = 8`**。原始 tiara 的 `MAX_K = 7` 会让二阶段 k=8
   在第一次调用时抛 `k-mer length not supported!`。已修复。
2. **每阶段只加载一个网络**，7 个 `.pkl` 不是 ensemble。默认按
   `mean_f1` 选最优，选中的会打印并写进清单。
3. **`prob_cutoff` 必须在新 benchmark 上重新标定**，默认值只够 smoke test。

### 9.1 作为流水线阶段运行（推荐）

推理已经接在 `publish` 之后，是流水线的**最后一个 stage**，调用方式与前面所有
步骤完全一致：

```bash
# 训练 + 发布（推理默认关闭，publish 之后就停）
python -m tiara2.cli run --from train --to publish

# 人工确认参数包无误后，再开推理
python -m tiara2.cli run --only infer --set infer.enabled=true

# 或者确认后直接一路跑到底
python -m tiara2.cli run --from train
```

`infer` 继承与其它 stage 相同的语义：`--dry-run` 只解析并打印将要使用的模型集合
（不加载 torch）、`--force` 忽略断点、resume 指纹**同时覆盖输入 fasta 和已发布的
参数包**——重新 publish 一版新参数会自动让旧预测失效，而不是静默跳过。

**默认 `infer.enabled: false`**。publish 成功不应该就自动开始跑几百 GiB 的分类；
预期流程是 `train → publish → 确认参数包 → 开推理`。

### 9.2 输入输出接口

```yaml
infer:
  enabled: false
  inputs: []                   # 文件 / 整个目录 / glob，三种混写都行
  out_dir: "{results_root}/predictions_{output_tag}"
  k_first: null                # null = 按 mean_f1 选最优
  k_second: null
  prob_cutoff:
    first: null                # 必须按新 benchmark 重新标定
    second: null
  min_len: 3000
  threads: null                # null -> resources.threads
  batch_records: 512
  device: null                 # null -> 有 cuda 用 cuda
  probabilities: true
  to_fasta: []                 # mit pla bac arc euk unk pro org all
  gzip: false
  emit_manifest: true
  manifest: null               # 复用冻结的清单，逐字节可复现
```

加一个 benchmark panel 是**改配置，不是改代码**。每个输入产出
`<name>.tsv` + `log_<name>.txt`，整批再产出 `inference_summary.json` 和
`inference_manifest_<tag>.json`。

配置错误在 **加载时**就报错，而不是等训练跑完才发现：`enabled: true` 但
`inputs` 为空、要求一个没训练过的 k、阈值不在 (0,1)、`to_fasta` 写错类名、
`device` 不是 cpu/cuda,全部拒绝。

细节见 `INFERENCE_NOTES.md`。

## 10. 测试

```bash
python3 -m compileall -q tiara tiara2 scripts tests
bash -n scripts/*.sh tests/*.sh

# 有 pytest 时
python3 -m pytest tests -q

# 最小流程冒烟测试
bash tests/run_smoke_test.sh
```

---

## 11. 关键文件

| 文件 | 用途 |
|---|---|
| `config/config.yaml` | 唯一运行配置和版本化路径 |
| `tiara2/config.py` | 配置解析与 k/path 约束 |
| `tiara2/stages/train.py` | 按 config 遍历 stage×k |
| `tiara2/model_backend.py` | 向 TF-IDF/HP/final trainer 透传 k |
| `tiara/training/train_tfidf_optimized.py` | config-driven TF-IDF |
| `tiara/training/featurize_cache.py` | stage×k feature cache |
| `tiara/training/hyperparameter_search_gpu.py` | HP 与 collapse 筛查 |
| `tiara/training/train_models_gpu.py` | config-driven final jobs 与探针 |
| `tiara2/stages/publish.py` | 发布质量闸门与动态模型计数 |
| `V2.2.0_CHANGES.md` | 本版本变更与资源说明 |

## 历史

- v2.1.2：validation 未平衡且被并入最终拟合，参数包塌缩为恒 `eukarya`，已归档。
- v2.1.3：修复数据平衡、路径/软链和三层质量门槛，四任务平均 F1 84.08%，作为 v2.2.0 的固定基线。
- v2.2.0：只测试 k-range 上移，等待统一 benchmark。


## v2.3.0 bootstrap: freeze and cleanup

Before the hierarchical redesign, freeze v2.2.0 with `tiara2 freeze-baseline`,
then inspect/apply `tiara2 cleanup-baseline`. See `V2.3.0_BOOTSTRAP.md`.
Cleanup is always dry by default and destructive upstream release requires
`--apply --yes --acknowledge-upstream-loss`.
