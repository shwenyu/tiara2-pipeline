# Tiara2 Pipeline Architecture

本文定义当前代码的真实边界、数据契约和 v2.1 实验范围。未来设计与已实现功能严格分开。

## 1. 当前系统边界

Tiara2 pipeline 负责：

- 数据索引与下载状态管理；
- Tiara S1 index 合并与 deferred download；
- taxonomy 驱动的 species/genus genome 选择；
- FASTA header 类别解析和历史容器重路由；
- fragment length binning、可选 dedup、regroup；
- census、训练采样率、现有 flat 模型训练与发布。

当前不负责：

- 已实现的 two-stage 训练/推理；
- v2.1 bacteria 模型训练；
- deferred Tiara S1 序列的自动下载与物化；
- 从完整 genome 重新生成 v2.1 fragments；
- genome-level ANI 去重的重新执行。

## 2. 版本和数据层

### Corpus layer：`corpus_tag`

覆盖：

```text
index merge → genus plan → source_ready → chop_bin → optional dedup → regroup
→ fragment bp balance
```

当前 tag：

```text
v2_1_breadth
```

### Training layer：`version_tag`

覆盖：

```text
census/class_rates → TF-IDF/features → HP → models → publish
```

当前 tag：

```text
v2_1_breadth_binned95
```

两层必须独立。只改训练参数时不应迫使语料重建。

## 3. 存储层

### COLD `/data`

持久资产：

```text
index/
metadata/
store/
raw/
import/
results_*/
models_*/
logs/
checkpoints/
versions.json
```

### HOT `/ssd`

可重建、高频读取：

```text
train_ready_<corpus_tag>/
corpus_ready_<corpus_tag>/
.work_pipeline_<corpus_tag>/
train_ready_flat_*/
tfidf_*/
seqpack_*/
feature_cache_*/
```

`check_storage.py` 使用设备号检查冷热层，而不只比较路径字符串。

## 4. 核心数据契约

### 4.1 Existing index

来源：

```text
index/selected.tsv
index/download_status.tsv
index/split_assignments.tsv
```

关键字段：

```text
entity_id / assembly_accession / sequence_accession
taxid / species_taxid
entity_type
klass / group_name / organelle_type
split
```

### 4.2 Tiara S1 index

来源：

```text
import/tiara1/tiara1_selection.tsv
```

本轮只参与索引合并。没有本地序列的行进入：

```text
deferred_downloads.tsv
```

### 4.3 Merged index

`merge_training_indexes.py` 输出：

```text
merged_training_index.tsv
current_available_candidates.tsv
deferred_downloads.tsv
merge_training_indexes_report.json
```

Availability 规则按实体类型区分：

- assembly：`download_status ∈ {ok,cached,cached_no_md5}`；
- organelle：`store/organelle/<accession>.fna.gz` 存在；
- Tiara-only：本轮 deferred。

不能用 assembly 的 `download_status` 规则判断细胞器，否则会把本地 mitochondria/plastids 全部排除。

## 5. Genome-level abundance balancing

入口：

```text
current_available_candidates.tsv
split_assignments.tsv
NCBI taxdump
```

算法单位是 genome/accession，不是 FASTA fragment。

### Stage A：species 内选择

对每个：

```text
(split, training_class, species_taxid)
```

只保留一个质量最佳 accession。排序优先级包括 RefSeq category、assembly level、完整性代理、contig count、日期和 accession 稳定 tie-break。

### Stage B：genus cap

对 species winners 再按：

```text
(split, training_class, genus_taxid)
```

最多保留：

```text
max_species_per_genus = 1
```

因此 train、validation、test 各自独立执行“每 genus 一个 species”。一个 split 的代表不会挤掉另一个 split 的代表。

### Unresolved taxonomy

无法解析 species/genus 时，为 accession 建立独立保守分组并计入报告，不能把所有 unknown 合成一个 genus，也不能静默删除。

### 输出

```text
genus_balance_selection.tsv
genus_balance_decisions.tsv
genus_balance_report.json
```

Decision：

```text
selected
duplicate_within_species
dropped_by_genus_cap
```

## 6. Fragment materialization

v2.0 已经完成 genome-level split 防泄漏和变长度等概率 fragment 生成。v2.1 不重新读取完整 genome，而是：

```text
旧 fragment corpus
→ 从 header 提取 accession
→ accession 是否存在于 genus plan
→ 保留或删除该 genome 的全部 fragments
→ 根据 header 重路由类别
```

在不加入新 genome、不移动 split、使用同一批既有 fragments 的前提下，这与“先选 genome 再保留其既有随机 fragments”条件等价。

`--ignore-target-bp` 表示不做 per-genome fragment/depth cap。被选中的 accession 保留全部已有 fragments。

## 7. Label subsystem

事实来源是 header：

```text
>fragment_id sg=<supergroup> label=<domain> epoch=<split>
```

解析优先级：

1. 精确 organelle compartment；
2. 明确 organelle label；
3. domain label；
4. generic prok + exact supergroup。

禁止 taxonomy 子串匹配：

```text
Archaeplastida 不是 plastid
Kinetoplastida 不是 plastid
```

历史文件名仅是 container hint。`organelle.fasta` 内的 `sg=Archaeplastida label=euk` 必须输出到 eukarya。

## 8. Split 与防泄漏

v2.1 继承 v2.0 的 genome-level split：

- accession 不重新分配 split；
- genus plan 只删除 accession；
- 无泄漏集合的子集不会因为删除记录而产生新泄漏；
- train/validation/test 的 genus cap 独立执行；
- `test_benchmark` 不进入训练物化默认 splits。

本轮仍应保留审计：

```text
base accession 跨 split
species_taxid 跨 split
ANI/leakage group 跨 split
```

当下一轮加入 Tiara 新 genome 时，必须重新执行 genome/species/ANI group split 分配，不能继续仅继承旧 split。

## 9. Stage contracts

### `acquire`

包装包内 `scripts/ncbi_pipeline.py`。负责 metadata、index、QC、下载、验证、export、split、organize。Tiara deferred 数据本轮不运行下载。

### Index merge（独立脚本）

`merge_training_indexes.py` 将 v2 与 Tiara S1 合并，但不产生网络副作用。

### Genus plan（独立脚本）

`plan_genus_balance.py` 产生 genome selection plan，不读写 FASTA。

### Materialize（独立脚本）

`subset_corpus_by_plan.py` 流式读取旧 fragments，在 `/ssd` 写新 source corpus。源只读，目标使用 `.part` 后原子改名。

### `chop_bin`

按 fragment 长度分 shard。当前不是从 genome 生成 fragments，应理解为 `bin_fragments_by_length`。

### `dedup`

MMseqs fragment-level 去重。当前默认关闭；不等于 genome-level split 防泄漏。

### `regroup`

把 shard/去重结果重组成训练消费的 corpus layout。

### Fragment bp balance（独立脚本）

`scripts/sample_fragments_by_bp.py` 是 corpus layer 与 training layer 之间的
数据检查点。输入是 regroup 后的最终 fragments；输出按 `version_tag` 隔离，
因此改变 class prior 或 Euk 子类权重不会触发重新 chop/dedup。

规划单位是每个 stratum 的总序列 bp。顶层五类使用 Tiara S3 prior；
Eukarya 的固定预算再通过 capped water-filling 分给 stage-2 子类。选择由
`seed + fragment_id` 决定，保证可重复。validation/test 不采样。

完成契约：

```text
sampling_plan.json
sampling_report.tsv
train/{bacteria,archaea,eukarya,mitochondria,plastids}.fasta
```

`tiara2 status` 只读取这些小文件并报告 `pending / incomplete / done`，不扫描
多 TB FASTA。训练必须读取 bp-balanced output；TF-IDF、seqpack、feature cache
均按 `version_tag` 隔离。旧版训练期 row-count subsample 必须关闭，防止在
bp-balanced corpus 上进行第二次抽样。

### `train`

沿用现有 flat 训练逻辑。读取 corpus、class rates、TF-IDF/features/seqpack，生成单层模型体系。

### `publish`

发布已经训练完成并验证的模型和元数据。

## 10. v2.1 训练范围

必须区分“数据管线支持”与“本轮模型纳入”：

| 类别 | 管线可索引/修复 | v2.1 训练 |
|---|---:|---:|
| bacteria | 是 | **否** |
| archaea | 是 | 是 |
| eukarya | 是 | 是 |
| mitochondria | 是 | 是 |
| plastids | 是 | 是 |
| virus | 可保留为测试资产 | 否 |

v2.1 的训练类别包含 bacteria、archaea、eukarya、mitochondria、plastids，不包含 virus。

## 11. Two-stage 状态

Two-stage 目前是未来方案，不是 v2.1 已实现功能。

尚未实现的内容包括：

- stage 1 与 stage 2 各自的数据生成契约；
- 两套训练调用和模型保存；
- stage 1 输出到 stage 2 输入的推理路由；
- 两阶段阈值校准；
- 级联评估与错误传播报告；
- publish 中的双模型版本管理。

`class_hierarchy` 或文档中的 stage1/stage2 名称不能作为“已经实现”的证据；只有 train/predict/publish 全链路消费它们后才算实现。

## 12. 运行顺序

```text
import Tiara index
→ merge indexes
→ per-split species/genus plan (cap=1)
→ materialize fragments + relabel
→ storage/status audit
→ chop_bin
→ optional dedup
→ regroup
→ exact census
→ fragment bp balance
→ existing flat train
→ publish
```

## 13. 未来演进

下一版本再处理：

1. bacteria 正式进入训练及类别先验重算；
2. Tiara deferred genome 下载与物化；
3. 新旧 genome 合并后的 ANI/split 防泄漏；
4. fragment-level dedup 的收益/成本评估；
5. per-genome depth budget；
6. two-stage 训练、推理、评估和发布全链路。
