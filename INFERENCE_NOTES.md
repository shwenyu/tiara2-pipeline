# 推理层检查与改造记录（v2.2.0）

对照你上传的 tiara 原始推理层六个文件（`prediction.py` / `classification.py` /
`models.py` / `transformations.py` / `utilities.py` / `bow.py`）逐行审查后的结论。

---

## 一、必须改（不改就跑不起来 / 结果错）

### 1. `bow.py` 的 `MAX_K = 7` 是 v2.2.0 的硬阻断

```python
MAX_K = 7
...
if k not in kmer_to_pos:
    raise ValueError("k-mer length not supported!")
```

v2.2.0 的 `k_second = [5, 6, 7, 8]` 含 k=8。只要二阶段选中 k=8 的模型，
推理第一次调用就会抛 `k-mer length not supported!`。这是训练侧改 k 之后
**唯一一个会让整条推理链直接崩掉**的地方。

已改为 `MAX_K = 8`，并把 k→kmer 索引表改成**按需惰性构建 + 缓存**。
原实现在 import 时就把 k=1..MAX_K 全部建出来；升到 8 之后那是约 87,000 个
numba typed dict 字符串键，且**每个 joblib worker 进程都要重建一遍**。
现在只跑 k=6 的任务不会再为 k=8 付这笔开销。

### 2. `models.py` 无法承载 `hidden_2 = none` 的已发布模型

原 `NNet1/NNet2` 构造函数固定两层隐藏层。但 HP 搜索是可以选出单隐藏层的，
而且**已经选出来了**——v2.1.3 实际发布了
`second_k-6_hidden_1-128_hidden_2-none_lr-0.01_dropout-0.5_epochs-50.pkl`。
这个 `.pkl` 用原始 `models.py` 根本构造不出对应网络，`load_params` 之前就会挂。

已抽出 `_layers()`，`hidden_2` 为 `None` / `"none"` / 空时自动退化为单隐藏层。

### 3. `transformations.TfidfWeighter.transform` 两个 bug

```python
result = result / np.linalg.norm(result, axis=1).reshape((-1, 1))
result = result.reshape(len(data), -1)
```

- **零范数除法 → NaN**：全 N、或长度不足 k 的片段，bow 行全零，除以自己的零范数
  得到 NaN。NaN 灌进网络后输出非有限值，最终落到 `unknown` 分支。这正是
  v2.1.2 benchmark 里那句「另外三条由于非有限特征输出 `unknown`」的来源。
  现在改为 `np.divide(..., where=norms > 0)`，零行保持精确零。
- **`len(data)` 用错**：`data` 是单条字符串时，`len(data)` 是**序列碱基数**而不是 1，
  reshape 直接报错。改为 `len(seqs)`。

### 4. stage × k 三元配对必须强校验

`Classification.__init__` 里 `zip(models, nnet_weights, self.params)` 只是
按位置配对，谁也没检查「这个 `.pkl` 的 k」和「这个 tf-idf 目录的 k」是不是同一个。
在 k 列表固定为 4/5/6 的年代问题不大，k 可配置之后就是隐患:
`first_k-7_*.pkl` 配 `k6-first-stage/model.npy` 只有在维度恰好不同才会报错。

新增 `_check_stage_consistency()`，加载前逐阶段校验
`params.k == tfidf.k == len(idfs) 对应的 4**k`、`fragment_len` 一致、
权重文件名里的 k 与 params 一致，任何一条不满足直接拒绝加载并列出全部问题。

---

## 二、不需要改的（确认过）

- `hidden_1 / hidden_2 / dropout / lr / epochs / 权重本身`：由训练侧决定，推理只读。
- **网络输入维度**：`params_filtered.update({"dim_in": 4 ** params_dict["k"]})`
  已经跟着 k 自动走，改 k 不需要动 `models.py` 的维度。
- **输出宽度**：一阶段 5、二阶段 3，由 `prediction.id_to_class` 的标签映射决定，
  与 k 无关。（一阶段索引 2、二阶段索引 1 是训练标签里没有的空位，本来就恒接近 0。）
- `utilities.chop / parse_params / SingleResult.generate_line`：逻辑正确，未改语义。
  仅在 `Classification` 里补了 `parse_params` 结果的类型转换——它返回的是字符串，
  一旦真的用 `.csv` 传参，`4 ** "7"` 会直接 TypeError。

---

## 三、效率优化

原实现的四个瓶颈，按影响从大到小：

| # | 原实现 | 现在 |
|---|--------|------|
| 1 | `list(SimpleFastaParser(handle))` 一次性把**整个 FASTA 读进内存** | 流式读取 + 定量分批（`batch_records`，默认 512） |
| 2 | `predict_proba` **每条 record 调一次**，每次只算几个 5kbp 片段 | 整批片段拼成一个矩阵，**一次 predict_proba**，再用 `np.add.reduceat` 按 record 切回去求均值 |
| 3 | `Parallel(n_jobs=...)` **每阶段每文件重建**，tf-idf 模型反复 pickle 给 worker | executor 只建一次，`max_nbytes=None` 避免把 idf 向量 memmap 到 /tmp |
| 4 | 网络处于 train 模式、autograd 开启；k-mer 计数用**逐位置 Python 字符串切片 + dict 查找** | `eval()` + `torch.inference_mode()`；k-mer 计数改为 2-bit 滚动编码（与训练侧同一套算法） |

第 2 条对 GPU 的意义最大：原来每条 contig 都要付一次 host→device 拷贝和一次 kernel
launch，实际算的却只有几行；批处理后同一次 launch 能算上千行。

第 4 条的 2-bit 滚动编码顺带修正了一个语义细节：非 ACGT 字符（N 等）现在会
**打断滚动窗口**，等价于原来「dict 里查不到就跳过」的行为，但不再为每个位置
做一次 O(k) 的字符串哈希。

**行为保持不变**：切片方式、tf-idf、片段概率取均值、阈值规则、prokarya 兜底、
两阶段串联顺序、输出列顺序都与原版一致。

---

## 四、统一调用层

新增 `tiara2/inference.py` + `python -m tiara2.cli classify`。

以前要跑推理，得有人自己知道哪个 `first_k-*.pkl` 配哪个 `k*-first-stage` 目录、
手抄一遍网络结构进 params dict、再记住 `prob_cutoff`。这正是让塌缩的 v2.1.2
参数包一路走到 benchmark 的那类失误，而 k 列表可配置之后只会更容易出错。

所以模型集合是**发现出来的，不是手写的**：

- `training_manifest.json` → 有哪些 stage×k、各自的 hidden/dropout/dim_out/mean_f1
- `tfidf_manifest.json` / `k<k>-<stage>-stage/` 目录 → 有哪些 tf-idf 模型
- `config` 的 `publish.model_tag` → 用哪个已发布参数包

两边**取交集**后每阶段解析出一个严格三元组，再交给 `Classification`（它会再校验一次）。
代码里没有任何 `[4,5,6]` / `[4,5,6,7]` 硬编码，v2.1.3 的老包和 v2.2.0 的新包都能直接读。

### 用法

```bash
# 看一眼会用哪套模型（不跑推理）
python -m tiara2.cli classify --emit-manifest infer_v2.2.0.json

# 正常推理
python -m tiara2.cli classify -i contigs.fa -o out.tsv --probabilities --threads 24

# 锁定某个 k（默认按 manifest 里记录的 mean_f1 最优挑，没记录则取最大 k）
python -m tiara2.cli classify -i contigs.fa --k-first 6 --k-second 8

# 用固定清单复现一次 benchmark，逐字节可重复
python -m tiara2.cli classify --manifest infer_v2.2.0.json -i contigs.fa -o out.tsv

# 直接指定目录，绕过 config
python -m tiara2.cli classify --nnet-dir ... --tfidf-dir ... -i contigs.fa
```

### 关于 ensemble

`Classification` 每阶段**只加载一个网络**。已发布的 7 个 `.pkl` 从来不是
ensemble，也从未被平均过——原实现就是 `zip(models, nnet_weights, params)`，
多出来的权重文件会被静默丢弃。统一调用层把这件事**显式化**了：默认按
`mean_f1` 选最优，选了哪个会打印出来并写进清单，而不是靠调用顺序碰运气。
如果以后真要做多 k ensemble，那是训练侧和 `Prediction` 的新功能，不是配置项。

---

## 五、必须记住的一件事：`prob_cutoff` 要重新标定

`tiara2/inference.py` 里的 `DEFAULT_CUTOFF` 是 v1.4 时代的旧值，
**只够用来 smoke test**。v2.2.0 换了 k、换了特征维度、换了模型，
阈值必须在新 benchmark 上按 `模型 × 任务 × panel × contig 长度` 重新扫。
v2.1.3 的最优阈值落在 0.938553–0.999999，和默认的 0.65 完全不是一个量级——
直接沿用旧阈值会让结果看起来「没提升」，但那是阈值的锅，不是模型的锅。

标定完把值写进推理清单 JSON，之后所有 benchmark 都用 `--manifest` 跑。

---

## 六、测试

新增 `tests/test_inference_plan.py`（8 项）。`tiara2/inference.py` 刻意把
torch/skorch/numba/Bio 的 import 关在 `make_classifier()` 里面，所以
**模型解析这部分——也就是历史上真正出过错的部分——可以脱离深度学习依赖单独测**。

覆盖：v2.2.0 的 5/6/7 + 5/6/7/8 布局、v2.1.3 老布局向后兼容、
mean_f1 优先于最大 k、锁定不存在的 k 必须报错、有网络但缺 tf-idf 的 k 不得被选中、
缺二阶段必须致命、清单往返逐字节稳定、`hidden_2-none` 正确解析。

全量：`passed=40 failed=0 skipped=2`。

---

## 七、infer 阶段（接在 publish 之后）

上面的改造把推理变成了一个可靠的**命令**，这一节把它变成一个**流水线阶段**。

### 为什么是 stage 而不是脚本

`publish` 把参数包拷进 package 之后，流水线就结束了。谁要预测，就得离开流水线、
手工拼模型集合、自己记住阈值。**v2.1.2 的塌缩参数包就是从这个交接口溢出去直接上了
benchmark 的**。把推理纳入 stage 契约之后，它自动获得：

| 能力 | 说明 |
|---|---|
| 同一套调用 | `tiara2 run --only infer` / `--from publish`，与前面七个阶段一致 |
| 同一套配置 | 单一 `infer:` 块，沿用相同的 `{results_root}` / `{output_tag}` 模板 |
| resume | 指纹**同时覆盖输入 fasta 和已发布的参数包**——重新 publish 会自动作废旧预测 |
| dry-run | 只解析并打印模型集合，**不加载 torch** |
| manifest | 记录本次用了什么模型、跑了什么输入、产出什么 |

resume 那一条是刻意的：如果指纹只看输入 fasta，那么重新训练、重新 publish 之后再跑
推理，会因为「输入没变」直接 skip，然后你拿着上一版的预测去算新版的 F1。

### 默认关闭

`infer.enabled: false`。一次训练跑完不应该因为 publish 成功就自动开始分类几百 GiB
数据。你要的流程就是 `train → publish → 确认无误 → 推理`，所以确认这一步是一个
显式动作：

```bash
python -m tiara2.cli run --from train --to publish       # 停在 publish
# …检查 nnet-models-v2.2.0 / tfidf-models-v2.2.0…
python -m tiara2.cli run --only infer --set infer.enabled=true
```

### 输入接口

`infer.inputs` 接受**文件 / 目录 / glob 任意混写**，自动去重，目录里只抓
`.fa/.fna/.fasta`（含 `.gz`）。加一个 benchmark panel 是改配置，不是改代码。

```yaml
infer:
  inputs:
    - "{base}/benchmark/fungi_main.fasta"   # 单文件
    - "{base}/benchmark/panels"             # 整个目录
    - "{base}/benchmark/*.fna"              # glob
```

### 输出接口

每个输入产出一对，整批再产出两个汇总文件：

```
<out_dir>/
  fungi_main.tsv                      # sequence_id + 两阶段结果 [+ 8 列概率]
  log_fungi_main.txt                  # 模型参数 + 分类统计
  fungi_main_eukarya.fasta            # 仅当 to_fasta 指定时
  inference_manifest_v2.2.0.json      # 本次用的模型集合
  inference_summary.json              # 所有输入的计数汇总
```

TSV 列顺序、`--to-fasta` 的类名缩写（`mit pla bac arc euk unk pro org all`）、
`log_<output>` 这个命名，全部沿用原始 tiara 的约定，下游脚本不用改。

`log_*.txt` 里写清楚了每个阶段的 k、阈值、权重文件名、tf-idf 目录和训练时记录的
`mean_f1`。一份没有这些信息的结果文件，正是当初塌缩参数包能悠然跑完一整轮
benchmark 而无人察觉的原因。

### 配置错误在加载时就报

`tiara2/config.py::validate()` 新增了 `infer` 块校验。不要等跑完 8000 秒训练才发现
推理配置写错：

- `enabled: true` 但 `inputs` 为空
- `k_first`/`k_second` 要一个 **`train.k_first`/`k_second` 里没训练过的 k**
  （比如 v2.2.0 已经不训 k=4 了，再写 `k_first: 4` 直接报错）
- 阈值不在 (0, 1)
- `to_fasta` 写了不存在的类名
- `device` 不是 `cpu`/`cuda`/null

### 测试

新增 `tests/test_infer_stage.py`（9 项）。`tiara2/stages/infer.py` 把 `tiara2.inference`
的 import 放在方法内部，所以「跑什么」的决策逻辑——输入解析、输出命名、resume
指纹、配置校验——可以完全脱离 torch 测试。

全量：`passed=49 failed=0 skipped=2`。
