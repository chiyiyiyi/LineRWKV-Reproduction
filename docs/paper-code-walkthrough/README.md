# LineRWKV「论文 ↔ 代码」对照讲解 · 目录

生成时间：2026-09-29

| 文件 | 内容 | 论文对应 |
|---|---|---|
| [00-总览.md](./00-总览.md) | 阅读路线、**符号对照表（论文 y/x/z ↔ 代码 T/C/I）**、系统数据流、论文↔代码索引、**代码与论文的 8 处差异清单**、四种规模配置 | 摘要、Sec. I、III-A、Alg. 1、Fig. 1–2 |
| [01-配置与数据.md](./01-配置与数据.md) | `config.py` 全部 45 行、`dataset.py` 全部 78 行 | Sec. IV-A |
| [02-模型-model.py.md](./02-模型-model.py.md) | `model.py` 全部 535 行：Encoder / Decoder / BandZeroDecoder / ChannelMix / TimeMix / WKV / Block / LineRWKV.forward / training_step | Sec. III-B ~ III-G，Eq. (1)–(19)，Fig. 3 |
| [03-训练-train.md](./03-训练-train.md) | `train.py`、`finetune.py`、`trainer.py`（学习率退火、检查点、两阶段训练流程） | Sec. III-G、IV-A |
| [04-推理-inference.py.md](./04-推理-inference.py.md) | `inference.py` 全部 664 行：递归 vs 并行两套 RWKV 实现、权重前缀切分、5 槽状态布局、数值稳定技巧 | Sec. III-C、III-D |
| [05-压缩与解压.md](./05-压缩与解压.md) | `compressor.py`、`decompressor.py` 逐段：预量化、逐行归一化与边信息、第一行 DPCM、主循环、数值告警、熵编码、两端口径差异 | Sec. III-A、III-E、III-F、Alg. 1 |
| [06-CUDA与脚本.md](./06-CUDA与脚本.md) | `cuda/wkv_cuda.cu` 前向/反向核逐行、`wkv_op.cpp` 绑定、bf16 版本、`launcher_*.sh` 与码率统计 | Sec. III-C 脚注、IV-A、IV-C |

## 配套材料

- `../_paper/paper_cols.txt`：从 PDF 按栏抽取的论文全文（带 `===== PAGE n =====` 标记，
  便于按页引用）。
- `../_paper/figs/`：论文 Fig. 1/2/3 所在页面渲染图。
- `../Code/`：从服务器复制到本地的仓库代码副本（与服务器内容一致）。

## 最快的上手路径

1. 读 `00-总览.md` 的第 2 节（符号表）和第 3 节（数据流图）——**这十分钟能省掉后面几小时**；
2. 读 `00-总览.md` 第 5 节（差异清单），知道哪些地方论文和代码对不上；
3. 按 `05` → `04` → `02` 的顺序读：先搞清“跑起来时发生了什么”，
   再搞清“推理时逐行递归怎么实现”，最后回到训练时的并行实现。
