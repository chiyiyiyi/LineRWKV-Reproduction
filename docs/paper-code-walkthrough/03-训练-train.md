# 03 · `train.py` / `finetune.py` / `trainer.py`

论文对应：Sec. III-G（p.7）与 Sec. IV-A（p.7）。

> “Training LineRWKV has been done over **four Nvidia A100 40-GB GPUs**. In order to limit
> training memory requirements and speed up convergence, a random subset of **16 contiguous
> bands** has been used in the initial training phases, followed by **fine-tuning with all the
> available bands and a smaller batch size**. For the M configuration, the initial batch size
> was 8 without any column sampling; in the fine-tuning phase with all bands, the batch size
> is 4 with a subsampling of 16 columns. Learning rate was linearly decreased from 10⁻⁴ to
> 10⁻⁶ over 4000 epochs.”

---

## 1. `train.py`（56 行）—— 第一阶段：16 波段预训练

```python
15  parser = argparse.ArgumentParser()
16  parser.add_argument('--log_dir', default='log_dir', help='Tensorboard log directory')
17  parser.add_argument('--save_dir', default='Results', help='Trained model directory')
18  param = parser.parse_args()
20  model_time = time.strftime("%Y%m%d_%H%M")

24  config = Config()
25  config.save_dir = os.path.join(param.save_dir, str(model_time))

30  train_dataset = HySpecNet11k(config, mode='train')
31  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False, num_workers=8, prefetch_factor=8)
32  val_dataset = HySpecNet11k(config, mode='val')
33  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=4, prefetch_factor=4)
```

- **20**：用时间戳当实验名（`log_dir/<时间戳>`、`Results/<时间戳>`），这也是仓库里
  `log_dir/linerwkv/{xs,s,m,l}` 与 `Results/linerwkv/{xs,s,m,l}` 这种命名**不是**
  脚本产物的原因——作者事后手工把几个正式实验重命名成了 xs/s/m/l。
- **30–31**：训练集走 `mode='train'` ⇒ `dataset.py:54-56` 每条样本随机 16 个连续波段。
- **33**：验证集固定 4 条样本（`dataset.py:33` 只读 val.csv 的最后 4 行）、
  batch_size=4、用**全 202 波段**。这就是“训练 16 波段、验证 202 波段”的来源，
  属于作者的省事做法（`val_l1` 数值只能当趋势看，跨配置不可直接比较）。
- 注意这里**没有任何列子采样开关**：`pos_size` 是模型内部按 `self.mode == 'train'`
  自动生效的（`model.py:378-384`）。

```python
37  log_dir = os.path.join(param.log_dir, model_time)
40  with open(os.path.join(log_dir,'config.txt'), 'w') as cfg_file:
41      cfg_file.write(json.dumps(config, default=lambda obj: obj.__dict__))
42  logger = pl.loggers.TensorBoardLogger(param.log_dir, name=model_time)
```

- **40–41**：把 `Config` 实例的 `__dict__` 直接 dump 成 JSON。
  这就是 `log_dir/linerwkv/{xs,s,m,l}/config.txt` 的内容——
  **它同时也是一个可被推理脚本加载的 config 文件**
  （`compressor.py:90-93` 会 `setattr` 回 `Config()`）。
  注意此时 `config.n_layer` 还不存在（它在 `LineRWKV.__init__` 里才被创建），
  所以 config.txt 里没有这个字段。

```python
46  save_dir = os.path.join(param.save_dir, model_time)
47  if not os.path.exists(save_dir):
48      os.makedirs(save_dir)
51  model = LineRWKV(config)
52  accelerator = "gpu" if config.device=="cuda" else "cpu"
53  trainer = pl.Trainer(accelerator=accelerator, logger=logger, callbacks=[train_callback(config)])
55  trainer.fit(model, train_loader, val_loader)
```

- **51**：模型在这里构造，`config.n_layer` 被写两次（`model.py:291/293`）。
  **但 `trainer.py` 里回调持有的是同一个 `config` 对象**，所以
  `train_callback` 看到的 `n_layer` 是 `n_layer_bands`。回调本身不用这个字段，无影响。
- **53**：没有设置 `max_epochs`！训练何时停完全由 `trainer.py:29-31` 决定：
  当 `current_epoch / epoch_count >= 1` 时保存并 `exit(0)`。
  **这实际上是把 `epoch_count` 当成了“训练总轮数”的硬编码终止条件。**
- **54**（注释掉）才是论文里“4 张 A100、多卡”的写法：
  `devices=4, max_epochs=config.N_epoch`。仓库里保留的是单卡版本，
  论文 Sec. IV-A 描述的 4×A100 是作者实际跑实验时的配置。

---

## 2. `finetune.py`（56 行）—— 第二阶段：全波段微调

与 `train.py` 的差异只有三处：

```python
18  parser.add_argument('--ckpt_file', default='', help='Checkpoint file')
32  train_loader = ...(train_dataset, batch_size=config.batch_size, shuffle=True, ..., num_workers=4, prefetch_factor=4)
53  trainer = pl.Trainer(accelerator=accelerator, devices=2, logger=logger, max_epochs=8000, callbacks=[train_callback(config)])
55  trainer.fit(model, train_loader, val_loader, ckpt_path=param.ckpt_file)
```

- **18 / 55**：用 `ckpt_path` 接着上一阶段的 Lightning checkpoint 继续训练。
  Lightning 的 `ckpt_path` 会恢复**模型权重 + 优化器状态 + epoch 计数**，
  所以这一步是真正的“续训”，而不是重新初始化。
- **32**：因为第二阶段用**全 202 波段**，显存吃紧，所以把 `num_workers` 从 8 降到 4、
  `prefetch_factor` 从 8 降到 4（`config.batch_size` 则需要在 `config.py` 里手工改小）。
  论文：“the fine-tuning phase with all bands, the batch size is 4 with a subsampling of
  16 columns” ⇒ 对应的代码改动是 `config.batch_size = 4` 和 `config.pos_size = 16`。
- **53**：`devices=2`（论文说 4 卡，这里是仓库里留下的双卡版本），
  并且**这次设置了 `max_epochs=8000`**（与 `trainer.py` 里的 `exit(0)` 逻辑并存，
  谁先触发谁生效）。
- 需要强调：**`finetune.py` 的 `--ckpt_file` 要的是 Lightning 的 `.ckpt`**，
  即 `log_dir/**/checkpoints/epoch=*.ckpt`，不是 `Results/**/rwkv-final.pth`。

---

## 3. `trainer.py`（111 行）—— 回调里的学习率调度与保存

### 3.1 学习率线性衰减（13–49）

```python
20          # LR schedule
21          w_step = args.warmup_steps
22          if args.lr_final == args.lr_init or args.epoch_count == 0:
23              lr = args.lr_init
24              if trainer.global_step < w_step:
25                  lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)
26          else:
27              progress = current_epoch / args.epoch_count
28              lr = args.lr_init - progress*(args.lr_init-args.lr_final)
29              if progress >= 1:
30                  torch.save(pl_module.state_dict(),  f"{args.save_dir}/rwkv-final.pth")
31                  exit(0)
```

- **27–28**：`lr = lr_init − (epoch/epoch_count)·(lr_init − lr_final)` —— 严格的**线性退火**，
  与论文 “Learning rate was linearly decreased from 10⁻⁴ to 10⁻⁶ over 4000 epochs” 一致。
  （`config.py` 默认 5e-4 → 1e-6；M 配置的 config.txt 里是 1e-4 → 5e-6。）
- **29–31**：跑满 `epoch_count` 就保存 `rwkv-final.pth` 并**直接 `exit(0)`**。
  这解释了为什么 `Results/linerwkv/{xs,s,m,l}/rwkv-final.pth` 是最终权重，
  而 `log_dir/**/checkpoints/*.ckpt` 是过程中的 Lightning checkpoint。
  ⚠️ `exit(0)` 会跳过 Lightning 的正常收尾（包括最后写 checkpoint），
  所以 `rwkv-final.pth` 才是权威产物。
- **22–25**：warmup 只在 `lr_final == lr_init`（即不做退火）时启用。
  当前配置 `lr_init=5e-4 ≠ lr_final=1e-6`，所以 **`warmup_steps` 实际不起作用**。

```python
33          if args.weight_decay_final > 0:
34              wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
35          else:
36              wd_now = args.weight_decay
38          for param_group in trainer.optimizers[0].param_groups:
39              if param_group["weight_decay"] > 0:
40                  param_group["weight_decay"] = wd_now
41              if args.layerwise_lr > 0:
42                  param_group["lr"] = lr * param_group["my_lr_scale"]
43              else:
45                  param_group["lr"] = lr
```

- **33–36**：权重衰减按指数从 `weight_decay` 过渡到 `weight_decay_final`；
  当前两者都是 0 ⇒ `wd_now = 0`。
- **41–42**：把 `model.py:330-332` 设的 `my_lr_scale`（1/2/3）乘上去，
  实现 RWKV 的分层学习率。
- **38**：`trainer.optimizers[0]` 假设只有一个 optimizer group 列表
  （`configure_optimizers` 返回的就是单个 AdamW），多优化器场景会不适用。

### 3.2 周期性保存（58–67）

```python
62              if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
63                  to_save_dict = pl_module.state_dict()
65                  torch.save(to_save_dict,  f"{args.save_dir}/rwkv-final.pth")
```

- 每 `epoch_save = 10` 个 epoch 覆盖写一次 `rwkv-final.pth`，
  所以这个文件既是“最终权重”也是“最近一次快照”——**训练中途被中断时，它仍然可用**。
- **61**：`if (trainer.is_global_zero)` 保证多卡时只有主进程写文件。

### 3.3 死代码（70–111）

`generate_init_weight` 是原 RWKV 仓库的“分阶段预训练权重合并”工具，
依赖 `model.args.my_pile_stage`（本项目的 `LineRWKV` 没有 `args` 属性）。
`train.py / finetune.py` 都不会调用它，可以完全跳过。

---

## 4. 从代码反推的完整训练流程

```
① 准备数据：HySpecNet-11k hard split（202 波段、128×128 的 GeoTIFF patch）
        └ 归一化： (x − 0) / 1000      ← 论文说应裁剪到 [0,10000] 并按 10000 归一
② 阶段一：python3 train.py --log_dir ... --save_dir ...
        ├ 每条样本随机取 16 个连续波段（dataset.py:54-56）
        ├ 模型内部按 pos_size 随机抽样列（model.py:378-384）
        ├ L1 损失（model.py:450）
        └ 每 10 epoch 存 Results/<ts>/rwkv-final.pth；Lightning 同时存 checkpoints/
③ 阶段二：python3 finetune.py --ckpt_file log_dir/**/checkpoints/xxx.ckpt
        ├ 全 202 波段、batch 调小、num_workers 调小
        └ 继续线性退火到 lr_final
④ 训练终止：current_epoch / epoch_count ≥ 1 时保存并 exit(0)
⑤ 产物：Results/linerwkv/{xs,s,m,l}/rwkv-final.pth（state_dict）
        log_dir/linerwkv/{xs,s,m,l}/config.txt（推理时要传给压缩/解压脚本的配置）
```

> ⚠️ **`rwkv-final.pth` 是 `pl_module.state_dict()`**，即一个纯 `OrderedDict`
> （`trainer.py:63`）。这正是 `inference.py:38-43` 能直接用
> `key.startswith('encoder.')` 这种前缀筛选来切分权重的原因。

---

## 5. 与论文 Sec. III-G 的逐条核对

| 论文（Sec. III-G） | 代码 | 是否一致 |
|---|---|---|
| 线预测器训练时用并行实现替代递归实现 | `LineRWKV.forward` 直接用 `Block`（内部 CUDA kernel） | ✅ |
| 编码器之后的操作对列独立，可子采样列 | `pos_size` + 1×1 卷积解码器 | ✅ |
| 最小化 ℓ1 损失 | `F.l1_loss(logits, y)` | ✅ |
| 训练不建模量化误差 | 训练阶段完全是 Q=1 的无损目标 | ✅ |
| 初期 16 连续波段 → 后全波段微调 | `dataset.py` + `finetune.py` | ✅（需手工接力） |
| 4×A100、batch 8 → 4 | `train.py:54` 被注释成单卡；`finetune.py:53` 为 `devices=2` | ⚠️ 仓库版本与论文不同 |
