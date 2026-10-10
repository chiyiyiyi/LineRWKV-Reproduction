# 04 · `inference.py` 逐行讲解（推理侧的两套 RWKV 实现）

`inference.py` 664 行，是本仓库**推理时真正跑的代码**（`compressor.py` 和
`decompressor.py` 都只 import 这个文件，不 import `model.py`）。

它包含两套完全不同的 RWKV 实现：

| 行号 | 类 | 机制 | 谁在用 |
|---|---|---|---|
| 121–289 | `LineRWKV_RNN_spatial` | **手写逐行递归**（无 CUDA） | 压缩端 + 解压端 |
| 293–456 | `LineRWKV_RNN_spectral` | **手写逐波段递归**（无 CUDA） | 仅解压端 |
| 459–587 | `RWKV_ChannelMix/TimeMix/Block/WKV` 的副本 | 与 `model.py` 相同，走 CUDA kernel | 供下面这个类使用 |
| 638–663 | `LineRWKV_RNN_spectral_parallel` | **并行**（一次算完所有波段） | 仅压缩端 |

论文对应：Sec. III-C（递归实现）+ Sec. III-D（并行实现）这两条路线的分工。

> 论文 Sec. III-D：
> “However, for a lossless (or prequantized) compressor, the **parallel implementation
> typically reserved for training** is used rather than the **recurrent implementation**
> described in Section III-C in order to ensure high throughput.”

---

## 1. 顶部（1–24 行）

```python
14  MyModule = nn.Module
15  def __nop(ob):
16      return ob
17  MyFunction = __nop
19  # JIT compilation for speedup, can be buggy
20  #if os.environ["RWKV_JIT_ON"] == "1":
21  #MyModule = torch.jit.ScriptModule
22  #MyFunction = torch.jit.script_method
24  DEBUG_TIME=False
```

- **14–17**：把 `MyModule/MyFunction` 定义成恒等装饰器，是原 RWKV 仓库为了
  “可切换 torchscript” 留的钩子。**JIT 被注释掉了（19–22 行）**，
  所以下面 `@MyFunction` 完全是空操作，函数就是普通 Python 方法。

---

## 2. 带自动加载权重的三个小模块（26–118 行）

`LineEncoder`（26–59）、`LineDecoder`（62–93）、`BandZeroDecoder`（96–118）的
**网络结构部分与 `model.py` 中同名类完全一致**（逐行对照见 `02-模型-model.py.md`），
唯一新增的是构造函数里的一段：

```python
37          with torch.no_grad():
38              w = torch.load(config.model_file, map_location='cpu')
39              w_encoder = {}
40              for key in w.keys():
41                  if key.startswith('encoder.'):
42                      w_encoder[key[len('encoder.'):]] = w[key]
43              self.load_state_dict(w_encoder)
44              self.to(config.device)
```

- **38**：`config.model_file` 就是 `Results/linerwkv/<cfg>/rwkv-final.pth`
  （由 `--model_file` 传入）。
  因为 `trainer.py:63` 保存的是完整 `state_dict()`，所以这里拿到的是
  `{'encoder.conv_in.weight': ..., 'decoder.convs.0.weight': ..., 'blocks_lines.0...': ...}`
  这样一张扁平的字典。
- **41–43**：**按前缀切分权重**。`LineEncoder` 只取 `encoder.` 开头的键、
  去掉前缀后 `load_state_dict`；`LineDecoder` 取 `decoder.`（74–77 行）；
  `BandZeroDecoder` 取 `bandzerodec.`（105–109 行）。
  这是一种“把一个 .pth 拆成若干独立小模型”的省事做法，
  代价是**同一个 .pth 会被 `torch.load` 4–5 次**（每个类一次）。
- **44**：加载后再 `.to(device)`。注意 `torch.load` 用的是 `map_location='cpu'`，
  所以即使目标是 CUDA 也不会在 CPU 上受限。

> 这也是为什么推理脚本里这 3 个模块**不需要再手动 `load_state_dict`**，
> 而 `LineRWKV_RNN_*` 需要自己处理（见下节）。

---

## 3. `LineRWKV_RNN_spatial`（121–289）—— 沿行方向的递归（论文 Sec. III-C）

论文 Sec. III-C 原文（p.5）：

> “Our line predictor is therefore based on the **RWKV neural network [23]** for sequence
> processing, and its **recurrent implementation is used for inference**.”

> “It can be noticed that the recursion requires a limited amount of memory. In particular,
> besides the features of the current line, the only feature vectors that need to be stored
> are the **states a_{y−2} and b_{y−2}** and **features of the previous line u_{y−2}**.
> This needs to be done for each layer of line and channel mixing that is used. However,
> this operation allows **keeping a memory of past lines** and using the attention mechanism
> for the prediction of the next line and is significantly advantageous in terms of memory
> consumption compared to keeping the features of a large number of lines.”

### 3.1 权重加载与“拍平成命名空间”（129–193）

```python
129         with torch.no_grad():
130             w = torch.load(config.model_file, map_location='cpu')
134             if 'pos_emb_x' in keys:
135                 w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']).reshape(config.ctx_len+1, -1)[:-1,:]
140             for x in keys:
142                 if '.time_' in x:
143                     w[x] = w[x].squeeze()
146                 if '.time_decay' in x:
147                     w[x] = w[x].float()
148                     w[x] = -torch.exp(w[x])
149                 elif '.time_first' in x:
150                     w[x] = w[x].float()
151                 else:
152                     w[x] = w[x].float()
154                 w[x].requires_grad = False
156                 if config.device == 'cuda' and x != 'emb.weight':
157                     w[x] = w[x].cuda()
```

- **134–135**：RWKV 原版（带位置编码的 NLP 模型）遗留的兼容分支，本项目权重里没有
  `pos_emb_x`，不会触发。
- **142–143**：`time_mix_k` 等参数在训练时形状是 `(1,1,F)`，
  这里 `squeeze()` 成 `(F,)`，方便后面直接逐元素相乘。
- **146–148**：**关键**——把训练时存的 `time_decay` 参数（= log α）
  预计算成 `−exp(time_decay)` = 论文 Eq. (4) 里的 `−α`。
  这解释了 `00-总览.md` 里 α 的换算关系，也意味着
  `compressor.py` 和 `decompressor.py` 加载的是同一份权重、做同样的换算。
- **156–157**：把所有权重搬到 GPU（`emb.weight` 是本项目不存在的字段，纯属遗留判断）。

```python
171         self.w = types.SimpleNamespace()
172         for x in keys:
173             xx = x.split('.')
174             here = self.w
175             for i in range(len(xx)):
176                 if xx[i].isdigit():
177                     ii = int(xx[i])
178                     if ii not in here:
179                         here[ii] = types.SimpleNamespace()
180                     here = here[ii]
181                 else:
182                     if i == len(xx) - 1:
183                         setattr(here, xx[i], w[x])
184                     elif not hasattr(here, xx[i]):
185                         if xx[i+1].isdigit():
186                             setattr(here, xx[i], {})
187                         else:
188                             setattr(here, xx[i], types.SimpleNamespace())
189                     here = getattr(here, xx[i])
```

- **171–189**：把扁平字典 `{'blocks_lines.0.att.key.weight': tensor}`
  **手工还原成树状结构**，于是可以写 `w.blocks_lines[0].att.key.weight`。

  为什么要这么绕？因为 `LineRWKV_RNN_*` **不是 `nn.Module` 的常规子模块树**
  （它没有在 `__init__` 里定义 `nn.Linear`，而是手工做矩阵乘法），
  所以不能用 `load_state_dict`，只能自己拼。
  这也带来一个好处：**没有 autograd 图、没有 `nn.Module` 的 hook 开销**，
  纯 FP32 手写矩阵乘法，速度反而可能更快。
- **191–193**：`self.eval()` + 清缓存。

### 3.2 `LN` 与状态布局（197–200）

```python
197     def LN(self, x, w):
198         return F.layer_norm(x, (self.config.dim_enc,), weight=w.weight, bias=w.bias)

200     # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp
```

- **198**：手工 LayerNorm，参数从 `SimpleNamespace` 里取。
- **200**：**整个递归实现的状态编码表**，每层占 5 行：

| `state[5i+k]` | 含义 | 论文符号 |
|---|---|---|
| `5i+0` | 通道混合用到的「上一时刻输入」 | `u_{y−2}`（式 8/9 里的 xx） |
| `5i+1` | 注意力用到的「上一时刻输入」 | 同上，另一个副本 |
| `5i+2` | WKV 的分子累加 | `a_{y−2}`（式 4） |
| `5i+3` | WKV 的分母累加 | `b_{y−2}`（式 5） |
| `5i+4` | 数值稳定用的指数尺度 | `p`（式 6 的归一化指数） |

  注意「注意力的上一时刻输入」和「通道混合的上一时刻输入」**各存一份**
  （`5i+0` 与 `5i+1`）：因为两者接收的输入不同
  （通道混合拿到的是 BLOCK 的输出，注意力拿到的是 LN 之后的输入），
  存两份才能在一次前向里同时更新。

### 3.3 `FF`（202–212）—— 通道混合 = 论文 Eq. (8)(9)(10)

```python
203     def FF(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
204         xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
205         xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
206         state[5*i+0] = x
207
208         r = torch.sigmoid(xr @ rw.T)
209         k = torch.square(torch.relu(xk @ kw.T))
210         kv = k @ vw.T
211
212         return r * kv
```

- **204–205**：`x` 是当前时刻（当前行/当前波段）的输入，`state[5i+0]` 是上一时刻的输入，
  于是这就是式 (8)(9) 的时间混合。**`state[5*i+0] = x` 把当前输入留作下一时刻的“上一时刻”**
  ——这就是论文说的 “features of the previous line u_{y−2}”。
- **208–212**：完全等价于 `model.py:108-114`（只是把 `nn.Linear` 展开成矩阵乘法）。
- 初始 `state = 0` ⇒ 式 (8)(9) 在第一步退化成 `xk = x`，
  与 Alg. 1 的 `Init u_{−1,x,z} = 0` 一致。

### 3.4 `SA`（214–245）—— 线性注意力 = 论文 Eq. (1)–(7)

```python
216         xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
217         xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
218         xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
219         state[5*i+1] = x
221         r = torch.sigmoid(xr @ rw.T)
222         k = xk @ kw.T
223         v = xv @ vw.T
225         kk = k
226         vv = v
227         aa = state[5*i+2]
228         bb = state[5*i+3]
229         pp = state[5*i+4]
230         ww = time_first + kk
231         p = torch.maximum(pp, ww)
232         e1 = torch.exp(pp - p)
233         e2 = torch.exp(ww - p)
234         a = e1 * aa + e2 * vv
235         b = e1 * bb + e2
236         ww = pp + time_decay
237         p = torch.maximum(ww, kk)
238         e1 = torch.exp(ww - p)
239         e2 = torch.exp(kk - p)
240         state[5*i+2] = e1 * aa + e2 * vv
241         state[5*i+3] = e1 * bb + e2
242         state[5*i+4] = p
243         wkv = a / b
245         return (r * wkv) @ ow.T
```

这段代码是整个仓库最“数学密集”的 30 行，逐段拆：

- **216–219**：式 (1)(2)(3) 的时间混合，并把当前输入存进 `state[5i+1]`。
- **221–223**：`r, k, v` 投影。
- **227–229**：取出上一时刻的状态 `(aa, bb, pp)`。注意 `pp` **不是**论文里的 p，
  而是“为了让 exp 不溢出而人为乘上的缩放指数”（RWKV 的 `max` 技巧）。
- **230–235**：计算**式 (6) 的分子 a 和分母 b**：

  ```
  a = e^{?}·aa + e^{β+k}·v
  b = e^{?}·bb + e^{β+k}
  ```

  其中 `ww = β + k = time_first + kk`，`p = max(pp, ww)`，
  `e2 = e^{ww − p}`。
  真正的 `e^{ww}` 被拆成 `e^{p} · e^{ww−p}`，而 `aa, bb` 本身已经带了 `e^{pp}` 的因子
  ⇒ `e1·aa = e^{p−pp}·aa = e^{p}·(aa·e^{−pp})`，两边同乘 `e^{p}` 后可以整体约掉，
  所以最终 `a/b` 结果不变、但**任何一步都不会上溢**。
  **这就是论文脚注所指的 “numerically-stable implementation”**，CUDA 核里
  完全同一套写法（`wkv_cuda.cu:28-32`）。
- **236–242**：**式 (4)(5) 的状态更新**：
  `ww = pp + time_decay` 里 `time_decay = −α` ⇒ 老状态乘 `e^{−α}`（衰减），
  当前 token 乘 `e^{k}`。同样用 `p = max(...)` 做归一化。
- **243**：`wkv = a / b` 就是式 (6) 的 `p_{y−1}`。
- **245**：`(r ⊙ wkv) @ W_o^T` 就是式 (7) 的 `o = W_o(σ(r) ⊙ p)`。

> **与 CUDA 实现的关系**：`cuda/wkv_cuda.cu` 的 `kernel_forward` 做的事和
> 214–245 行**逐行等价**，区别只是：
> ① CUDA 版把 `B·C` 条序列铺成一个 thread 处理一条，天然并行；
> ② CUDA 版一次性跑完 T 步、内部只保留标量 `aa/bb/pp`，因此
> **不需要保存中间状态**；而手写版每步都读写 `state`，适合一次只跑一步的推理。

### 3.5 `forward`（248–289）—— 一次处理一整行

```python
248     def forward(self, a):
249         # a: (B,F,C,I) encoding of reconstructed line l-1 with all bands to be used for line-based prediction
254         # output: (BC,I,F) spatial prediction of line l for all bands
255         with torch.no_grad():
259             B, FF, C, I = a.size()
260             a = torch.reshape(torch.permute(a,[0,2,3,1]),[B*C*I,config.dim_enc]) # (B,F,C,I) --> (BCI,F)
266             x_spatial = a+0.0
268             if self.state_lines == None:
269                 self.state_lines = torch.zeros(config.n_layer_lines * 5, B*C*I, config.dim_enc, device=self.RUN_DEVICE)
270                 for i in range(config.n_layer_lines):
271                     self.state_lines[5*i+4] -= 1e30
273             for i in range(config.n_layer_lines):
274                 if i == 0:
275                     x_spatial = self.LN(x_spatial, w.blocks_lines[i].ln0)
277                 ww = w.blocks_lines[i].att
278                 x_spatial = x_spatial + self.SA(self.LN(x_spatial, w.blocks_lines[i].ln1), self.state_lines, i, ...)
282                 ww = w.blocks_lines[i].ffn
283                 x_spatial = x_spatial + self.FF(self.LN(x_spatial, w.blocks_lines[i].ln2), self.state_lines, i, ...)
287             x_spatial = torch.reshape(x_spatial,[B*C,I,config.dim_enc]) # (BCI,F) --> (BC,I,F)
289             return x_spatial
```

- **249–254 注释**：**把接口说得很清楚**——输入是「**已重建**的上一行（全部波段）」的编码特征，
  输出是该行所有列、所有波段的特征域预测 û。
  “reconstructed”这个词是解压端能用递归实现的前提。
- **260**：`(B,F,C,I) → (B·C·I, F)`：把**列和波段全部折叠进 batch**，
  于是 RWKV 的“序列”只剩一行（长度 1）。“一次处理一整行”就是这么来的，
  这也解释了论文 Sec. IV-C 的结论——**显存与列数、波段数成正比，与行数无关**。
- **268–271**：**首次调用时初始化状态**，之后永久保留（`self.state_lines` 不随调用清零），
  这正是论文说的“continuous pushbroom operation”（推扫成像连续运行）。
  `state[5i+4] -= 1e30` 把 `pp`（指数尺度）初始化成极小值，
  等价于 CUDA 里的 `pp = MIN_VALUE = -1e38`（`wkv_cuda.cu:22`），
  保证第一次 `max(pp, ww)` 一定取 `ww`。
- **273–285**：`n_layer_lines` 层，每层 `LN0(仅第0层) → SA + 残差 → FF + 残差`，
  与 `model.py` 的 `Block.forward` 一字不差地对应。
- **287**：折回 `(B·C, I, F)`——**I 是波段维，也是“序列”维的下一个角色**，
  直接喂给谱预测器。

> **显存估算（论文 Sec. IV-C 的 673 MB 从哪来）**：
> `state_lines` 的形状是 `(5·n_layer_lines, B·C·I, F)`。
> 以 XS（`F=32, n_layer_lines=2, C=128, I=202`）为例：
> `5×2×128×202×32 = 8,273,920` 个 float32 ≈ **33 MB**。
> 加上 `a`、`x_spatial`、各层中间激活和 PyTorch 自身开销，
> 几百 MB 量级与论文报告的 673 MB 相符，且**与行数无关**。

---

## 4. `LineRWKV_RNN_spectral`（293–456）—— 沿波段方向的递归

这个类与 `LineRWKV_RNN_spatial` **几乎逐行相同**（权重加载、`LN`、`FF`、`SA` 完全一致），
只有三处差别：

1. **状态变量名和形状不同**：
   ```python
   299         self.state_bands = None
   436                 self.state_bands = torch.zeros(config.n_layer_bands * 5, BC, config.dim_enc, device=self.RUN_DEVICE)
   ```
   空间版的状态大小是 `B·C·I`（一行内所有像素），这里是 `B·C`（一个波段的所有像素）——
   **小 I 倍**，因为谱预测器一次只处理一个波段。
2. **多了一个 `reset_state`**：
   ```python
   418     def reset_state(self):
   419         self.state_bands = None
   ```
   每处理新的一行都要重置波段递归（`decompressor.py:165` 每行开头调用一次）。
   **空间状态 `state_lines` 绝对不能重置**——它承载跨行的记忆。
3. **`forward` 的接口和第一步计算不同**：
   ```python
   421     def forward(self, a, x_spatial):
   422         # a: (BC,F) encoding of reconstructed line l-1 for current band
   423         # x_spatial: (BC,F) spatial prediction for current band
   426         # output: (BC,1,F) decodable prediction of line l for current band
   433             delta = a - x_spatial # (BC,F)
   440             for i in range(config.n_layer_bands):
   441                 if i == 0:
   442                     delta = self.LN(delta, w.blocks_bands[i].ln0)
   445                 delta = delta + self.SA(self.LN(delta, w.blocks_bands[i].ln1), self.state_bands, i, ...)
   450                 delta = delta + self.FF(self.LN(delta, w.blocks_bands[i].ln2), self.state_bands, i, ...)
   454             delta = torch.reshape(delta,[BC,1,config.dim_enc]) # (BC,F) --> (BC,1,F)
   456             return delta
   ```

- **433**：`delta = a − x_spatial`，正是论文 Eq. (11) 的 `Π_z = u_z − û_z`，
  逐波段计算。**这是解压端唯一能用的形式**，因为 `a`（当前波段特征）依赖
  刚刚解码出来的那个像素。
- **454**：返回形状 `(BC,1,F)`，可以直接喂给 `LineDecoder`。

> **为什么压缩端不用它？** 因为逐波段调用 `forward` 意味着 I 次 Python 级循环、
> I 次 kernel launch，吞吐极低。压缩端可以一次性拿到所有波段，
> 所以改用下节的并行版本。这就是论文 Sec. III-D 那句
> “the parallel implementation … is used rather than the recurrent implementation
> in order to ensure high throughput” 的直接原因。

---

## 5. 并行谱预测器（459–663）

459–587 行是 `RWKV_ChannelMix / RWKV_TimeMix / WKV / RUN_CUDA` 的**副本**
（与 `model.py:88-213` 逐行相同，只有 `inference.py:546-547` 把
`T_MAX = 512` 和 JIT 编译又写了一遍）。它们存在的原因只有一个：
被下面这个类使用。

```python
638 class LineRWKV_RNN_spectral_parallel(MyModule):
639     def __init__(self, config):
642         self.config = config
644         config.n_layer = config.n_layer_bands
645         self.blocks_bands = nn.ModuleList([Block(config, i) for i in range(config.n_layer_bands)])
647         with torch.no_grad():
648             w = torch.load(config.model_file, map_location='cpu')
649             w_blocks = {}
650             for key in w.keys():
651                 if key.startswith('blocks_bands.'):
652                     w_blocks[key] = w[key]
653             self.load_state_dict(w_blocks)
654             self.to(config.device)
656     def forward(self, a, x_spatial):
658         delta = a - x_spatial # (C,I,F)
660         for block in self.blocks_bands:
661             delta = block(delta)
663         return delta
```

- **644–645**：**必须是标准 `nn.Module`**，因为要用 `Block`（内部走 CUDA kernel）。
  `config.n_layer` 又被改写一次以匹配 `n_layer_bands`。
- **650–653**：只取 `blocks_bands.` 前缀的权重，直接用 `load_state_dict`
  （因为这里确实是标准的 `nn.Module`，键名与 `model.py` 中 `LineRWKV.blocks_bands`
  完全一致）。
- **658**：`delta = a − x_spatial`，形状 `(C,I,F)`（压缩端场景下 B=1），
  与 `model.py:401` 训练时的语义**完全一致**，只是这里直接对“波段序列”做 RWKV。
- **660–661**：`n_layer_bands` 个 Block，走 `torch.utils.cpp_extension` 编译的
  `wkv_cuda` 前向核。
- **663**：返回**所有波段**的 Π̂，压缩端随后用 `delta[:,:-1,:]` 做波段左移
  （`compressor.py:190`），实现因果。

> 这是全文唯一在推理路径上使用 CUDA 扩展的地方，也是
> `compressor.py` 必须把 `device` 设为 `cuda` 才有效率的原因；
> 而解压端的谱预测完全是手写矩阵乘法，**在纯 CPU 上也能跑**。

---

## 6. 两套实现的一致性检查（可以自己验证的等价关系）

| 步骤 | 训练/压缩端（并行） | 解压端（递归） |
|---|---|---|
| 行间预测 | `model.py:389-399` 一次算 T 行 | `inference.py:273-285` 每次算 1 行 |
| 行递归状态 | 隐含在 `WKV` CUDA 核里（不带状态） | `self.state_lines`（跨调用常驻） |
| 波段预测 | `inference.py:660-661` 一次算 I 个波段 | `inference.py:440-452` 每波段一步 + `reset_state()` |
| 波段递归状态 | 隐含在 CUDA 核里 | `self.state_bands`（每行重置） |
| Π 的定义 | `model.py:401` / `inference.py:658` | `inference.py:433` |
| 数值稳定性 `max` 技巧 | `cuda/wkv_cuda.cu:28-40` | `inference.py:230-242` / `400-413` |

两套实现输出的差异只来自 **FP32 浮点累加顺序**（CUDA 核按 T 顺序累加 vs
手写按 step 累加其实顺序相同，主要差异在 `-use_fast_math` 和向量化）。
论文 Sec. III-E 正是为了这个才引入 `numerical warning` 机制：

> “Whenever there is a mismatch in the hardware-software stack between the compressor and
> the decompressor, the prediction will only be accurate down to numerical precision …
> This means that a numerical perturbation might cause the predicted value to **cross the
> rounding threshold**, causing a decoding error. In order to avoid this, whenever the
> prediction is closer to the rounding threshold than **10⁻³** … we signal **the side of the
> threshold** as extra information.”

对应代码见 `compressor.py:175-177` / `193-195` 与 `decompressor.py:180` / `196`。
