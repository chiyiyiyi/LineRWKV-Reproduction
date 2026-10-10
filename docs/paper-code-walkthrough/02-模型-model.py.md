# 02 · `model.py` 逐行讲解（网络本体）

本文件对应论文 Sec. III-B ~ III-E（p.5–6）、Fig. 3、Eq. (1)–(19)，以及 Sec. III-G 的训练侧差异。

`model.py` 共 535 行，可以切成 8 块：

| 行号 | 模块 | 论文位置 |
|---|---|---|
| 18–42 | `LineEncoder` | Sec. III-B，Fig. 2 的 ENCODER |
| 45–69 | `LineDecoder` | Sec. III-E |
| 72–85 | `BandZeroDecoder` | Sec. III-F “For the first band, we employ spatial prediction only…” |
| 88–114 | `RWKV_ChannelMix` | Eq. (8)–(10) |
| 117–167 | `RWKV_TimeMix` | Eq. (1)–(7) |
| 171–213 | `WKV`（CUDA autograd） | Eq. (4)–(6) 的数值稳定实现 |
| 216–262 | `Block` | Fig. 3 的 LINE/BAND MIXING + CHANNEL MIXING |
| 265–532 | `LineRWKV`（整体） | Sec. III-A/III-G，Alg. 1 |

---

## 0. 顶部导入（1–16 行）

```python
1  import os, math, gc, importlib
6  from torch.utils.cpp_extension import load
8  import pytorch_lightning as pl
10 from pytorch_lightning.strategies import DeepSpeedStrategy
12 #import deepspeed
13 #from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
15 import scipy.io as sio
```

- **6**：`torch.utils.cpp_extension.load` 是**运行时 JIT 编译 CUDA 扩展**的入口，第 175 行会用到。
- **8–10**：整套训练用 PyTorch Lightning 组织；`DeepSpeedStrategy` 只被注释掉的代码引用。
- **12–13**：DeepSpeed 相关全部注释掉，说明作者在某次重构后放弃了 DeepSpeed 优化器。
- **15**：`scipy.io` 只出现在被注释掉的调试分支（`model.py:479`），是遗留 import。

---

## 1. `LineEncoder`（18–42 行）—— 跨轨（跨列）编码器

论文 Sec. III-B（p.5）原话：

> “The goal of the encoder is to capture the correlation that exists **across image columns**
> and encode it into a feature space for further processing. The encoder function is, thus,
> a **1-D operation that is shared for all bands and for all lines**. A simple design for the
> encoder is a sequence of blocks composed of **1-D convolution, layer normalization, and
> nonlinear activation**. … we denote the number of such blocks with N_enc. The encoding of
> the input line will result in a feature vector u for each spatial–spectral pixel (i.e.,
> a tensor of size 1 × N_x × N_z × F).”

```python
19      def __init__(self, config):
20          super(LineEncoder, self).__init__()
21          self.config = config
22
23          self.conv_in = nn.Conv1d(1, config.dim_enc, kernel_size=config.kernel_size, stride=1, padding=config.kernel_size//2)
24          self.norm_in = nn.LayerNorm(config.dim_enc)
25
26          self.convs = nn.ModuleList([nn.Conv1d(config.dim_enc, config.dim_enc, kernel_size=config.kernel_size, stride=1, padding=config.kernel_size//2) for _ in range(config.N_layers_encoder-1)])
27          self.norms = nn.ModuleList([nn.LayerNorm(config.dim_enc) for _ in range(config.N_layers_encoder-1)])
```

- **23 `conv_in`**：`Conv1d(1 → F, k=3, padding=1)`。输入通道是 **1**：因为波段维已经被
  独立开（见下面 `forward`），每一路只沿“列”这一个方向卷积。
  `padding=kernel_size//2` 说明**卷积是双向、非因果的**。

  > **这一点非常重要**：论文 Sec. III-A 明确说 “it does **not** use information from the
  > previous pixels on the same line or future lines”，但**允许使用整条当前行/上一行的所有列**。
  > 原因就是“一整行”是卫星推扫成像的最小可并行单元，整行数据都已到手。
  > 所以编码器里的非因果卷积不违反因果性。

- **24 `norm_in`**：`LayerNorm(F)`，对应论文 “layer normalization [38]”。
- **26–27**：再堆 `N_layers_encoder − 1` 个同尺寸卷积块。
  当 `N_layers_encoder = 1`（XS 配置）时这两个 `ModuleList` 为空，
  只剩 `conv_in + norm_in + GELU`，即论文的 “N_enc 个块”最小情况。

```python
29      def forward(self, x):
30          (B,T,C,I) = x.shape
31          x = torch.permute(x,(0,1,3,2)) # (B,T,I,C)
32          x = x.reshape(B*T*I,C).unsqueeze(1) # (BTI,1,C)
33
34          x = self.conv_in(x) # (BTI,F,C)
35          x = self.norm_in(x.permute(0,2,1)).permute(0,2,1)
36          x = F.gelu(x)
37          for i,conv in enumerate(self.convs):
38              x = conv(x)
39              x = self.norms[i](x.permute(0,2,1)).permute(0,2,1)
40              x = F.gelu(x) # (BTI,F,C)
41          x = torch.permute(x.reshape(B,T,I,self.config.dim_enc,C), (0,1,3,4,2)) # (B,T,F,C,I)
42          return x
```

- **30**：输入 `(B=块数, T=行, C=列, I=波段)`。
- **31–32**：把 `(B,T,I)` 折叠成 batch，把 **C（列）当成信号长度**，
  得到 `(B·T·I, 1, C)`。这就是论文说的 “1-D operation shared for all bands and all lines”。
- **34–40**：`Conv1d → LayerNorm → GELU`，重复 `N_layers_encoder` 次
  （`nn.Conv1d` 要求 `(N, C_in, L)`，所以每次 LN 前都要 `permute` 把通道维换到最后一维）。
- **41**：折回 `(B, T, F, C, I)`。

  注意这里的维度顺序是 `F` 在 `C` 前面，是为了后面第 375 行
  `torch.permute(a,[0,3,4,1,2]) → (B,C,I,T,F)` 方便，
  也就是**让“列”和“波段”变成可以并行铺开的 batch 维，把“行”留作序列维**。
- **42**：返回编码结果 `u`（论文里的 `u_y`，对全部行/列/波段）。

> 训练时一次输入 T 行，编码器会把 T 行一起算（省时间）；推理时
> `compressor.py:183` 每次只传 `(1,1,C,I)` 一行，行为完全一致（共享权重、无跨行混合）。

---

## 2. `LineDecoder`（45–69 行）—— 解码器

论文 Sec. III-E（p.6）原话：

> “A decoder neural network produces a prediction of the raw pixel value from the
> feature-domain representation of the spatial prediction û_{y,x,z} and spatial–spectral
> error Π̂_{y,x,z} … This is obtained by **feeding their sum** to a sequence of N_dec blocks
> composed of **1 × 1 convolution, LayerNorm, and nonlinear activation**.”

```python
50      self.conv_out = nn.Conv1d(config.dim_enc, 1, kernel_size=1, stride=1, padding=0)
51      self.convs = nn.ModuleList([nn.Conv1d(config.dim_enc, config.dim_enc, kernel_size=1, stride=1, padding=0) for _ in range(config.N_layers_decoder-1)])
52      self.norms = nn.ModuleList([nn.LayerNorm(config.dim_enc) for _ in range(config.N_layers_decoder-1)])

54      def forward(self, x, delta):
55          #x: (P(T-1),I-1,F)
56          #delta: (P(T-1),I-1,F)
57          x = x.permute((0,2,1)) # (P(T-1),F,I-1)
58          delta = delta.permute((0,2,1)) # (P(T-1),F,I-1)
59
60          x_out = x+delta
61          for i,conv in enumerate(self.convs):
62              x_out = conv(x_out)
63              x_out = self.norms[i](x_out.permute(0,2,1)).permute(0,2,1)
64              x_out = F.gelu(x_out) # (BTI,F,C)
65          x_out = self.conv_out(x_out) # (P(T-1),1,I-1)
66
67          #x_out = self.conv_out(x) + self.conv_out(delta) # (P(T-1),1,I-1)
68
69          return x_out[:,0,:] # (P(T-1),I-1)
```

- **50–52**：**全部用 `kernel_size=1`**。这是论文两条论述的交点：
  1. Sec. III-E：“1 × 1 convolution”；
  2. Sec. III-G：1×1 卷积**不混合列**，所以“编码器之后的所有操作对列独立”，
     训练时可以放心地随机抽样若干列（`pos_size`）来省显存。

  若把这里换成 `kernel_size=3`，列子采样就不再等价，整个训练加速技巧会失效
  （`model.py:377` 的注释 `# requires 1x1 decoder` 就是在提醒这一点）。
- **54–58**：把 `(..., I-1, F)` 转成 `(..., F, I-1)`，让**波段**成为 Conv1d 的信号长度。
  注意输入只用了 `I−1` 个波段：**band 0 由 `BandZeroDecoder` 单独处理**（见下节）。
- **60**：`x_out = x + delta`，即论文的 “feeding their sum”：`û + Π̂`。
- **61–65**：`N_layers_decoder − 1` 个 `Conv1d(F→F, 1×1) + LN + GELU`，最后一层
  `conv_out: F → 1` 把特征压成一个标量——**每个像素一个预测值**。
- **67**：被注释掉的替代写法（分别过卷积再相加），作者试过但没采用。
- **69**：去掉单通道维，返回 `(B·(T−1), I−1)`，即除 band 0 之外的所有波段预测。

---

## 3. `BandZeroDecoder`（72–85 行）—— 第一波段的专用解码器

论文 Sec. III-F（p.6）原话：

> “For the first band, we employ **spatial prediction only** by feeding the output of the
> line predictor to a **dedicated decoder neural network** with architecture comparable to
> that described in Section III-E; the prediction error is then entropy-coded.”

```python
73      def __init__(self, config):
77          self.conv_out = nn.Conv1d(config.dim_enc, 1, kernel_size=1, stride=1, padding=0)
79      def forward(self, x):
80          #x: (P(T-1),1,F)
81          x = x.permute((0,2,1)) # (P(T-1),F,1)
82
83          x_out = self.conv_out(x) # (P(T-1),1,1)
85          return x_out
```

- 结构是 `LineDecoder` 的**退化版**：没有隐藏层，只有一个 `F → 1` 的 1×1 卷积。
- **80**：输入只有 **1 个波段**（band 0），因为第一波段没有“前一个波段”可供谱预测。

### 3.1 一个论文没写、但代码里很关键的细节

论文 Alg. 1 把它记成 `E_{y,x,0} = Decoder₀(û_{y,x,0}) − I^q_{y,x,0}`，看起来是一个
**绝对值的预测**。但代码里在解码器输出上又加了一项：

```python
model.py:407        x0_hat = x0_hat + x0_in     # x0_in = x[:,:-1,:,0:1]，即"上一行同列 band 0"
compressor.py:172   predicted_value = predicted_value[:,0,0] + (data_rec[0,line_no,:,0] - mu)/sigma
decompressor.py:178 predicted_value = predicted_value[:,0,0] + next_line_hat[0,0,:,0]
```

三处完全一致，都指向“**上一行同列、band 0 的（归一化）像素值**”。也就是说，这个专用
解码器实际学的是**相对上一行的增量**（沿轨方向的 DPCM 残差），而不是绝对值。
论文的 Alg. 1 / Fig. 2 都没有画出这一条跳连（Fig. 2 的 caption 明确写了
“for z > 1”，即只画了一般波段），这是实现层面的额外设计。

---

## 4. `RWKV_ChannelMix`（88–114 行）—— 通道混合

论文 Eq. (8)–(10)（p.5 右栏）：

> r_{y−1} = W'_r (µ'_r u_{y−1} + (1 − µ'_r) u_{y−2})  (8)
> k_{y−1} = W'_k (µ'_k u_{y−1} + (1 − µ'_k) u_{y−2})  (9)
> o_{y−1} = σ(r_{y−1}) ⊙ ( W'_v max(k_{y−1}, 0)² )     (10)

```python
93      self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
95      with torch.no_grad():  # fancy init of time_mix
96          ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
97          ddd = torch.ones(1, 1, config.dim_enc)
98          for i in range(config.dim_enc):
99              ddd[0, 0, i] = i / config.dim_enc
100         self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
101         self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
103     self.key = nn.Linear(config.dim_enc, config.dim_ffn, bias=False)
104     self.receptance = nn.Linear(config.dim_enc, config.dim_enc, bias=False)
105     self.value = nn.Linear(config.dim_ffn, config.dim_enc, bias=False)

107     def forward(self, x):
108         xx = self.time_shift(x)
109         xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
110         xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
111         k = self.key(xk)
112         k = torch.relu(k) ** 2
113         kv = self.value(k)
114         return torch.sigmoid(self.receptance(xr)) * kv
```

- **93**：`ZeroPad2d((0,0,1,-1))` 作用在最后两维上，等价于“在序列维前面补 1 个 0、
  去掉最后 1 个”⇒ `xx[t] = x[t−1]`，`xx[0] = 0`。
  这正是论文的 **单位延迟 z⁻¹**（Fig. 3 里那个 `z⁻¹` 虚线框），
  也对应 Alg. 1 的初始化 `Init u_{−1,x,z} = 0`。
- **95–101**：RWKV 原版的“花式初始化”。`ddd = [0, 1/F, 2/F, …, (F−1)/F]`，
  再取 `(1 − layer_id/n_layer)` 次幂。其效果是：
  **浅层更倾向直接用当前时刻 x（µ≈1），深层更倾向混入历史 x（µ 更小）**。
  论文只说 “µ are some hyperparameters / trainable?”，代码说明它们**是可训练参数**
  （`nn.Parameter`），只是初值这么取。
  注意每个 `ChannelMix` 用自己的 `layer_id`，所以 `config.n_layer` 必须在构造前
  设成对应的层数（见 `model.py:291-294`）。
- **103–105**：`key: F → dim_ffn`，`value: dim_ffn → F`，`receptance: F → F`，全无 bias。
  这与 Eq. (10) 的 `W'_r, W'_k, W'_v` 一一对应，只是把中间维度从 F 放大到 `dim_ffn`
  再压回 F（论文没写这个瓶颈结构，代码里 `dim_ffn` 默认等于 `dim_enc`）。
- **108–110**：实现 Eq. (8)(9) 的时间混合。因为 `numpy/torch` 的广播，
  `time_mix_*` 形状 `(1,1,F)` 与 `xk` 形状 `(N,T,F)` 可直接逐元素相乘。
- **111–113**：`k = W'_k x_k`，再 `relu(·)²`，再 `W'_v`。**`relu(·)²` 就是 Eq. (10) 的
  `max(k,0)²`**，是 RWKV 的核心非线性（比 GELU 便宜）。
- **114**：`σ(W'_r x_r) ⊙ (W'_v relu²)` ⇒ 完全等价于 Eq. (10)。名字 “receptance”
  是 RWKV 论文对“用 sigmoid 而不是 softmax 做接收门”的叫法（receptance = reception +
  acceptance）。

> 论文 Eq. (8)–(10) 里的 `u_{y−1} / u_{y−2}` 是**行**（线预测器视角）；在谱预测器里同样的
> 公式描述的是**波段** `z−1 / z−2`。代码是同一个类，只靠输入张量的序列维是“行”还是“波段”来区分。

---

## 5. `RWKV_TimeMix`（117–167 行）—— 线性注意力（WKV）

论文 Eq. (1)–(7)（p.5 右栏）：

> r_{y−1} = W_r (µ_r u_{y−1} + (1 − µ_r) u_{y−2})   (1)
> k_{y−1} = W_k (µ_k u_{y−1} + (1 − µ_k) u_{y−2})   (2)
> v_{y−1} = W_v (µ_v u_{y−1} + (1 − µ_v) u_{y−2})   (3)
> **a_{y−1} = e^{−α} ⊙ a_{y−2} + e^{k_{y−1}} ⊙ v_{y−1}**  (4)
> **b_{y−1} = e^{−α} ⊙ b_{y−2} + e^{k_{y−1}}**          (5)
> p_{y−1} = ( a_{y−2} + e^{β + k_{y−1}} ⊙ v_{y−1} ) / ( b_{y−2} + e^{β + k_{y−1}} )   (6)
> o_{y−1} = W_o ( σ(r_{y−1}) ⊙ p_{y−1} )             (7)
> where W_*, α, and β are **trainable parameters**, µ_* are hyperparameters …
> The states in the recursion are initialized as a = b = 0.

```python
146     self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
147     self.key = nn.Linear(config.dim_enc, config.dim_att, bias=False)
148     self.value = nn.Linear(config.dim_enc, config.dim_att, bias=False)
149     self.receptance = nn.Linear(config.dim_enc, config.dim_att, bias=False)
150     self.output = nn.Linear(config.dim_att, config.dim_enc, bias=False)

152     def jit_func(self, x):
153         xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
154         xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
155         xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
156         xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
157         k = self.key(xk)
158         v = self.value(xv)
159         r = self.receptance(xr)
160         sr = torch.sigmoid(r)
161         return sr, k, v

163     def forward(self, x):
164         B, T, C = x.size()  # x = (Batch,Time,Channel)
165         sr, k, v = self.jit_func(x)
166         rwkv = sr * RUN_CUDA(B, T, self.config.dim_att, self.time_decay, self.time_first, k, v)
167         return self.output(rwkv)
```

- **147–150**：对应 Eq. (1)–(3) 的 `W_r, W_k, W_v` 与 Eq. (7) 的 `W_o`。
  注意内部维度是 `dim_att`（默认 = `dim_enc`）；`assert self.dim_att % 32 == 0`
  （`model.py:279`）是为了 CUDA kernel 的线程块大小（`min(C,32)`）。
- **153–156**：Eq. (1)–(3) 的时间混合，与 ChannelMix 完全同构。
  唯一的差别是 `time_mix_v` 的初值多加了 `0.3 * ratio_0_to_1`（第 143 行），
  让 `v` 通道的混合比例在层间变化更快——这是 RWKV 原版的小技巧，论文未提。
- **157–160**：`k, v, r` 三个线性投影，`sr = σ(r)`。
- **166**：**核心**：`sr ⊙ WKV(ω, β, k, v)`。这里的 `WKV` 就是 Eq. (4)–(6) 的递归，
  只不过被写成了 GPU 上的前向/反向核，以便训练时并行（详见第 6 节）。
- **167**：Eq. (7) 的输出投影 `W_o`。

### 5.1 `time_decay` / `time_first` 与论文 α、β 的确切换算

```python
131             decay_speed = torch.ones(config.dim_att)
132             for h in range(config.dim_att):
133                 decay_speed[h] = -5 + 8 * (h / (config.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
134             self.time_decay = nn.Parameter(decay_speed)
138             zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(config.dim_att)]) * 0.5
139             self.time_first = nn.Parameter(torch.ones(config.dim_att) * math.log(0.3) + zigzag)
```

- **131–134 `time_decay`**：初值均匀散布在 **[−5, +3]**。
  它在 `WKV.forward` 里立刻被替换成 `w = −exp(time_decay)`（`model.py:184`），
  而递归里用的是 `ww = pp + w`，也就是**每一步把历史状态乘以 e^{w}**。

  对照论文 Eq. (4) 的 `e^{−α}`，得到换算关系：

  ```
  w = −e^{time_decay}      ⟹      e^{−α} = e^{w}   ⟹   α = e^{time_decay}
  ```

  即**代码里存的是 log α**。初值 −5 ⇒ α≈0.0067（历史几乎立刻衰减，近似只看当前行）；
  初值 +3 ⇒ α≈20 ⇒ 历史立刻归零。所以 `time_decay` 越大学得越“短视”。
  `inference.py:148`、`inference.py:320` 在加载权重时做的 `w[x] = -torch.exp(w[x])`
  就是把这个换算提前算好。

- **138–139 `time_first`**：初值 `log(0.3)` 加上一个 `[−0.5, 0, +0.5]` 的三周期锯齿扰动。
  它就是 Eq. (6) 里的 **β**，作用是在归一化的 WKV 里给**当前** token 一个额外的
  “第一时刻加成” `e^{β+k}`。RWKV 原版的 `u` 参数，这里直接改名成 `time_first`。

---

## 6. `RUN_CUDA` 与 `WKV`（171–213 行）

论文脚注（p.5 右栏）：

> “We refer the reader to [23] for details about a **numerically-stable implementation**.”

```python
171 def RUN_CUDA(B, T, C, w, u, k, v):
172     return WKV.apply(B, T, C, w, u, k, v)
174 T_MAX = 512 # config.ctx_len
175 wkv_cuda = load(name=f"wkv_{T_MAX}", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"], verbose=True,
                    extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3",
                                       "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
176 class WKV(torch.autograd.Function):
178     def forward(ctx, B, T, C, w, u, k, v):
184         w = -torch.exp(w.contiguous())
189         y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
190         wkv_cuda.forward(B, T, C, w, u, k, v, y)
191         ctx.save_for_backward(w, u, k, v, y)
193         return y
196     def backward(ctx, gy):
203         gw = torch.empty((B, C), ...); gu = ...; gk = ...; gv = ...
208         wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
210         gw = torch.sum(gw, dim=0)
211         gu = torch.sum(gu, dim=0)
213         return (None, None, None, gw, gu, gk, gv)
```

- **175**：JIT 编译。`-DTmax=512` 把 `T_MAX` 编译进 CUDA 源码，
  该值被用在反向核里的**栈数组** `F q[Tmax], r[Tmax]`（`wkv_cuda.cu:63`）。
  **这是“训练最多 512 行”这一硬约束的根源。**
  `--maxrregcount 60` 与 `-res-usage` 是作者为了压低寄存器占用、提升 occupancy 加的。
- **184**：把可训练的 `time_decay` 参数变成等式 (4) 里的 `−α`（负值 ⇒ 衰减）。
- **189–190**：输出 `y` 显式分配成 `(B,T,C)` 连续内存，直接交给 CUDA 核写入——
  避免 PyTorch 的中间张量开销。这就是论文 “linear complexity instead of quadratic” 的落地方式。
- **210–211**：`gw/gu` 是**逐 batch 元素**的梯度（形状 `(B,C)`），因为 `w,u` 是
  每个通道一个标量、在 batch 内共享，所以要把 batch 维求和。这也是为什么
  CUDA 核内部把 `w,u` 当作“每个 `_c` 一个标量”来用（`wkv_cuda.cu:15-16`）。

---

## 7. `Block`（216–262 行）—— Fig. 3 的一个完整块

论文 Fig. 3 说明（p.5）：

> “Line and spectral predictors are made of the repetition of **line/band mixers** and
> **channel mixers** according to the depicted architecture. z⁻¹ denotes the unit delay in
> the sequence dimension (lines or bands). **LN** denotes the LayerNorm, and **LIN** denotes
> the linear projection of features.”

论文 Sec. III-C：

> “The output o_{y−1} of the last channel mixing module in the line predictor is **summed
> with the input of the module in a residual fashion**, as depicted in Fig. 3, to generate
> û_y …”

```python
222         self.ln1 = nn.LayerNorm(config.dim_enc)
223         self.ln2 = nn.LayerNorm(config.dim_enc)
225         if self.layer_id == 0:
226             self.ln0 = nn.LayerNorm(config.dim_enc)
228         self.att = RWKV_TimeMix(config, layer_id)
229         self.ffn = RWKV_ChannelMix(config, layer_id)
231         if config.tiny_att_dim > 0 and self.layer_id == config.tiny_att_layer:
232             ...   # tiny attention，本项目默认关闭
238         if config.dropout > 0:
239             self.drop0 = nn.Dropout(p = config.dropout)
240             self.drop1 = nn.Dropout(p = config.dropout)

242     def forward(self, x, x_emb=None):
244         B, T, C = x.size()
245         if self.layer_id == 0:
246             x = self.ln0(x)
248         if self.config.dropout == 0:
249             x = x + self.att(self.ln1(x))
250             x = x + self.ffn(self.ln2(x))
251         else:
252             x = self.drop0(x + self.att(self.ln1(x)))
253             x = self.drop1(x + self.ffn(self.ln2(x)))
```

- **245–246 `ln0`**：**只在第一层**额外加一个 LayerNorm，对应 Fig. 3 最左边那个
  “LN”方块（输入进入 predictor 前的归一化）。在递归实现里它是硬编码的
  `if i == 0: x_spatial = self.LN(x_spatial, w.blocks_lines[i].ln0)`
  （`inference.py:274-275`、`inference.py:441-442`）。
- **249–250**：`x = x + TimeMix(LN(x)); x = x + ChannelMix(LN(x))`
  —— 与 Fig. 3 的两条残差支路完全一致。
  论文 Sec. III-C 把“残差相加”描述在最后一个 channel mixing 之后，代码里则是
  **每个子模块后都加一次**，两者在数学上等价（因为两个子块共享同一个输入 x，
  且残差是恒等映射）。
- **231–237 tiny attention**：RWKV-v4 的可选组件，本项目
  `tiny_att_layer=-1, tiny_att_dim=0` ⇒ 整个分支不构造。注意这段代码引用了
  `config.n_embd`（本项目无此字段），所以**一旦打开就会崩**，属于原仓库遗留。
- **238–240 / 251–253**：dropout 分支，默认 `dropout=0` 走上面的简洁路径。

---

## 8. `LineRWKV`（265–532 行）—— 组装整个预测器

### 8.1 `__init__`（266–298）

```python
287         self.encoder = LineEncoder(config)
288         self.decoder = LineDecoder(config)
289         self.bandzerodec = BandZeroDecoder(config)
290
291         config.n_layer = self.n_layer_lines
292         self.blocks_lines = nn.ModuleList([Block(config, i) for i in range(self.n_layer_lines)])
293         config.n_layer = self.n_layer_bands
294         self.blocks_bands = nn.ModuleList([Block(config, i) for i in range(self.n_layer_bands)])
```

- **287–289**：论文 Fig. 2 里的三个非 RWKV 模块（encoder / decoder / band-0 decoder）。
- **291–294**：**两套独立的 RWKV 堆叠**：
  `blocks_lines` 处理行的序列，`blocks_bands` 处理波段的序列。
  这正是论文 Sec. III-A 所说的：

  > “Note that the context used by LineRWKV for prediction is slightly different from
  > conventional nonneural designs. Specifically, the predictor employs all lines above the
  > current pixel and all previous bands for the current pixel, but it **does not use
  > information from the previous pixels on the same line** or future bands. This is due to
  > the LineRWKV design, which employs **separate blocks in the spectral, along-track, and
  > across-track dimensions**.”

- **291/293**：`config.n_layer` 被**就地改写**两次。这是必须的，因为
  `RWKV_TimeMix/ChannelMix` 的初始化依赖 `config.n_layer`（第 96、124–125、135 行）。
  副作用：`config` 对象在 `LineRWKV` 构造完成后，`n_layer` 的值会停在 `n_layer_bands`。
  由于 `train.py:41` 早于模型构造就 dump 了 config，所以 `config.txt` 里没有 `n_layer`
  这个字段（可以自行核对 `log_dir/linerwkv/*/config.txt`）。

### 8.2 `configure_optimizers`（300–347）

```python
307         for n, p in self.named_parameters():
308             if ("time_mix" in n) and (config.layerwise_lr > 0):
309                 lr_1x.add(n)
310             elif ("time_decay" in n) and (config.layerwise_lr > 0):
311                 lr_2x.add(n)
314             elif ("time_first" in n) and (config.layerwise_lr > 0):
315                 lr_3x.add(n)
316             elif (len(p.squeeze().shape) >= 2) and (config.weight_decay > 0):
317                 lr_decay.add(n)
318             else:
319                 lr_1x.add(n)
330                 {"params": [...lr_1x...], "weight_decay": 0.0, "my_lr_scale": 1.0},
331                 {"params": [...lr_2x...], "weight_decay": 0.0, "my_lr_scale": 2.0},
332                 {"params": [...lr_3x...], "weight_decay": 0.0, "my_lr_scale": 3.0},
347             return torch.optim.AdamW(optim_groups, lr=config.lr_init, betas=config.betas, eps=config.adam_eps, weight_decay=0, amsgrad=False)
```

- 完全照搬 RWKV 官方仓库的训练配方：把 `time_mix / time_decay / time_first`
  分成 1×/2×/3× 学习率三组，其余权重 1×。
- **312 行的 `time_faaaa` 分支是死代码**（原仓库别名），实际参数名是 `time_first`。
- **347**：`weight_decay=0`（`config.weight_decay = 0`），走的是 else 分支；
  如果打开权重衰减，会走 **342** 行的 `AdamW`。
- `my_lr_scale` 通过 `trainer.py:41-42` 的
  `param_group["lr"] = lr * param_group["my_lr_scale"]` 生效。

### 8.3 `forward`（357–437）—— 全文件最核心的 80 行

先看注释定义的记法（第 358–362 行）：

```python
358         # x: (B,T,C,I)
359         # B: batch size
360         # T: number of lines (sequence length)
361         # C: number of columns
362         # I: number of bands
```

#### （a）归一化 367–369

```python
367         mu = torch.mean(x,dim=(1,2),keepdim=True) # (B,1,1,I)
368         sigma = torch.std(x,dim=(1,2),keepdim=True) # (B,1,1,I)
369         x = (x-mu)/(sigma+1e-8) # (B,T,C,I)
```

逐波段、在「行×列」平面上算均值和标准差。论文没有任何公式描述这一步，但 Sec. III-E
提到了“denormalized”，且侧信息文件里确实在传 μ/σ（`compressor.py:216-231`）。
**训练用整块统计量、推理用逐行统计量的不一致**见 `00-总览.md` 差异清单第 2 条。

#### （b）编码 + 列/波段并行化 371–386

```python
371         a = self.encoder(x) # (B,T,F,C,I)
374         x0_in = torch.reshape(torch.permute(x[:,:-1,:,0:1],[0,2,1,3]),[B*C,T-1,1])
375         a = torch.reshape(torch.permute(a,[0,3,4,1,2]),[B*C,I,T,config.dim_enc]) # (B,T,F,C,I) --> (BC,I,T,F)
378         if self.mode == 'train':
379             self.pos = torch.randperm(B*C)[:self.config.pos_size]
380             a = a[self.pos,:,:,:] # (P,I,T,F)
381             P = a.size(0)
382             mu = mu.repeat((1,C,1,1)).reshape([B*C,1,I])[self.pos] # (P,1,I)
383             sigma = sigma.repeat((1,C,1,1)).reshape([B*C,1,I])[self.pos]
384             x0_in = x0_in[self.pos,:,:] # (P,1,T-1)
385         else:
386             P = B*C
```

- **374 `x0_in`**：取**每一行 band 0 的归一化像素**，重排成 `(B·C, T−1, 1)`。
  第 t 个元素 = 第 t 行的 band 0 ⇒ 对应“目标行的上一行”，供 band-0 解码器做跳连
  （见第 3.1 节）。
- **375**：`(B,T,F,C,I) → (B,C,I,T,F) → (B·C, I, T, F)`。
  经过这一步，**列 C 和波段 I 都变成了 batch 维，T 保持为序列维**——
  于是线预测器可以对 `B·C·I` 条独立序列并行跑 RWKV。
  这就是论文 Sec. III-G 说的“**并行实现**”。
- **378–385 列子采样**：`randperm(B·C)[:pos_size]` 随机挑若干**列**。
  注意随机是在 `B·C` 上做的，所以同一 batch 内不同图共享同一组列索引，
  但 `mu/sigma/x0_in` 都跟着一起挑，保证一致。
  论文 Sec. III-G 的原话（见 `01-配置与数据.md` 第 A 节 `pos_size`）在此落地。
  **这也解释了为什么 `train.py` 用 batch_size=8 却不会爆显存**：
  真正进预测器的元素数是 `pos_size × I × T`，而 `pos_size` 只有 16～128。

#### （c）线预测器（RWKV over lines）388–399

```python
389         a = torch.reshape(a,[-1,T,config.dim_enc]) # (PI,T,F)
390         x_spatial = torch.reshape(a,[-1,T,config.dim_enc]) # (PI,T,F)
391         x_spatial_emb = x_spatial
392         if config.dropout > 0:
393             x_spatial = self.drop0_lines(x_spatial)
394         if config.tiny_att_dim > 0:
395             for block in self.blocks_lines:
396                 x_spatial = block(x_spatial, x_spatial_emb)
397         else:
398             for block in self.blocks_lines:
399                 x_spatial = block(x_spatial)
```

- 这就是论文 Sec. III-C 的 Line Predictor：一串 `Block`，序列维是**行**。
- **399**：输出 `x_spatial` 即论文的 **û**（第 t 个位置的输出是由第 ≤t 行推出的、
  对第 t+1 行特征的预测）。

#### （d）线残差 Π（论文 Eq. 11）401–403

```python
401         delta = a[:,1:,:] - x_spatial[:,:-1,:] # (PI,T-1,F)
402         delta = delta.reshape((P,I,T-1,config.dim_enc)).permute((0,2,1,3)).reshape((P*(T-1),I,config.dim_enc))
403         x_spatial = x_spatial[:,:-1,:].reshape((P,I,T-1,config.dim_enc)).permute((0,2,1,3)).reshape((P*(T-1),I,config.dim_enc))
```

- **401**：`delta[t] = a[t+1] − x_spatial[t]`，即
  **第 t+1 行的真实特征 − 对它的预测** ⇒ 论文 Eq. (11)：

  > Π_{y,x,z} = u_{y,x,z} − û_{y,x,z}   (11)

  论文紧接着解释了这个残差为什么可用：

  > “Notice that Π_{y,x,z} requires pixel I^q_{y,x,z} for its computation. Therefore, this
  > **needs to have been already decoded**, and its availability is ensured by treating the
  > first band as a special case.”

  训练/压缩端用真值，解压端用刚解出的重建值——这也是为什么解压端必须逐波段串行。
- **402–403**：把 `(P·I, T−1, F)` 重排成 `(P·(T−1), I, F)`：
  **序列维从“行”换成“波段”**，为谱预测器做准备；同时 `x_spatial` 也做同样重排，
  但它保留全部 I 个波段（后面 band 0 与 band≥1 各取所需）。

#### （e）band 0 的预测 405–412

```python
405         x0_hat = self.bandzerodec(x_spatial[:,0:1,:]) # (P(T-1),1,1)
406         x0_hat = torch.reshape(x0_hat,[P,T-1,1])
407         x0_hat = x0_hat + x0_in
408         if self.mode != 'train':
409             x0_hat = torch.permute( torch.reshape(x0_hat,[B,C,T-1,1]), [0,2,1,3] ) # (B,T-1,C,1)
410             x0_hat = x0_hat*sigma[...,0:1] + mu[...,0:1]
411         else:
412             x0_hat = x0_hat*sigma[...,0:1] + mu[...,0:1] # (P,T-1,1)
```

- **405**：取**波段 0**的 û 喂给 `BandZeroDecoder`。
- **407**：加上 `x0_in`（上一行 band 0 的归一化值）——论文没写的跳连，见第 3.1 节。
- **408–412**：反归一化回原像素尺度。注意 `train` 模式下不重排回 `(B,T−1,C,...)`，
  而是保持 `(P,T−1,...)`；因为训练时已经随机抽样过列（`pos`），
  重排回 `(B,C)` 反而会破坏抽样语义。**这是 train / 非 train 分支存在的原因之一。**

#### （f）谱预测器（RWKV over bands）414–424

```python
415         delta_spectral = delta+0.0
416         delta_spectral_emb = delta_spectral
417         if config.dropout > 0:
418             delta_spectral = self.drop0_bands(delta_spectral)
419         if config.tiny_att_dim > 0:
420             for block in self.blocks_bands:
421                 delta_spectral = block(delta_spectral, delta_spectral_emb)
422         else:
423             for block in self.blocks_bands:
424                 delta_spectral = block(delta_spectral) # (P(T-1),I,F)
```

- 对 Π 序列做 RWKV，序列维是**波段**。论文 Sec. III-D：

  > “A causal model over the Π_z sequence is needed to obtain features describing the pixel
  > to be predicted. For this spectral prediction model, we propose to also use RWKV blocks.”

  以及为什么在 Π 而不是在 û 上做：

  > “notice that spectral prediction is performed on the **difference sequence Π_z** rather
  > than the outputs of the line predictors as this allows to use the information of the
  > pixel in the same line to be encoded **but from previous bands**, which is typically the
  > highest source of correlation.”

- **424**：输出 `delta_spectral[z]` 是**由 Π_{≤z} 推得的对 Π_{z+1} 的预测**（Π̂）。

#### （g）解码 + 拼回 band 0 426–435

```python
426         x = self.decoder(x_spatial[:,1:,:], delta_spectral[:,:-1,:]) # (P(T-1),I,F) --> (P(T-1),I-1)
427         x = x.reshape((P,T-1,I-1))
429         if self.mode != 'train':
430             x = torch.permute( torch.reshape(x,[B,C,T-1,I-1]), [0,2,1,3] ) # (B,T-1,C,I-1)
431             x = x*sigma[...,1:] + mu[...,1:]
432         else:
433             x = x*sigma[...,1:] + mu[...,1:]
435         x = torch.cat([x0_hat,x],dim=-1) # (B,T-1,C,I) 或 (P,T-1,I)
437         return x
```

- **426**：`decoder(û_{z=1..I−1}, Π̂_{z=0..I−2})`。
  第一个参数是波段 1…I−1 的线预测特征，第二个参数是**波段左移一位**的谱预测——
  于是“预测波段 z”时喂进去的正是“基于波段 <z 推出来的 Π̂”，
  与论文 Fig. 2 中 `Δ̂_{z−1}` 的箭头一致，实现严格因果。
- **427**：展开回 `(P, T−1, I−1)`；**430** 再变成 `(B,T−1,C,I−1)`（非训练时）。
- **435**：把 band 0 的预测拼回去，得到完整的 `(…, T−1, C, I)` 预测值。

  > 输出行数是 **T−1**：输入 T 行只能预测出后面的 T−1 行，
  > 与 `dataset.py:75-76` 的 `image[:-1] / image[1:]` 切片严格对应。

### 8.4 `training_step`（440–454）

```python
441         self.mode = 'train'
442         x, y = batch
444         logits = self(x)
446         B, T, C, I = y.size()
447         y = torch.reshape(torch.permute(y,[0,2,1,3]),[-1,T,I])[self.pos,:-1,:]
450         loss = F.l1_loss(logits, y)
452         self.log('train_l1',loss)
454         return loss
```

- **447**：`y` 从 `(B,T,C,I)` 变成 `(B·C,T,I)`，再用 `forward` 里存下的 `self.pos`
  做**同样的列抽样**，最后丢掉最后一行 → `(P, T−1, I)`，与 `logits` 对齐。
  `self.pos` 由 `forward` 内部产生（`model.py:379`），所以**必须**先 `self(x)` 再用它。
- **450**：`F.l1_loss` ⇒ 论文 Sec. III-G：

  > “LineRWKV is trained by minimizing the ℓ1 loss between the predicted and true pixel
  > values. Notice that this means that we **only optimize for lossless compression while
  > never accounting for quantization during training**. Nevertheless, lossy rate-distortion
  > tests still show excellent performance.”

  这就是“为什么模型只按无损目标训练，却在有损档位也赢”的官方解释。

### 8.5 `validation_step`（461–479）

```python
462         self.mode = 'val'
466         logits = self(x)
472         l1_val = F.l1_loss(logits, y[:,:-1,:,:])
473         self.log('val_l1',l1_val)
```

- 验证时 `mode='val'`，`forward` 不做列抽样、输出保留 `(B,T−1,C,I)`，
  与 `y[:,:-1]` 直接比较。
- 464–465 与 475–479 的注释掉代码是作者调试用的（把 GT/预测/误差写进 TensorBoard 或 `.mat`）。

### 8.6 `generate_init_weight`（482–532）—— 死代码

```python
491         for n in self.state_dict():
499                 layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer   # ← self.args 不存在
```

- **499** 引用了不存在的 `self.args`；正常训练路径从不调用它
  （`trainer.py` 里那个同名函数也只在 `my_pile_stage` 流程里用，同样不会被 `train.py` 触发）。
  这是从原始 RWKV 仓库整体搬过来的初始化工具，**本项目实际使用的是 PyTorch 默认初始化**。

---

## 9. 小结：`model.py` 与论文的等式对应关系

| 论文 | 代码 |
|---|---|
| Eq. (1)(2)(3) 时间混合 r,k,v | `model.py:152-161`（`jit_func`） |
| Eq. (4)(5) 状态递推 a,b | `cuda/wkv_cuda.cu:34-40`（前向核） |
| Eq. (6) 归一化注意力 p | `cuda/wkv_cuda.cu:28-32` |
| Eq. (7) 输出投影 o = W_o(σ(r)⊙p) | `model.py:166-167` |
| Eq. (8)(9)(10) 通道混合 | `model.py:108-114` |
| Eq. (11) Π = u − û | `model.py:401` |
| Eq. (12)–(16) 波段混合 | 同一套 `RWKV_TimeMix` + `cuda/wkv_cuda.cu`，序列维换成波段 |
| Eq. (17)(18)(19) 波段通道混合 | `model.py:108-114` |
| Sec. III-E 解码器 û + Π̂ | `model.py:60`（`x_out = x + delta`） |
| Sec. III-F 第一行 DPCM / 第一波段专用解码器 | `model.py:405-412` + `compressor.py:132-149` |
| Sec. III-G 并行训练 + 列子采样 | `model.py:378-385`、`model.py:389-399` |
