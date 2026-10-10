# 06 · `cuda/` 内核与 `launcher_*.sh`

---

## A. 为什么需要 CUDA 扩展

论文 Sec. III-C（p.5）：

> “As discussed in Section II-C, it is desirable to have an attention-based operation such as
> the scaled dot-product attention of the Transformer to process the sequence of lines, both
> to **parallelize training** and to exploit the power of attention to adapt to self-similar
> patterns. However, **Transformers require keeping several lines and their features in
> memory, which is prohibitive, and have quadratic computational cost in the number of
> lines**, which is also prohibitive. On the other hand, training a traditional RNN on a large
> scale would be extremely slow and would not scale well.”

> “We refer the reader to [23] for details about a **numerically-stable implementation**.”

落地方式：RWKV 的递推（Eq. 4–6）在**训练时**必须能对“时间/行/波段”这一维并行展开，
同时反向传播需要能沿时间轴高效求梯度。PyTorch 原生实现会生成一条 T 步的图，
所以作者把前向 + 反向都写成了一个 CUDA 扩展。

**在哪里被载入？**

```python
model.py:174-175
    T_MAX = 512 # config.ctx_len
    wkv_cuda = load(name=f"wkv_{T_MAX}", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"], verbose=True,
                    extra_cuda_cflags=[...])

inference.py:546-547   （与上面完全相同的两份代码）
```

- 两处 `load(...)` 的 `name` 都是 `wkv_512`，`sources` 都是相对路径
  `cuda/wkv_op.cpp` / `cuda/wkv_cuda.cu`。
  ⇒ **必须在 `Code/linerwkv/` 目录下运行脚本**，否则相对路径找不到源码
  （`launcher_*.sh` 里的 `cd $model` 就是为了这个）。
- 编译是**惰性且昂贵**的：第一次 import `model.py` / `inference.py` 时会触发 nvcc 编译，
  产物缓存在 `~/.cache/torch_extensions/`。
  `name` 里带 `512` 是为了让不同 `T_MAX` 的编译产物不互相覆盖。
- `verbose=True` 会把 nvcc 的全部输出打到终端。

---

## B. `cuda/wkv_cuda.cu` —— 前向核（7–42 行）

```cuda
7  __global__ void kernel_forward(const int B, const int T, const int C,
8                                 const F *__restrict__ const _w, const F *__restrict__ const _u,
9                                 const F *__restrict__ const _k, const F *__restrict__ const _v,
10                                F *__restrict__ const _y) {
10     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
11     const int _b = idx / C;
12     const int _c = idx % C;
13     const int _offset = _b * T * C + _c;
15     F u = _u[_c];
16     F w = _w[_c];
17     const F *__restrict__ const k = _k + _offset;
...
22     F aa = 0, bb = 0, pp = MIN_VALUE;
```

- **10–13**：**一个线程负责一个 `(batch, 通道)` 车道**（lane），
  `T`（时间/行/波段）这一维由线程内部 `for` 循环串行走完。
  这就是为什么 `model.py:182` 要断言 `B * C % min(C, 32) == 0`：
  线程块大小是 `min(C,32)`，`B*C` 必须能被整除。
- **15–16**：`u` = `time_first`（论文的 **β**），`w` = `−α`
  ——注意 `w` 在 Python 侧已经被 `WKV.forward` 换成 `−exp(time_decay)`；
  这两个量**每个通道一个标量**，在 batch 内共享，所以直接按 `_c` 索引。
- **17**：`_offset = b·T·C + c`，于是 `k[i*C]` 就是第 i 个时间步、第 c 个通道。
  **同一 warp 内相邻线程的 `_c` 连续 ⇒ 每次访存恰好是一整条连续 32 个 float，
  合并访存效率最高。** 这正是把张量布局固定成 `(B, T, C)` 并强制
  `memory_format=torch.contiguous_format`（`model.py:189`）的原因。
- **22**：`aa, bb` 是“已经除以 `exp(pp)` 的滑动和”，`pp` 是当前的缩放指数。
  注释原文就是 `// aa and bb are running sums divided by exp(pp) (to avoid overflow)`。
  `MIN_VALUE = -1e38`（`wkv_cuda.cu:4`），保证第一次 `max(pp, ww)` 一定取 `ww`
  ——对应 Alg. 1 的 `a = b = 0` 初值和手写版 `state[5i+4] -= 1e30`。

```cuda
23     for (int i = 0; i < T; i++) {
24         const int ii = i * C;
25         const F kk = k[ii];
26         const F vv = v[ii];
28         F ww = u + kk;
29         F p = max(pp, ww);
30         F e1 = exp(pp - p);
31         F e2 = exp(ww - p);
32         y[ii] = (e1 * aa + e2 * vv) / (e1 * bb + e2);
34         ww = w + pp;
35         p = max(ww, kk);
36         e1 = exp(ww - p);
37         e2 = exp(kk - p);
38         aa = e1 * aa + e2 * vv;
39         bb = e1 * bb + e2;
40         pp = p;
41     }
```

**这 18 行 = 论文 Eq. (4)(5)(6) 的完整实现。** 逐行对照：

| CUDA 行 | 数学含义 | 论文 |
|---|---|---|
| 28 `ww = u + kk` | `β + k` | Eq. (6) 分母中的 `e^{β+k}` |
| 29–31 | 把 `max(pp, ww)` 作为公共指数，避免 exp 溢出 | 脚注“numerically-stable implementation” |
| 32 `y = (e1·aa + e2·vv)/(e1·bb + e2)` | **输出** = 分子/分母 | **Eq. (6)** `p_{y−1} = (a_{y−2} + e^{β+k}⊙v)/(b_{y−2}+e^{β+k})` |
| 34 `ww = w + pp` | `w = −α` ⇒ 老状态乘 `e^{−α}` | **Eq. (4)(5)** 的 `e^{−α}` 项 |
| 35–37 | 对 `max` 再做一次归一化 | 同 |
| 38 `aa = e1·aa + e2·vv` | `a_{y−1} = e^{−α}⊙a_{y−2} + e^{k}⊙v_{y−1}` | **Eq. (4)** |
| 39 `bb = e1·bb + e2` | `b_{y−1} = e^{−α}⊙b_{y−2} + e^{k}` | **Eq. (5)** |
| 40 `pp = p` | 保存缩放指数供下一步 | — |

> 注意 **32 行写在 38/39 行之前**：输出用的是**更新前**的 `aa, bb`（即 `a_{y−2}, b_{y−2}`），
> 而 38/39 行把状态推进到 `a_{y−1}, b_{y−1}`。
> 这一顺序与论文 Eq. (6) 的写法（分子分母用 `y−2` 下标）严格一致。

> 另外，`y[ii]` 输出的是**归一化后的注意力**（Eq. 6 的 `p`），
> 而 `σ(r)⊙p` 和 `W_o` 是在 PyTorch 侧完成的
> （`model.py:166-167`：`rwkv = sr * RUN_CUDA(...)`，再 `self.output(rwkv)`）。
> 换句话说，**CUDA 只负责 Eq. (4)(5)(6)，Eq. (1)(2)(3)(7) 留在 Python 里**。

---

## C. `cuda/wkv_cuda.cu` —— 反向核（44–119 行）

```cuda
63     F q[Tmax], r[Tmax];
66     for (int i = 0; i < T; i++) {           // 第一遍：正向
72         F ww = u + kk; ...                  // 复现前向
76         const F qq = gy[ii] / (e1 * bb + e2);
77         gw += (ga - gb * yy) * e1 * qq;
78         gu += (vv - yy) * e2 * qq;
79         q[i] = qq;
80         r[i] = ww - p;
86         ga = e1 * (aa + ga);
87         gb = e1 * (bb + gb);
88         aa = e1 * aa + e2 * vv;
89         bb = e1 * bb + e2;
91     }
93     _gw[_offsetBC] = gw * _w[_c];  // multiply by w because of w -> -exp(w) in python forward()
94     _gu[_offsetBC] = gu;
96     aa = 0, bb = 0, pp = MIN_VALUE;
97     for (int i = T - 1; i >= 0; i--) {      // 第二遍：反向
105         F e1 = qq * exp(rr);
106         F e2 = exp(kk + pp);
107         gk[ii] = e1 * (vv - yy) + e2 * (aa * vv + bb);
108         gv[ii] = e1 + e2 * aa;
115         aa = e1 * aa + e2;
116         bb = e1 * bb - e2 * yy;
118     }
```

- **63 `F q[Tmax], r[Tmax]`**：把第一遍需要留给第二遍的中间量放在**线程栈**上，
  长度由编译宏 `-DTmax=512` 决定（`model.py:175`）。
  **这就是训练时行数不能超过 512 的硬性原因**：
  栈数组大小在编译期固定；同时 512 个 float 也接近寄存器/栈的合理上限。
- **66–91 第一遍（正向）**：一边复现前向递推，一边累加
  **`gw`（对 `−α` 的梯度）和 `gu`（对 β 的梯度）**。
  这两个量是**每个通道一个标量**，但梯度来自所有时间步，所以在时间轴上加。
  77–78 行的形式 `gw += (ga − gb·y)·e1·q` 是 RWKV 官方推导出的闭式梯度
  （`y` 是前向输出、`q` 是 `gy/分母`）。
- **93**：**链式法则的关键一步**。Python 里做的是 `w = −exp(time_decay)`，
  所以 `∂L/∂time_decay = ∂L/∂w · ∂w/∂time_decay = gw · (−exp(time_decay)) = gw · w`。
  注释 `// multiply by w because of w -> -exp(w) in python forward()` 就是这句。
  配合 `model.py:210-211` 的 `torch.sum(gw, dim=0)`，得到每个参数一个标量梯度。
- **96–118 第二遍（逆向）**：用第一遍存下的 `q[i], r[i]` 反推
  `gk`（对 k 的梯度）和 `gv`（对 v 的梯度）——这两个是**逐时间步**的张量
  （形状 `(B,T,C)`），因为 k/v 本身就是逐时间步的。
  `aa, bb` 在这里扮演“未来梯度的累积器”，因此从 `T−1` 倒着走。
- **121–133**：把上面的实现包装成两个 C 函数供 `wkv_op.cpp` 调用：

  ```cuda
  122     dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
  123     assert(B * C % threadsPerBlock.x == 0);
  124     dim3 numBlocks(B * C / threadsPerBlock.x);
  125     kernel_forward<<<numBlocks, threadsPerBlock>>>(...);
  ```

  注释再次强调 `--maxrregcount 60` 与 `model.py:175` 的编译选项对应。

---

## D. `cuda/wkv_op.cpp` —— PyTorch 绑定

```cpp
3   void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y);
6   void forward(...)  { cuda_forward(B, T, C, w.data_ptr<float>(), ...); }
9   void backward(...) { cuda_backward(...); }
13  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
14      m.def("forward", &forward, "wkv forward");
15      m.def("backward", &backward, "wkv backward");
16  }
18  TORCH_LIBRARY(wkv, m) {
19      m.def("forward", forward);
20      m.def("backward", backward);
21  }
```

- 只是把 `.cu` 里的 C 函数暴露给 Python：`wkv_cuda.forward(B,T,C,w,u,k,v,y)`。
- 与 `model.py:190` / `model.py:208` 的调用一一对应。
- **13–21 同时注册了 `PYBIND11_MODULE` 和 `TORCH_LIBRARY(wkv, ...)`**，
  但对本项目只有 `PYBIND11_MODULE` 那条起作用（`load()` 拿到的是模块对象）。
  `TORCH_LIBRARY` 用的名字 `wkv` 与 bf16 版本相同，
  **若同时载入两个扩展会冲突**——本项目只载入 `wkv_op.cpp + wkv_cuda.cu`，不会出问题。

---

## E. bf16 版本（本仓库未使用）

`cuda/wkv_cuda_bf16.cu` 与 `cuda/wkv_op_bf16.cpp` 是 RWKV 官方的
**BFloat16 版本**：结构与 fp32 版逐行相同，只是 `k/v/y` 用 `at::BFloat16`，
而 `w` 仍保持 fp32（衰减因子对精度敏感），并在计算前 `float(...)` 提升到 fp32。

它们**没有被本项目引用**（`model.py` 和 `inference.py` 的 `load()` 都只列了非 bf16 的两个文件）。
论文 Sec. IV-C 也说明了原因：

> “The current implementation is far from optimized. … Few examples include model
> compilation for the target accelerator (e.g., via TensorRT) and **mixed-precision
> inference where parts of the model run integer-quantized or in floating-point half
> precision**.”

即“混合精度”被列为**未来工作**；当前实现是纯 FP32
（`compressor.py:97` 甚至硬编码了 `config.half = False`）。

**编译选项回顾（`model.py:175` / `inference.py:547`）**

| 选项 | 作用 |
|---|---|
| `-res-usage` | 打印寄存器/共享内存占用，便于调优 |
| `--maxrregcount 60` | 限制每线程 60 个寄存器，提升 occupancy（与核注释呼应） |
| `--use_fast_math` | 快速数学函数（`exp` 近似等）——**这是压缩端/解压端数值差异的主要来源** |
| `-O3` / `-Xptxas -O3` | 最高优化等级 |
| `--extra-device-vectorization` | 向量化访存（配合 `(B,T,C)` 连续布局） |
| `-DTmax=512` | 编译期固定栈数组长度 |

> `--use_fast_math` 让 CUDA 前向的 `exp` 与 CPU 手写版的 `torch.exp` 产生微小差异，
> 这正是论文 Sec. III-E 说的 “mismatch in the hardware-software stack between the
> compressor and the decompressor”，也是 `numerical_warning` 机制存在的根本原因。

---

## F. `launcher_train.sh`

```bash
4   rootdir="/home/valsesia/Downloads/linerwkv_git/linerwkv"     # 改成你的路径
6   model="linerwkv"
8   cd $model
10  save_dir="$rootdir/Results/$model/"
11  log_dir="$rootdir/log_dir/$model/"
13  python3 train.py --log_dir $log_dir --save_dir $save_dir
```

- 唯一需要改的是第 4 行 `rootdir`。
- `cd $model`（进入 `Code/linerwkv/`）是必须的：CUDA 扩展源码用的是相对路径
  `cuda/wkv_op.cpp`。
- `train.py:25` 会在 `save_dir` 下再拼一个时间戳子目录，
  所以最终产物路径是 `Results/linerwkv/<时间戳>/rwkv-final.pth`。

---

## G. `launcher_compress.sh` / `launcher_decompress.sh`

两个脚本结构完全对称，关键变量：

```bash
7   entropy_encoder_binary="$rootdir/Code/compressor"      # 预编译的 Golomb 编码器
9   model_file="$rootdir/Results/$model/$datestr/rwkv-final.pth"
10  config_file="$rootdir/log_dir/$model/$datestr/config.txt"
13  image="ENMAP01-____L2A-DT0000004981_...-SPECTRAL_IMAGE"
14  input_image="$rootdir/Dataset/$image.bsq"
15  N_rows=128 ; N_cols=128 ; N_bands=202
18  delta=0                     # 量化步长 = 2*delta+1
20  gpu=0 ; device="cuda"

29  residuals_file=..._residuals_$delta.bsq
30  compressed_file=..._compressed_$delta.cmp
31  side_info_file=..._side_info_$delta.h5
32  reconstructed_file=..._reconstructed_$delta.mat
33  numerical_warning_file=..._numerical_$delta.bin
34  side_info_file_mu=..._side_info_$delta.h5_mu.cmp
```

产物对照论文/代码：

| 文件 | 内容 | 论文/代码位置 |
|---|---|---|
| `*_residuals_<δ>.bsq` | ZigZag 后的 uint16 残差（中间文件） | `compressor.py:204/239` |
| `*_compressed_<δ>.cmp` | 熵编码后的残差（**主要码流**） | `compressor.py:245` |
| `*_numerical_<δ>.bin` | 数值告警位（算术编码） | `compressor.py:207-214` |
| `*_side_info_<δ>.h5` | 浮点 μ/σ（供解压端读） | `compressor.py:229-231` |
| `*_side_info_<δ>.h5_mu.cmp` / `_sigma.cmp` | μ/σ 的整数差分熵编码结果 | `compressor.py:217-227` |
| `*_reconstructed_<δ>.mat` | 重建图 `Iq·Q` | `compressor.py:234-235` |

### G.1 码率统计（脚本末尾）

```bash
ratecmp=`wc -c $compressed_file | awk '{print $1}'`
ratenum=`wc -c $numerical_warning_file | awk '{print $1}'`
rateside0=$(( 64*$N_bands ))
ratesidemu=`wc -c $side_info_file_mu | awk '{print $1}'`
ratesidesigma=`wc -c $side_info_file_sigma | awk '{print $1}'`
rate=$(( $ratecmp+$ratenum+$rateside  ))
echo "$rate bytes"
```

- 总字节数 = 残差码流 + 数值告警 + `64×N_bands` + μ 边信息 + σ 边信息。
- **`.h5` 不计入**（原因见 `05-压缩与解压.md` 第 6.1 节）。
- 换算成论文的 bpppc 公式：`bpppc = rate × 8 / (N_rows × N_cols × N_bands)`。
  以 128×128×202 = 3,309,312 个采样为例，**每 1 个字节 ≈ 0.0024 bpppc**。
- 这个统计方式是**近似**的（`64*N_bands` 那一项来源不明，且 μ/σ 的 `.h5` 未计），
  但论文报告的 0.02 bpppc 边信息开销与这里的量级一致。

### G.2 熵编码器二进制

`Code/compressor`（48 KB）和 `Code/decompressor`（48 KB）是**仓库预编译好的
Linux x86-64 可执行文件**，实现 CCSDS-123.0-B-2 的 sample-adaptive Golomb 编码。
README 原文：

> “Entropy encoder and decoder are provided as **binary files, compiled for Linux x86-64**.
> They are directly called by the compressor.py and decompressor.py files.”

调用参数（两处一致）：

```
--in_depth 16 --in_byte_ordering little --u_max 18 --y_0 1 --k 3 --y_star 6
```

| 参数 | 含义 |
|---|---|
| `--in_depth 16` | 输入是 16-bit 无符号（ZigZag 后的残差） |
| `--u_max 18` | 最大动态范围 18 位（对应 `compressor.py:178` 的 `clamp(±16383)`，即 15 位幅值 + 1 位符号） |
| `--y_0 1` / `--k 3` / `--y_star 6` | CCSDS-123 自适应 Golomb 的初始化/更新参数 |

> **可移植性提醒**：这两个二进制是 x86-64 Linux 版的，
> 在没有 x86-64 或没有 exec 权限的环境（例如某些容器、ARM 服务器）会直接失败。
> 论文 Sec. IV-B 也提到该 Golomb 编码器 “is not capable of producing rates below
> 1 bpppc”，所以要复现低于 1 bpppc 的结果必须换算术编码器。
