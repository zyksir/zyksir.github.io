---
layout: post
title: "Triton 写的 Kernel，性能其实很能打"
subtitle: "This is a subtitle"
date: 2026-03-25
author: "Yikai"
header-img: "img/background/post-default-bg.jpg"
tags: []
---

前阵子在 GTC 上跟不少人聊，发现大家普遍有一个共识：想写出性能最优的 kernel，要么学 cute DSL，要么上 CUTLASS。Triton 更多被看作是一个"方便但性能一般"的选项。

但在实际开发中，我发现事情没那么绝对。至少在一些相对简单的 kernel 场景下，Triton 已经能提供非常出色的算子——甚至略优于手搓的 CUDA 或 cute DSL 的实现。本文我会通过几个具体的例子和实测数据来说明这一点。

## LayerNorm

第一个例子是两个最常见的 Normalize 算子：LayerNorm 和 RMSNorm（定义见 [PyTorch LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)、[PyTorch RMSNorm](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)）。这两个操作还常与后续算子融合（Fused）。

### RMSNorm

我们先来看不做任何 Fusion 的情况。

这方面，开源社区已经有不少高质量的实现可供参考：[quack](https://github.com/Dao-AILab/quack) 用 cute DSL 实现了一版，[Flash Attention](https://github.com/Dao-AILab/flash-attention) 用 Triton 实现了一版，[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 用 CUDA 实现了一版。我认为这些实现基本代表了当前开源中的最高水平。

不过有一点值得补充：Flash Attention 虽然已经用 Triton 写了 LayerNorm，但它的实现为了兼容 backward pass，牺牲了一些前向性能，并不是一个纯粹的前向 LayerNorm。换句话说，Flash Attention 中的版本并不代表 Triton 所能达到的性能上限。因此，我还是选择自己重新实现了一版（[实现代码](https://github.com/zyksir/CudaDiveDeep/blob/main/layernorm/my_layernorm.py) | [benchmark 脚本](https://github.com/zyksir/CudaDiveDeep/blob/main/layernorm/bench_layernorm.py)）。

以下分别是在 H100 和 B200 上的实测结果，实验结果用 `python bench_layernorm.py` 即可复现。输入 shape 为 `[12800, N]`，测试的是长序列（Diffusion 场景）下不同 hidden size 的吞吐（单位 GB/s）：

**H100**（镜像：`lmsysorg/sglang:dev`）

| N    | Flash Attention (Triton) | Ours (Triton) | FlashInfer (CUDA) | QuACK (cute DSL) |
|------|-------------------------:|--------------:|------------------:|-----------------:|
| 128  |                    73.78 |        783.47 |            305.77 |           797.19 |
| 3840 |                  2253.13 |       2750.58 |           2524.30 |          2734.40 |
| 4096 |                  2463.13 |       2781.84 |           2580.17 |          2756.49 |
| 6144 |                  2854.33 |       2865.27 |           2150.57 |          2849.27 |
| 8192 |                  2898.92 |       2911.14 |           2488.05 |          2899.73 |

**B200**（镜像：`lmsysorg/sglang:glm5-grace-blackwell`）

| N    | Flash Attention (Triton) | Ours (Triton) | FlashInfer (CUDA) | QuACK (cute DSL) |
|------|-------------------------:|--------------:|------------------:|-----------------:|
| 128  |                   210.20 |        797.81 |            320.19 |           799.94 |
| 3840 |                  5118.03 |       5485.12 |           3199.11 |          5049.26 |
| 4096 |                  5391.03 |       5529.05 |           3299.84 |          5154.21 |
| 6144 |                  5088.31 |       5482.43 |           2819.55 |          5486.63 |
| 8192 |                  5868.90 |       6114.58 |           3193.40 |          5847.96 |

可以看到，无论在 H100 还是 B200 上，我们用 Triton 实现的版本在各个 hidden size 下都与 QuACK（cute DSL）基本持平，甚至略优；而 FlashInfer（CUDA）和 Flash Attention（Triton）的实现则在部分场景下有明显差距。在 B200 上这一趋势更加明显。

### Fused RMSNorm + Scale & Shift

再来看 Diffusion 模型中常见的场景：y = (scale + 1) · norm(x) + shift。我在前面的 Triton kernel 基础上做了简单修改，与 SGLang 中 [cute DSL 的实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py)对比，性能略优。输入 shape 依然为 `[12800, N]`：

**H100**

| N    | PyTorch | Ours (Triton) | SGLang (cute DSL) |
|------|--------:|--------------:|------------------:|
| 1024 |  329.67 |       2490.82 |           2457.02 |
| 3072 |  362.30 |       2871.96 |           2848.10 |
| 4096 |  369.27 |       2948.85 |           2920.12 |
| 6144 |  375.90 |       2989.69 |           2913.16 |
| 8192 |  380.79 |       3037.77 |           3030.36 |

**B200**

| N    | PyTorch | Ours (Triton) | SGLang (cute DSL) |
|------|--------:|--------------:|------------------:|
| 1024 |  526.68 |       4459.59 |           4098.89 |
| 3072 |  545.95 |       5137.64 |           5060.57 |
| 4096 |  562.53 |       5847.35 |           5360.87 |
| 6144 |  579.48 |       4759.59 |           4753.59 |
| 8192 |  588.68 |       5693.42 |           5560.85 |


需要说明的是，我测试的场景还比较有限，而且与 cute DSL 版本的性能差异其实非常小，因此不能就此得出"Triton 一定优于 cute DSL"的结论。但这已经足以支撑本文的核心观点：**在部分 Memory Bound 的场景下，Triton 完全可以逼近甚至匹敌专家手写的 CUDA / cute DSL 代码。**

考虑到 cute DSL 的学习曲线相当陡峭，用 Triton 来实现高度定制化的 kernel 其实是一个非常务实的选择。整个 kernel 加 benchmark 脚本我花了大约两天时间，其中一大半花在了调研各种开源实现和横向对比上，真正写 Triton 代码的时间并不多。也欢迎大家在更多场景下测试验证。

### QK LayerNorm

这里额外对比一个特殊场景：**QK LayerNorm**。输入 shape 为 `[B, H, N]`，其中 N 通常为 128 或 64，H 为 num heads，weight shape 为 `[N]`。

这个场景的特殊之处在于，输入并不总是 contiguous 的。考虑 `qkv = self.to_qkv(x); q, k, v = torch.chunk(qkv, dim=-1)` 这种典型用法，只有 `[H, N]` 维度是连续的，`B` 维度则不是。而前面实现的 LayerNorm kernel 要求输入连续。此外，从前面的结果也可以看到，N = 128 时直接调用 FlashInfer 的效果其实很差。因此，SGLang 选择针对这个场景单独写一个 kernel，这个做法是合理的。我也针对这个场景专门写了一版（[实现代码](https://github.com/zyksir/CudaDiveDeep/blob/main/triton/qk_layernorm.py) | [benchmark 脚本](https://github.com/zyksir/CudaDiveDeep/blob/main/triton/bench_qk_layernorm.py)）。

下面分别对比三种实现方式：

1. **layer_norm (my)** — 直接复用通用的 LayerNorm kernel 来做 Q Norm。
2. **qk_norm (my)** — 专门为 Q Norm 场景写的 kernel，可以处理非连续的输入。
3. **fused_qk (my)** — 一个 kernel 同时处理 Q Norm 和 K Norm，函数签名与当前 SGLang 的版本一致，唯一区别是非 in-place 的。我个人认为在 Diffusion 场景下不差这点显存，in-place 并非必要。

实验结果用 `python bench_qk_layernorm.py` 即可复现。输入 shape 为 `[B, 24, 128]`（对应 Qwen-Image 模型的配置），分别测试 contiguous 和 non-contiguous 两种情况：

**H100 — Non-contiguous 输入（throughput, GB/s）**

| B     | PyTorch | layer_norm (my) | qk_norm (my) | fused_qk (my) | SGLang (cute DSL) |
|-------|--------:|----------------:|--------------:|---------------:|------------------:|
| 1024  | 3970.29 |         1097.14 |       3688.93 |        4268.97 |           3734.56 |
| 4096  | 2397.25 |         1093.51 |       2841.04 |        3089.91 |           3054.67 |
| 12800 | 2678.92 |         1042.60 |       2873.86 |        2970.73 |           2795.14 |

**H100 — Contiguous 输入（throughput, GB/s）**

| B     | PyTorch | layer_norm (my) | qk_norm (my) | fused_qk (my) | SGLang (cute DSL) |
|-------|--------:|----------------:|--------------:|---------------:|------------------:|
| 1024  | 3824.63 |         2891.10 |       3652.43 |        4248.65 |           3736.72 |
| 4096  | 2642.82 |         2595.48 |       2654.70 |        2801.24 |           2656.40 |
| 12800 | 2908.33 |         2868.07 |       2907.03 |        2954.84 |           2789.19 |

**B200 — Non-contiguous 输入（throughput, GB/s）**

| B     | PyTorch | layer_norm (my) | qk_norm (my) | fused_qk (my) | SGLang (cute DSL) |
|-------|--------:|----------------:|--------------:|---------------:|------------------:|
| 1024  | 4224.25 |         1384.62 |       4368.59 |        5568.24 |           4104.05 |
| 4096  | 3395.21 |         1466.98 |       5240.74 |        6323.57 |           4921.51 |
| 12800 | 3576.24 |         1447.01 |       5293.08 |        5884.29 |           4605.01 |

**B200 — Contiguous 输入（throughput, GB/s）**

| B     | PyTorch | layer_norm (my) | qk_norm (my) | fused_qk (my) | SGLang (cute DSL) |
|-------|--------:|----------------:|--------------:|---------------:|------------------:|
| 1024  | 4900.05 |         4222.25 |       4379.33 |        5586.45 |           4090.27 |
| 4096  | 6574.06 |         6083.06 |       4868.56 |        5766.84 |           4914.45 |
| 12800 | 6330.76 |         6233.05 |       5135.25 |        5605.78 |           4449.87 |

几个值得注意的点：通用 LayerNorm kernel 在 non-contiguous 场景下性能骤降（从 ~2800 跌至 ~1000 GB/s），这是因为引入了 `tensor.contiguous()` 这个额外的 copy 操作。而我们的 Fused Triton 版本（fused_qk）在两种输入模式下均表现得很不错。在 B200 上优势更为突出。

## 对更简单的 Kernel，请尝试 torch.compile

我还额外实现了一个针对 `y = w * x + b` 的 kernel（[实现代码](https://github.com/zyksir/CudaDiveDeep/blob/main/triton/mult_add.py) | [benchmark 脚本](https://github.com/zyksir/CudaDiveDeep/blob/main/triton/bench_mult_add.py)）。SGLang 中其实也用 Triton 实现了一个 scale_shift kernel，我是写完之后才发现的——不过意外地发现它的性能不如我的版本。但这些都不重要了，因为两者都不如 `torch.compile` 生成的 kernel。

按理说，这时候我应该把 `torch.compile` 生成的 Triton kernel 拿出来研究一下，看看它做了什么优化。但我毕竟不是专职的 kernel engineer，感兴趣的同学可以自行探索。我还额外试了 `torch.cat`，同样发现 `torch.compile` 能带来明显的加速。

**这其实解释了一个常见的困惑：为什么 `torch.compile(model)` 有时能加速，有时却没效果？**

原因在于：对于 LayerNorm 这类本身已经高度优化的 kernel，`torch.compile` 并不会带来提升，甚至偶尔会更慢；但对于 `y = w * x + b` 或 `torch.cat` 这类极其简单的操作，`torch.compile` 生成的 fused kernel 效果很好。明白了这一点，以后就没必要对整个 model 无脑 `torch.compile` 了——那样 CPU overhead 太大。更好的做法是：**只对这几个能从 torch compile 中受益的小 kernel 单独 compile**。

以下是 `y = w * x + b` 的实测结果（throughput, GB/s），输入 shape 为 `[1, S, 5120]`：

**Mult and Add: y = w * x + b**

**H100**

| S    | PyTorch | Ours (Triton) | SGLang (Triton) |
|------|--------:|--------------:|----------------:|
| 512  | 1436.44 |       1334.07 |         1254.17 |
| 4096 | 2158.38 |       2098.66 |         1836.13 |
| 6144 | 2219.55 |       2174.83 |         1881.81 |
| 8192 | 2252.52 |       2218.57 |         1900.80 |

**B200**

| S    | PyTorch | Ours (Triton) | SGLang (Triton) |
|------|--------:|--------------:|----------------:|
| 512  | 2188.33 |       1725.52 |         1956.76 |
| 4096 | 4419.27 |       3858.74 |         3801.70 |
| 6144 | 4688.39 |       4353.55 |         4125.89 |
| 8192 | 4795.70 |       4520.54 |         4252.69 |

我们的 Triton 版本在 H100 上优于 SGLang 的 Triton 实现——一方面是 SGLang 里的 block size 估计没有专门调优过，另一方面那个 kernel 的目标场景也没有我这么 specific。无论如何，两者都不如 `torch.compile` 生成的 fused kernel。

## Summary

最后需要强调的是，本文并不是在说"会 Triton 就够了"。对于 Attention 和 GEMM 这两个至关重要的算子，Triton 目前肯定是不够的——这部分内容会在后续文章中展开（没错，已经开始给自己挖坑了）。

本文也没有拉踩任何项目的意思。SGLang、Flash Attention、QuACK 都是非常优秀的开源库，而我这种专门针对一个极其具体的场景去手写 kernel 的做法，本身就带有一定的"不公平优势"，测试 case 也远称不上充分。

但这篇文章想反驳的是一个常见的偏见：**"Triton 就是不如 cute DSL 或 CUDA"**。至少从我的实验来看，用 Triton 写的这些 kernel 在 Diffusion 场景下具备一定的通用性，性能也不差——在多数测试场景下甚至略优于各个 baseline。
