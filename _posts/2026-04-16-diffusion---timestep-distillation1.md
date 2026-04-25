---
layout: post
title: "Diffusion   TimeStep Distillation(1) - 从 SDS 到 VSD 到 DMD"
subtitle: "This is a subtitle"
date: 2026-04-16
author: "Yikai"
header-img: "img/background/post-default-bg.jpg"
tags: []
---

让我在最前面做一个非常粗糙的总结就是，最开始的论文SDS为了数学上的优雅推导，加了一个很强的假设，所以效果不好；朱军老师的新论文修改了假设；但是修改了假设之后发现有一个Function推导不出来了，所以选择引入一个辅助模型来估计这个Function。而DMD基本就是这个朱军老师论文的衍生版本。但我觉得DMD对辅助模型的理解不对，所以错误的把这个模型称为了"Fake"模型，这个命名很容易让读者觉得这是在做对抗，而不是辅助，可能更合适的名字应该是"score estimation model"。如果觉得我理解的有问题欢迎讨论。

> 参考文献：[ProlificDreamer (Wang et al., 2023)](https://arxiv.org/abs/2305.16213)；[DMD2 - 微卷的大白的博客](https://zhuanlan.zhihu.com/p/1953596250491434750)

---

首先，DMD 系列的核心 Loss 公式是：

$$
D_{\text{KL}}(p_{\text{fake}} \| p_{\text{real}}) 
= \mathbb{E}_{x \sim p_{\text{fake}}} \left[ \log \frac{p_{\text{fake}}(x)}{p_{\text{real}}(x)} \right] 
= \mathbb{E}_{\substack{z \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\ x = G_\theta(z)}} \left[ -\left( \log p_{\text{real}}(x) - \log p_{\text{fake}}(x) \right) \right]
$$

初看这个公式，你一定会疑惑：一个蒸馏框架里应该只有 teacher 和 student 两个角色，我们要最小化的应该是 $D\_{\text{KL}}(p\_{\text{student}} \| p\_{\text{teacher}})$。这里 real 可以理解为 teacher，但是 student 去哪了？这个 fake 又是什么？

论文里的解释是："一边让图像更 real，一边让图像更不 fake"。说实话我对这个解释完全不理解。为了弄清楚这个概念，我追溯到了最早提出这一思想的论文——朱军老师的 **ProlificDreamer**。下面我们从 ProlificDreamer 的前身 SDS 出发, 梳理整个idea。

---

## 1. SDS 的核心思路

ProlificDreamer 的前身是 **Score Distillation Sampling (SDS)**，来自 DreamFusion。它的目标是从头训练一个 3D 模型，核心思路是：3D 模型渲染出的 2D 图片应该是"真实的"。什么叫真实？我们有一个预训练好的 diffusion 模型，我们认为它能生成的图像就是"真实的"——我们管这个 diffusion 模型叫 **real 模型**。

那怎么判断一张渲染图 $x_0$ 是否"真实"呢？SDS 的做法是：

1. 对渲染图 $x_0$ 加噪：$x\_t = \alpha\_t x\_0 + \sigma\_t \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$
2. 比较加噪后的分布 $q\_t^\theta(x\_t \| c)$ 和预训练 diffusion 模型定义的分布 $p\_t(x\_t \| y\_c)$ 之间的 KL 散度

SDS 的 Loss 可以写成：

$$
\mathcal{L}_{\text{SDS}}(\theta) = \mathbb{E}_{t, c} \left[ \frac{\sigma_t}{\alpha_t} \omega(t)\, D_{\text{KL}}\!\left( q_t^\theta(x_t | c) \;\|\; p_t(x_t | y_c) \right) \right]
$$

对应的梯度为：

$$
\nabla_\theta \mathcal{L}_{\text{SDS}} \approx \mathbb{E}_{t, \epsilon, c}\left[ \omega(t) \left( \epsilon_{\text{pretrain}}(x_t, t, y_c) - \epsilon \right) \frac{\partial g(\theta, c)}{\partial \theta} \right]
$$

其中 $g(\theta, c)$ 是渲染函数，代表生成一个3D object并投影到2D的过程，$\epsilon_{\text{pretrain}}$ 是预训练 diffusion 模型的噪声预测网络。我们不需要知道具体是怎么推导得到这个公式的，只需要知道在这个过程中引入了一个假设：3D 参数 $\theta$ 的分布就是一个 Dirac delta $\mu(\theta) = \delta(\theta - \theta^{(1)})$。在这个假设下：

$$
q_t^\mu(x_t | c, y) \approx \mathcal{N}(x_t \,|\, \alpha_t\, g(\theta^{(1)}, c),\; \sigma_t^2 I)
$$

对应的 score 就是简单的高斯噪声 $\epsilon$。这个假设的优点是Loss Function推导起来非常简单，缺点是容易导致模型在远离训练数据的地方失去泛化能力。这直接导致了 SDS 生成的 3D 结果过饱和、过于平滑、多样性差。

## 2. ProlificDreamer 与 Variational Distribution

朱军老师的核心idea是：**不要假设 $\theta$ 是一个点，而应该假设它服从一个分布 $\mu(\theta \| y)$，然后去优化这个分布**。按照我朴素的理解，就是之前用高斯分布来估计 $q\_t^\theta(x\_t \| c)$ 的方式太简单了。

这就引出了 **Variational Score Distillation (VSD)** 的优化目标：

$$
\mu^* = \arg\min_{\mu}\; \mathbb{E}_{t, c} \left[ \frac{\sigma_t}{\alpha_t} \omega(t)\, D_{\text{KL}}\!\left( q_t^\mu(x_t | c, y) \;\|\; p_t(x_t | y_c) \right) \right]
$$

其中 $q\_t^\mu(x\_t \| c, y) = \int q\_0^\mu(x\_0 \| c, y)\, p\_0^t(x\_t \| x\_0)\, dx\_0$ 是渲染图分布经过前向扩散后的边缘分布。有点像是 Gaussian mixture，对每一个 $ \theta $, x_0会贡献一个高斯分量。

论文中证明了，不同 timestep 上的 KL 散度都共享同一个全局最优解。


现在问题来了：我们要优化分布 $\mu$，但 $q_t^\mu$ 的 score（即 $\nabla\_{x\_t} \log q\_t^\mu$）是未知的——它不像 $p_t$ 那样有一个预训练好的网络可以直接算。

ProlificDreamer 的做法是：**引入一个辅助模型 $\epsilon_\phi$ 来估计 $q_t^\mu$ 的 score**：

$$
-\sigma_t \nabla_{x_t} \log q_t^\mu(x_t | c, y) \approx \epsilon_\phi(x_t, t, c, y)
$$

这个 $\epsilon_\phi$ 就是所谓的 **"fake 模型"**。它的作用是学习当前渲染图分布加噪后的 score function。在朱军老师的原始论文中，$\epsilon_\phi$ 用预训练 diffusion 模型的 LoRA 微调来实现。

作为一个infra人，看到这里就没有必要太在意数学上的细节了。让我来做一个非常粗糙的总结就是，之前SDS为了数学上的优雅，加了一个很强的假设，所以效果不好，所以朱军老师修改了假设；但是修改了假设之后发现有一个Function推导不出来了，所以选择引入一个辅助模型来估计这个Function。通过论文推导，VSD 的参数更新公式为：

$$
\nabla_\theta \mathcal{L}_{\text{VSD}} = \mathbb{E}_{t, \epsilon, c}\left[ \omega(t) \left( \epsilon_{\text{pretrain}}(x_t, t, y_c) - \epsilon_\phi(x_t, t, c, y) \right) \frac{\partial g(\theta, c)}{\partial \theta} \right]
$$

对比 SDS 的梯度公式，唯一的区别就是把 $\epsilon$（高斯噪声）替换成了 $\epsilon_\phi$（学习到的 score）。

用 score function 的语言来写：梯度正比于 real score 和 fake score 的差。
注意 score function 和噪声预测网络之间有标准换算关系 
$-\sigma\_t \nabla\_{x\_t} \log p\_t(x\_t) = \epsilon\_\theta(x\_t, t)$，
所以 VSD 梯度中的 $\epsilon\_{\text{pretrain}} - \epsilon\_\phi$ 
本质上就是 $-\sigma\_t (s\_{\text{real}} - s\_{\text{fake}})$。

---

## 3. 从 ProlificDreamer 到 DMD：fake 到底是什么？

现在我们回过头看 DMD 的公式，一切就清晰了。

**DMD 本质上是把 ProlificDreamer 的 VSD 思想应用到了蒸馏场景中。**

在 ProlificDreamer 中：
- **real 模型**：预训练的 diffusion 模型，定义了"真实图像"的分布 $p_t$
- **fake 模型**：辅助的 score 估计网络 $\epsilon_\phi$，学习 variational distribution $q_t^\mu$ 的 score
- **被优化的对象**：3D 模型参数 $\theta$

在 DMD 蒸馏中：
- **real 模型**：teacher diffusion 模型（预训练好的多步模型）
- **fake 模型**：一个辅助 score 网络，估计 student 生成图像分布的 score
- **被优化的对象**：student 模型 $G_\theta$

所以 **fake 并不是"假的"或"对立的"，它代表的是 student 生成图像真实分布的 score 估计**。我们优化的目标是让 student 生成的图像分布（由 fake 模型描述其 score）逼近 teacher 定义的真实图像分布（real 模型）。

更准确地说：
- $p_{\text{real}}$：teacher diffusion 模型定义的图像分布
- $p_{\text{fake}}$：student 生成器 $G_\theta$ 当前生成的图像分布（其 score 由 fake 模型估计）
- 最小化 $D\_{\text{KL}}(p\_{\text{fake}} \| p\_{\text{real}})$：让 student 分布逼近 teacher 分布

所以，fake 模型的训练过程也呼之欲出了：它本质上就是一个标准的 diffusion 训练过程，只不过训练数据不是真实图像，而是 student 模型当前生成的图像。

具体来说：

1. 从 student 模型采样生成一批图像：$z \sim \mathcal{N}(0, I)$，$x_0 = G_\theta(z)$
2. 对这些图像做前向扩散加噪：$x_t = \alpha_t x_0 + \sigma_t \epsilon$
3. 训练 $\epsilon_\phi$ 去预测噪声 $\epsilon$，使用标准的 denoising score matching loss：

$$
\mathcal{L}_{\text{fake}}(\phi) = \mathbb{E}_{t, \epsilon}\left[ \left\| \epsilon_\phi(x_t, t) - \epsilon \right\|^2 \right]
$$

这和训练一个普通的 diffusion 模型完全一样——唯一的区别是"数据集"换成了 student 的输出。这样训练出来的 $\epsilon_\phi$ 自然就学到了 $q_t^\mu$ 的 score，因为它就是在 student 生成的分布上做的 score matching。

实践中有两个要注意的点：

- **初始化**：$\epsilon_\phi$ 通常从预训练 diffusion 模型初始化，然后用 LoRA 微调。这样既利用了预训练的知识，又保持了训练效率。
- **持续更新**：随着 student $G_\theta$ 不断优化，它生成的图像分布 $q_0^\mu$ 也在变化，所以 fake 模型需要持续跟着更新。在训练循环中，fake 模型和 student 模型是交替优化的。

这其实和 GAN 的训练范式很像：discriminator（fake 模型）需要跟上 generator（student）的变化，两者交替更新。只不过这里的 "discriminator" 不是在做真假判别，而是在做 score estimation。

---

## 4. 工程小技巧：已知梯度如何用 PyTorch backward？

在推导完理论后，我们面临一个工程问题。通常我们是有一个 Loss function $\mathcal{L}(\theta)$，然后调用 `loss.backward()` 让 PyTorch 自动帮我们求梯度。但在 VSD/DMD 中，我们**直接推导出了梯度公式**，而不是 Loss 本身。

怎么让 PyTorch 的 backward 机制帮我们干这件事？这里用了一个巧妙的 **stop gradient** 技巧：

在已知梯度 $\text{grad}$ 的情况下，构造如下伪 Loss：

$$
\mathcal{L}_{\text{pseudo}} = \frac{1}{2} \left\| x - \text{stopgrad}(x - \text{grad}) \right\|^2
$$

我们来验证：对 $x$ 求导：

$$
\frac{\partial \mathcal{L}_{\text{pseudo}}}{\partial x} = x - \text{stopgrad}(x - \text{grad}) = x - (x - \text{grad}) = \text{grad}
$$

因为 `stopgrad` 把括号里的东西当常数处理，所以求导后恰好等于我们想要的梯度。这样我们就能无缝接入 PyTorch 的自动微分框架了。

在代码中大概长这样：

```python
# grad 是我们通过 VSD/DMD 公式计算出来的梯度
# x 是 student 模型的输出
with torch.no_grad():
    grad = calculate_dmd2_grad()
    target = (x - grad).detach()  # stop gradient
loss = 0.5 * F.mse_loss(x, target)
loss.backward()
```

---

## 小结

理解了这个框架之后，下一步我们就可以去看 FastGen 的具体代码实现了。本来想一口气讲完的但是实在写不动了。未完待续...