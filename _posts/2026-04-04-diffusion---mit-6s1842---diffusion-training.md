---
layout: post
title: "Diffusion   MIT 6.S184课程笔记(2)   Diffusion Training"
subtitle: "This is a subtitle"
date: 2026-04-04
author: "Yikai"
header-img: "img/background/post-default-bg.jpg"
tags: []
---

## 序言

在看 diffusion 的过程中，我觉得最大的难点就在于理解它是怎么 training 的。这里面有一点我觉得非常优雅：背后的数学理论和工程实现是很完美地解耦开的——scientist 给出 idea 和公式推导，engineer 负责高效实现。这种解耦也导致了一种错觉：training 的理论部分对工程师"似乎"不那么重要。但当我开始尝试实现一些 distillation 相关的框架时，才发现如果对 training 没有一个扎实的理解，很多工程决策根本无从下手。

最后说一下 diffusion training 和 LLM training 的区别。对于大语言模型，模型规模动辄上百 B，工程上的核心挑战是如何用各种并行策略把模型放下并提升训练效率。但目前主流的图像/视频生成模型的参数量相对小很多，因此 training 这边的核心难点就集中在一件事上：**loss function 到底是什么**。至于数据质量对最终模型效果的影响，那当然是更重要的话题，但这超出了本文的讨论范围。

本文延续上一篇的风格，偏工程视角，会尽可能减少纯数学推导，重点在于把推导的脉络和直觉讲清楚。

## 基本原理

### 目标回顾

在上一篇博客中，我们知道了：只要能得到 ODE/SDE 的向量场 $u_t^\theta$，就可以通过 Euler / Euler-Maruyama 等数值方法从噪声生成图像。那么 training 的核心目标就很明确了——找到一个 **training target** $u_t^{\text{target}}(x)$，然后通过最小化 MSE 来训练神经网络：

$$\mathcal{L}(\theta) = \left\| u_t^\theta(x) - u_t^{\text{target}}(x) \right\|^2$$

现在的问题是：$u_t^{\text{target}}(x)$ 长什么样？我们唯一已知的信息只有训练数据 $z \sim p_{data}$。接下来的推导会一步一步引入 assumption，逐渐缩小范围，最终得到一个可以计算的公式。

**一个我之前的困惑:** 居然加噪声的过程（$x\_t = \alpha\_t z + \beta\_t \epsilon$）我们完全知道，那为什么不直接让模型去预测噪声 $\epsilon$，搞一个 MSE 就完了？为什么推理又使用 $x\_{t-h} = x\_t - h \cdot u\_t^\theta(x\_t)$ 这也看起来和前向diffusion毫无关系的迭代公式。

这里有一个很关键的点：**"怎么推导出 training loss function"和"怎么做 inference"，其实是解耦的**。把这两件事串联起来的桥梁，正是 ODE/SDE 的数学框架。Training 阶段，我们推导出 $u_t^{\text{target}}(x)$ 的公式；inference 阶段，我们可以自由选择用 Euler、Heun 或者其他任何数值方法来求解。Training target 的形式是由数学理论决定的，而不是由推理时的迭代方式反推出来的。

所以，那种"加噪怎么加的，我就反过来让模型预测噪声"的朴素想法，并没有严格的理论保证。虽然我们最终会发现 DDPM 的 loss 确实长得像在预测噪声（后面会推导），但这是从 score matching 框架严格推导出来的结论，而不是起点。

### 数学概念

**Conditional Probability Path**

第一步是定义**条件概率路径（conditional probability path）**。对于每个数据点 $z \in \mathbb{R}^d$，我们定义一族分布 $p_t(\cdot \mid z)$，满足：

$$p_0(\cdot \mid z) = p_{\text{init}}, \quad p_1(\cdot \mid z) = \delta_z$$

含义很直观：$t=0$ 时是纯噪声分布，$t=1$ 时坍缩到单个数据点 $z$。这条路径描述的是"噪声如何逐渐变成一个特定的数据样本"。

由此可以得到**边缘概率路径（marginal probability path）**：

$$p_t(x) = \int p_t(x \mid z)\, p_{\text{data}}(z)\,\mathrm{d}z$$

也就是对所有数据点 $z$ 的条件路径取平均。

**高斯条件概率路径**

一个最典型的 assumption 是**高斯条件概率路径（Gaussian conditional probability path）**：

$$p_t(\cdot \mid z) = \mathcal{N}(\alpha_t z,\, \beta_t^2 I_d)$$

从中采样就是 $x\_t = \alpha\_t z + \beta\_t \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I_d)$。这里 $\alpha_t$ 和 $\beta_t$ 满足边界条件 $\alpha\_0 = \beta\_1 = 0$，$\alpha\_1 = \beta\_0 = 1$。这也正是 DDPM 中使用的 assumption。

**Marginalization Trick**

现在引入一个关键定理（我们跳过证明，只关注它说了什么）：

$$u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x \mid z)\, \frac{p_t(x \mid z)\, p_{\text{data}}(z)}{p_t(x)} \,\mathrm{d}z$$

这个定理建立了两个向量场之间的关系：$u_t^{\text{target}}(x)$ 是能把 $p_{init}$ 转化为 $p_{data}$ 的向量场（我们最终想要的），而 $u_t^{\text{target}}(x \mid z)$ 是能把 $p_{init}$ 转化为**单个数据点** $\delta_z$ 的向量场（相对容易求解的）。

这意味着我们的策略应该是：**先搞定 $u_t^{\text{target}}(x \mid z)$ 的表达式，再通过上面的公式得到 $u_t^{\text{target}}(x)$**。

### 条件向量场的显式公式

对于 Flow Model（ODE）+ 高斯条件概率路径，令 $\dot{\alpha}\_t = \partial\_t \alpha\_t$，$\dot{\beta}\_t = \partial\_t \beta\_t$，可以证明：

$$u_t^{\text{target}}(x \mid z) = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\,\alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t}\, x$$

（证明的起手式是"注意到" $\psi\_t^{\text{target}}(x \mid z) = \alpha\_t z + \beta\_t x$ 恰好满足高斯条件概率路径的定义，然后代入 ODE 求导即可。这里的"注意到"倒不算反人类——$\alpha\_t z + \beta\_t x$ 本质上就是高斯分布的 reparameterization trick，是一个很自然的构造。）

### Diffusion Model 的 SDE 形式

对于 Diffusion Model（SDE），如果 $X\_0 \sim p\_{\text{init}}$ 且具有如下形式：

$$\mathrm{d}X_t = \left[u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2}\,\nabla \log p_t(X_t)\right]\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$$

则 $X\_t \sim p\_t$，$0 \leq t \leq 1$。
- 这里的 $u_t^{\text{target}}(x)$ 仍然可以用 $u_t^{\text{target}}(x \mid z)$ 来表示
- 新出现的 **score function** $\nabla \log p_t(x)$ 同样可以从条件量推导得到：(证明略)

$$\nabla \log p_t(x) = \frac{\int \nabla p_t(x \mid z)\, p_{\text{data}}(z)\,\mathrm{d}z}{p_t(x)}$$

对于高斯条件概率路径，条件 score 有一个非常简洁的形式：

$$\nabla \log p_t(x \mid z) = -\frac{x - \alpha_t z}{\beta_t^2}$$

**Langevin Dynamics：一个重要的特例**

如果我们额外要求分布是静态的，即 $p_t = p$（不随时间变化），那么 $\partial\_t p\_t(x) = 0$，进一步可以得到 $u_t^{\text{target}} = 0$(这里有一个很复杂又不是很严谨的证明，总之我站在工程师的角度理解为怎么简单怎么来了，我们找到了一个最简单优雅的解就行)。SDE 简化为：

$$\mathrm{d}X_t = \frac{\sigma_t^2}{2}\,\nabla \log p(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$$

这就是**朗之万动力学（Langevin dynamics）**。它有一个漂亮的性质：如果 $X_0 \sim p$，则 $X_t \sim p$（$t \geq 0$），即 $p$ 是该 SDE 的平稳分布。在实际应用中，即使起点 $X_0 \sim p' \neq p$，只要满足一些温和的条件，分布也会逐渐收敛到平稳分布（$p'_t \to p$）。这个性质会在后续的 Consistency Model 中扮演核心角色。

## Flow Matching Loss Function 推导

训练一个 Flow Model，最自然的 loss function 是**Flow Matching Loss**：

$$\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif},\, x \sim p_t} \left[\| u_t^\theta(x) - u_t^{\text{target}}(x) \|^2\right]$$

问题在于 $u_t^{\text{target}}(x)$ 涉及对所有数据点的积分，无法直接计算。但我们可以转而优化**Conditional Flow Matching Loss**：

$$\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)} \left[\| u_t^\theta(x) - u_t^{\text{target}}(x \mid z) \|^2\right]$$

关键结论是：$\mathcal{L}\_{\mathrm{FM}}(\theta) = \mathcal{L}\_{\mathrm{CFM}}(\theta) + C$，其中 $C$ 是与 $\theta$ 无关的常数(证明略)。因此优化 $\mathcal{L}_{\mathrm{CFM}}$ 就等价于优化 $\mathcal{L}_{\mathrm{FM}}$。

这就是 Flow Matching 的精髓：我们看起来在显式地对 tractable 的条件向量场做回归，其实是隐式地优化 intractable 的边缘向量场。根据 $\mathcal{L}_{\mathrm{CFM}}$ 训练 $u_t^\theta(x)$ 的整个过程，就被称为 **Flow Matching**。

**高斯条件路径下的具体形式**

将已知的 $u_t^{\text{target}}(x \mid z)$ 代入 $\mathcal{L}_{\mathrm{CFM}}$，并用 $x = \alpha\_t z + \beta\_t \epsilon$ 做变量替换：

$$\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0, I_d)} \left[\| u_t^\theta(\alpha_t z + \beta_t \epsilon) - (\dot{\alpha}_t z + \dot{\beta}_t \epsilon) \|^2\right]$$

这个公式已经完全可以计算了：采样一个数据点 $z$、一个时间 $t$、一个噪声 $\epsilon$，加噪得到 $x_t$，让模型预测向量场，和 target $(\dot{\alpha}\_t z + \dot{\beta}\_t \epsilon)$ 做 MSE——就是这么简单。

如果进一步令 $\alpha_t = t$，$\beta_t = 1 - t$（即 CondOT probability path），则 $\dot{\alpha}_t = 1$，$\dot{\beta}_t = -1$，公式化简为：

$$\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t,\, z,\, \epsilon} \left[\| u_t^\theta(tz + (1-t)\epsilon) - (z - \epsilon) \|^2\right]$$

## Score Matching Loss Function 推导

要训练 Diffusion Model（SDE），除了向量场 $u_t^\theta$ 之外，还需要学习 score function。为此引入一个 **score network** $s_t^\theta(x)$ 来近似 $\nabla \log p_t(x)$，对应的 **Score Matching Loss** 为：

$$\mathcal{L}_{\mathrm{SM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)} \left[\| s_t^\theta(x) - \nabla \log p_t(x) \|^2\right]$$

和 Flow Matching 的思路完全一样，边缘 score $\nabla \log p_t(x)$ 是 intractable 的，但条件 score $\nabla \log p_t(x \mid z)$ 是 tractable 的。定义 **Conditional Score Matching Loss**：

$$\mathcal{L}_{\mathrm{CSM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)} \left[\| s_t^\theta(x) - \nabla \log p_t(x \mid z) \|^2\right]$$

同样可以证明 $\mathcal{L}\_{\mathrm{SM}}(\theta) = \mathcal{L}\_{\mathrm{CSM}}(\theta) + C$。

### DDPM Loss：Denoising Score Matching

对于高斯条件概率路径，将 $\nabla \log p\_t(x \mid z) = -\frac{x - \alpha\_t z}{\beta\_t^2}$ 代入，并做一个重参数化 $-\beta\_t s\_t^\theta(x) = \epsilon\_t^\theta(x)$（即把 score network 转化为 **noise predictor**），loss 变成：

$$\mathcal{L}_{\mathrm{DDPM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0, I_d)} \left[\| \epsilon_t^\theta(\alpha_t z + \beta_t \epsilon) - \epsilon \|^2\right]$$

到这里，开头那个疑问终于有了答案：**$\mathcal{L}_{\mathrm{DDPM}}$ 的确就是在预测噪声**。但这是从 score matching 的理论框架严格推导出来的。"预测噪声"本质上等价于"学习 score function"，而 score function 正是驱动 SDE 的核心。

### Score Network 与 Vector Field 的互转

对于高斯概率路径，一个非常有趣的性质是 **score network 和 vector field network 可以互相转换**：

$$u_t^\theta(x) = \left(\beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t\right) s_t^\theta(x) + \frac{\dot{\alpha}_t}{\alpha_t}\, x$$

这意味着我们不需要分别训练 $s_t^\theta$ 和 $u_t^\theta$——训练其中任何一个，就可以通过上式得到另一个。也就是说，**用 flow matching 训练出来的模型，也可以跑 SDE sampling；用 score matching 训练出来的模型，也可以跑 ODE sampling**。

将转换公式代入 SDE，得到完整的采样方程：

$$X_0 \sim p_{\text{init}}, \quad \mathrm{d}X_t = \left[\left(\beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t + \frac{\sigma_t^2}{2}\right) s_t^\theta(x) + \frac{\dot{\alpha}_t}{\alpha_t}\, x\right]\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$$

其中 $\sigma_t$ 是一个可以自由选择的扩散系数，$\sigma_t = 0$ 时退化为 ODE（确定性采样），$\sigma_t > 0$ 时为 SDE（随机采样）。理论上完美训练后两者等价，但实践中最优的 $\sigma_t$ 需要通过实验来确定。

### 重新回顾 DDPM

前面我们从 flow matching 和 score matching 的角度推导出了 $\mathcal{L}_{\mathrm{DDPM}}$。但DDPM（[Ho et al., 2020](https://arxiv.org/abs/2006.11239)）论文本身并不是这样推导的。它走的是一条完全不同的路，通过离散时间的马尔可夫链 + ELBO得到了同样的结论。MIT 6.S184 讲义的 Section 4.3 也提到，这两种推导方式在连续时间极限下是等价的。这里简要记录 DDPM 原始视角下的推导结论，详细推导可参考[《深入浅出扩散模型系列：基石 DDPM》](https://zhuanlan.zhihu.com/p/650394311)。

**前向扩散过程（Forward / Diffusion Process）。** 给定原始数据 $x\_0 \sim p\_{\text{data}}$，DDPM 分 $T$ 步逐渐混入高斯噪声，得到一条离散序列 $x\_0, x\_1, \ldots, x\_T$。每一步的转移分布为：

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(\sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\right)$$

即 $x\_t = \sqrt{1 - \beta\_t}\, x\_{t-1} + \sqrt{\beta\_t}\, \epsilon\_t$，$\epsilon_t \sim \mathcal{N}(0, I)$。当 $T$ 足够大时，$x_T$ 趋于标准正态分布。

利用高斯分布的可加性（closure property），可以直接将 $x_t$ 表示为 $x_0$ 的函数，跳过所有中间步：

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \bar{\epsilon}_t, \quad \bar{\epsilon}_t \sim \mathcal{N}(0, I), \quad \bar{\alpha}_t = \prod_{i=1}^t (1 - \beta_i)$$

这个"一步到位"的公式在训练时极为重要——不需要真的跑 $t$ 步前向过程，直接用 reparameterization trick 从 $x_0$ 采样到任意时刻 $x_t$。对比前面连续时间框架下的 $x\_t = \alpha\_t z + \beta\_t \epsilon$，可以看到两者的形式完全一致（只是离散 vs. 连续的记号差异）。

**反向去噪过程（Reverse / Denoise Process）** 利用贝叶斯公式，后验分布可以写成：

$$q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1})\, q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$

等式右边三项都是高斯分布，因此左边也是高斯分布，其均值可以计算为：

$$\mu_t = \frac{1}{\sqrt{1-\beta_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon\right)$$

基于这个后验均值，DDPM 构造了去噪的迭代公式：

$$x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \boldsymbol{\epsilon}_\theta(x_t, t)\right) + \sigma_t \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, I)$$

其中 $\boldsymbol{\epsilon}\_\theta(x\_t, t)$ 就是神经网络预测的噪声，$\sigma_t$ 是一个预设的方差系数。这个公式对应了[DDPM Scheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L539C57-L539C77)

**优化目标。** DDPM 的出发点是最大化数据的对数似然：

$$\max_\theta \; \mathbb{E}_{x_0 \sim p_{\text{data}}} \left[\log p_\theta(x_0)\right]$$

通过变分推断（ELBO）可以将其转化为最小化一系列 KL 散度。由于涉及的分布都是高斯的，KL 散度最终归结为均值之间的 MSE，得到：

$$\min_\theta \; \sum_{t=1}^T \lambda(t)\, \mathbb{E}_{x_0 \sim p_{\text{data}},\, \bar{\epsilon}_t \sim \mathcal{N}(0, I)} \left\|\boldsymbol{\epsilon}_\theta(x_t, t) - \bar{\epsilon}_t\right\|^2$$

其中 $\lambda(t)$ 是每个时间步的权重系数。DDPM 论文中实验发现直接令 $\lambda(t) = 1$（即忽略理论推导中的权重）效果最好。

**和连续时间框架的联系。** 对比前面推导的 $\mathcal{L}\_{\mathrm{DDPM}}(\theta) = \mathbb{E}\_{t, z, \epsilon}\left[\|\epsilon\_t^\theta(\alpha\_t z + \beta\_t \epsilon) - \epsilon\|^2\right]$，两者本质上是同一个东西——前者是离散时间的求和，后者是连续时间的期望；前者通过 ELBO 推导，后者通过 score matching 推导。正如讲义 Section 4.3 所述，离散版本可以看作连续 SDE 的近似，而在连续极限下 ELBO 变成等式而非下界，数学上更加"干净"。殊途同归。