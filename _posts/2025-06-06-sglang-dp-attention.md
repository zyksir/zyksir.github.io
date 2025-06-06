---
layout: post
title: "sglang dp attention"
subtitle: "parallelism in sglang"
date: 2025-06-06
author: "Yikai"
header-img: "img/background/post-default-bg.jpg"
tags: []
---

## Background

在目前训练/推理的各个场景里，Attention层的并行策略和MoE层的并行策略是不一样的。这个已经逐渐成为了一种趋势，因为MoE层里有EP这个额外的并行策略考虑。目前Sglang的代码演进的很快，在理解一些high level的概念之后再去看代码，会发现都是一脉相承，会更顺畅的理解代码。

本文会重点关注Attention层的并行策略，也就是Sglang里`dp attention`这个概念。本文会试图回答以下问题: 什么是 dp attention，为什么dp attention能带来性能提升，未来有什么演进的方向。欢迎大家多多交流。

在正式介绍之前，引用SGLang v0.4 blog里的一张图，介绍一些代码里的一些参数以及与之对应的概念: 

![img](https://lmsys.org/images/blog/sglang_v0_4/dp_attention.svg)

- `tp_size`: 这个其实说的是World-Size。上图可以理解成`tp_size=4`

- `dp_size`: 等价于`attention_dp_size`，代表attention层会同时处理多少个micro-batch。上图中`dp_size=4`

- `attn_tp_size`: 其值等于`tp_size//dp_size`，上图中`attn_tp_size=1`

## DP Attention

Attention层的输入是一个完整Feature
Attention层的输出是不完整的Feature，需要进行Atten TP Group之间的AllReduce才能得到完整有意义的Feature
那么下面这段代码其实是有点冗余的，正确且精简的代码如下所示；其中 attn_tp_group等价于tp_mod_dp_group，只是tp_mod_dp这个词对于我这个熟悉预训练的人来说非常违和，下面是我的一点碎碎念
划分拓扑时，attn_tp是优先于dp的。这个优先级是算是我自己想的一个概念，但是在megatron里其实就有体现了。换言之，如果有4个节点，TP=4，DP=2，attn_tp_group 会被分到[0, 1]，dp会被分到[0, 2]。[0, 1]意味着gpu在物理上更近，也意味着通信效率严格大于等于[0, 2]这个group。
def forward_normal(...):
  hidden_states = self_attn()
  if attn_tp_group.size() > 1: # equal to tp_mod_dp_group
    all_reduce(hidden_states, attn_tp_group)
  hidden_states, residual = post_attention_layernorm(hidden_states, residual)
  if attn_dp_group.size() > 1:
    all_gather(hidden_states_for_moe, hidden_states, attn_dp_group)


Decoder Layer Normal Forward
# DeepseekV2DecoderLayer
def forward_normal(...):
  hidden_states, residual = input_layernorm(hidden_states, residual)
  hidden_states = self_attn(hidden_states)
  if attn_tp_size > 1:
    hidden_states = all_reduce(hidden_states, attn_tp_group)
  hidden_states, residual = post_attention_layernorm(hidden_states, residual)
  if attn_dp_size > 1:
    hidden_states = all_gather(hidden_states, attn_dp_group)
  hidden_states = self.mlp(hidden_states)
  hidden_states = all_reduce(hidden_states, tp_group)
  hidden_states = slice(hidden_states, attn_dp_rank)
  # if attn_dp_size > 1:
  #   hidden_states = reduce_scatter(hidden_states, attn_dp_group)
  # if attn_tp_size > 1:
  #   hidden_states = all_reduce(hidden_states, attn_tp_group)
值得一提的是，AllGather和ReduceScatter是一个对偶操作，二者必然是成对出现的，不然Shape就对不上了。最后两行的两个通信可以被合并成一个通信算子+一个Copy，这样是Sglang里的实现，all_reduce(x, tp_group) + slice(x, attn_dp_rank)，这个以增加通信量为代价，减少了一次通信的调用，可能可以降低了Latency。(其实我认为只是Sglang开发者没考虑的这么细，被我注释掉的代码可以平替最后两行，我认为是更优的选择)
在attn_tp_size=1的场景下，把reduce-scatter拆成all-gather + slice的操作是低效的；slice这个算子本身耗时在2us左右，主要的效率损失预估来自于all-reduce相对于reduce-scatter引入的额外通信overhead
这里的AllGather其实就是AlltoAll实现的一种替代。AllGather + Permutation的Fusion就变成了DeepEP的Dispatch算子；而Unpermutation + Slice的Fusion则是DeepEP里的Combine算子。这个MLP的实现会放在FusedMoE和EPMoE的实现里讲。目前只需要知道，这两个Class的实现要求拿到所有Batch的数据，换言之就是需要上面的这个AllGather操作。
Decoder Layer DeepEP Forward
# DeepseekV2DecoderLayer
def forward_deepep(...):
  hidden_states, residual = input_layernorm(hidden_states, residual)
  if attn_tp_size > 1:
    hidden_states = all_gather(hidden_states, attn_tp_group)
  hidden_states = self_attn(hidden_states)
  if attn_tp_size > 1:
    hidden_states = reduce_scatter(hidden_states, attn_tp_group)
  hidden_states, residual = post_attention_layernorm(hidden_states, residual)
  hidden_states = self.mlp(hidden_states)
DeepEP的实现的一个最大的特点是，可以省去那个AllGather操作了，因为在MLP Layer内部会做Dispatch和Combine，这里就包含了通信。但是可以看到attn_tp_group里的通信算子从AllReduce变成了AllGather + ReduceScatter，这是为什么呢？
这部分不理解也没啥影响，在Decode阶段设置attn_tp_size=1是合理且常见的
在代码里，deepep生效时，input_is_scattered必然是True；否则input_is_scattered必然是False。input_is_scattered意味着一个完整的Sequence可能被切分放到了不同AttnTPRank里，而在计算Attention的时候我们需要完整的Sequence，因此我们需要AllGather+ReduceScatter
input_is_scattered必然是True，是因为DeepEP要求EP Group内每一个GPU上的数据都是不一样的。这样做Dispatch和Combine才有意义。而要做到这一点，这里引入了Sequence Parallelism + TP的方式，来吧一个Batch在TP内部也根据Sequence维度切分。
FusedMoE - TP
如果不开启--enable-ep-moe，那么会以TP的方式切分不同Expert上的专家。也就是每个GPU上都拥有所有专家的 1/tp_size。这里不做细述。
EPMoE
如果开启了--enable-ep-moe，那么tp_size实际上就是ep_size了。这里会选择将不同的Expert放于不同的GPU上。
因为Cuda Graph要求每个Kernel的输入/输出shape是固定的，因此这里permute之后的输出其实是num_tokens * topk，每个GPU只有其中某一段是有数据的，其他全为0。groupgemm在编写的时候也需要考虑到这种情况。S