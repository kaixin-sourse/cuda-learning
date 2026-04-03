# CUDA C++ Week 5 详细学习资料

## 1. 本周目标

这一周你要学习 scan 这种非常基础、但在并行算法里极其常见的构件，并把它和 histogram 这类数据统计任务联系起来。

本周重点有 3 个：

1. 理解 inclusive scan 和 exclusive scan 的区别。
2. 理解 Blelloch scan 的上扫 / 下扫过程。
3. 完成一个 scan 示例和一个 histogram 应用示例，并和 CPU 结果对拍。

学完这一周，你至少应该能做到：

- 说清 `inclusive scan` 和 `exclusive scan` 的差别。
- 看懂 Blelloch scan 为什么要先 up-sweep 再 down-sweep。
- 跑通一个 scan 示例并验证结果。
- 跑通一个 histogram 示例并验证结果。

## 2. 什么是 scan

给定输入数组：

```text
[x0, x1, x2, x3, x4]
```

### inclusive scan

输出是：

```text
[x0, x0+x1, x0+x1+x2, x0+x1+x2+x3, x0+x1+x2+x3+x4]
```

### exclusive scan

输出是：

```text
[0, x0, x0+x1, x0+x1+x2, x0+x1+x2+x3]
```

两者区别很简单：

- inclusive scan：当前位置包含自己
- exclusive scan：当前位置不包含自己，通常第一个元素是 0

## 3. 为什么 scan 很重要

scan 不只是一个“前缀和题目”，它常常是很多并行算法里的基础模块。

比如：

- stream compaction
- radix sort
- histogram 后的前缀边界计算
- 稀疏数据重排

所以 Week 5 的重点不是“背一个前缀和代码”，而是开始理解 scan 为什么是并行算法的常用积木。

## 4. Blelloch scan 的核心思路

Blelloch exclusive scan 通常分两步：

## 4.1 up-sweep

先做树形归约，把局部和一层层往上合并，最后得到总和。

直观上像这样：

```text
[1, 2, 3, 4, 5, 6, 7, 8]
  -> 局部相加
  -> 更大粒度相加
  -> 根节点拿到总和
```

## 4.2 down-sweep

再把根节点清零，然后沿树结构向下传播前缀信息，最终得到 exclusive scan 输出。

你现在不需要一开始就背公式，先抓住这个直觉：

**up-sweep 负责“收集和”，down-sweep 负责“分发前缀”。**

## 5. 本周代码说明

## 5.1 `src/blelloch_scan_demo.cu`

这个程序演示的是一个单 block 版本的 Blelloch exclusive scan。

它的重点是让你看懂：

- 输入如何放进 shared memory
- up-sweep 每一层怎么合并
- down-sweep 每一层怎么交换和传播
- 为什么每一层都需要 `__syncthreads()`

注意：这个版本是教学版，专门用来讲算法流程，所以先固定在单 block 规模上，不追求一开始就支持超大数组的多 block scan。

## 5.2 `src/histogram_shared_atomic.cu`

这个程序实现了一个 histogram 示例：

- 输入是一批整数 bin id
- 每个 block 先在 shared memory 里维护局部 histogram
- 最后再用 atomic 把 block 内局部结果合并到 global histogram

这个例子的意义是：

- 让你看到 scan/histogram 这类“数据统计与重排任务”的并行写法
- 为下一周的 atomic 和更复杂并行模式做准备

## 6. 本周必须完成的练习

### 练习 1：手算一组 exclusive scan

比如输入：

```text
[3, 1, 4, 2]
```

请你手算 exclusive scan 输出，再对照程序理解。

### 练习 2：运行 `week5_blelloch_scan_demo`

要求：

- 确认 CPU 和 GPU scan 结果一致
- 看懂 up-sweep 和 down-sweep 两段循环的索引含义

### 练习 3：运行 `week5_histogram_shared_atomic`

要求：

- 确认 CPU 和 GPU histogram 结果一致
- 理解为什么先做 shared memory 局部 histogram，再合并到 global histogram

### 练习 4：修改数据分布

在 histogram 代码里修改随机数据分布，例如让某几个 bin 更容易出现，观察结果是否符合预期。

### 练习 5：修改 scan 输入

把 scan 的输入数组改成你自己设置的一组数据，确认输出是否正确。

## 7. 本周完成标准

- 能区分 inclusive scan 和 exclusive scan
- 能解释 Blelloch scan 的 up-sweep / down-sweep 思路
- 能跑通 scan 示例并完成 CPU 对拍
- 能跑通 histogram 示例并完成 CPU 对拍

## 8. 本周最该记住的一句话

**Scan 的价值不只是算前缀和，而是给并行数据重排提供“位置索引”。**
