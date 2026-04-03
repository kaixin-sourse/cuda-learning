# CUDA C++ Week 4 详细学习资料

## 1. 本周目标

这一周你要开始接触 CUDA 里最经典的一类并行算法：reduction。

本周重点有 4 个：

1. 理解 sum reduction 为什么适合做树形并行。
2. 理解 branch divergence 为什么会浪费执行效率。
3. 理解 coalesced memory access 为什么重要。
4. 初步理解 shared memory bank conflict 是什么，以及为什么要开始注意 shared memory 访问模式。

学完这一周，你至少应该能做到：

- 写出一个正确的 sum reduction。
- 看懂“朴素版 -> 改进版 -> 进一步改进版”的优化方向。
- 能用自己的话解释 divergence、coalesced access、bank conflict。
- 能做一次“不同 reduction 版本计时对比”。

## 2. Week 4 和前面几周的关系

Week 1-3 你已经学了：

- kernel、thread/block/grid
- host/device 数据流
- shared memory 和 `__syncthreads()`
- 基础 matmul tile 化

Week 4 的重点是把这些基础搬到一个更典型的并行模式上：**把很多元素规约成一个值**。

比如 sum reduction 的目标就是：

```text
y = x[0] + x[1] + x[2] + ... + x[n-1]
```

在 CPU 上你可以直接 for 循环累加；在 GPU 上，如果还是一个线程顺序加完，那就没有发挥并行优势。所以 reduction 的关键是“分组并行累加 + 逐步合并”。

## 3. 本周核心概念

## 3.1 树形 reduction

最直观的并行 reduction 思路是：

1. 先让很多线程各自负责一部分输入
2. block 内先做局部求和
3. 得到每个 block 的 partial sum
4. 最后再把 partial sums 合并成总和

block 内局部求和通常可以做成树形结构：

```text
step 1: x0+x1, x2+x3, x4+x5, x6+x7
step 2: (x0+x1)+(x2+x3), (x4+x5)+(x6+x7)
step 3: 全部合并
```

这种结构的意义是：不是一个线程做完所有加法，而是很多线程分层协作。

## 3.2 branch divergence

GPU 线程通常以 warp 为单位执行。一个 warp 里如果不同线程走不同分支，就可能产生分支发散。

比如某些 reduction 写法里有这种判断：

```cpp
if (threadIdx.x % (2 * stride) == 0) {
    ...
}
```

这个条件会让一个 warp 内部分线程执行、部分线程不执行，容易带来 divergence，而且 `%` 本身也不是一个很便宜的操作。

所以 reduction 优化中，常见方向之一就是减少这类不必要的分支和取模判断。

## 3.3 coalesced memory access

如果一个 warp 内相邻线程访问 global memory 中相邻地址，通常更容易形成合并访存，这对带宽利用更有利。

如果访问模式很散、跨度很奇怪，就更容易浪费带宽。

在 reduction 里，优化时要开始关注：

- 每个线程从 global memory 读哪些位置
- 这些读取是不是尽量连续
- 是否能让一个线程一次读多个元素，减少 kernel 层级和访存开销

## 3.4 bank conflict

shared memory 虽然快，但它也不是“怎么访问都一样快”。

shared memory 被划分成多个 bank，如果同一时刻多个线程访问模式不合适，可能产生 bank conflict，导致访问被串行化。

Week 4 你不需要把所有 bank conflict 模型细节一次背完，但要建立一个意识：

**shared memory 不是只要用了就一定快，访问模式也很重要。**

## 4. 本周代码说明

## `src/reduction_compare.cu`

这个程序实现了 3 个版本的 sum reduction：

1. `reductionInterleavedKernel`
   - 更朴素
   - 用 `%` 和 interleaved addressing
   - 更容易出现 divergence 和额外开销

2. `reductionSequentialKernel`
   - 用 sequential addressing
   - 避免 `%` 判断
   - 通常比第一个版本更合理

3. `reductionTwoLoadsKernel`
   - 每个线程先从 global memory 读两个元素
   - 再进入 shared memory 归约
   - 通常能减少 block 数量和一部分 global load 压力

程序会做：

- CPU 参考结果计算
- 三个 GPU 版本结果校验
- 三个 GPU 版本计时对比

## 5. 本周必须完成的练习

### 练习 1：跑通 reduction 对比程序

运行 `week4_reduction_compare`，记录 3 个版本的时间和结果。

### 练习 2：解释三个版本差别

请你用自己的话回答：

- 为什么 interleaved 版本通常不够理想
- sequential 版本主要改了什么
- two-loads 版本为什么可能更快

### 练习 3：修改 block size

尝试把 block size 改成：

- 128
- 256
- 512

观察计时变化。

### 练习 4：修改输入规模

尝试把 `n` 改成：

- `1 << 20`
- `1 << 22`
- `1 << 24`

观察不同数据规模下三个版本表现是否变化。

## 6. 本周完成标准

- 能写出正确 reduction
- 能解释树形归约思路
- 能解释 branch divergence、coalesced access、bank conflict 的基本含义
- 有一份“优化前后时间对比 + 自己的解释”

## 7. 本周最该记住的一句话

**Reduction 优化的核心，不只是把加法并行化，而是让线程分支更规整、访存更友好、shared memory 使用更合理。**
