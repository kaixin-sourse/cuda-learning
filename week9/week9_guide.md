# CUDA C++ Week 9 详细学习资料

## 1. 本周目标

这一周你要从“优化单个 kernel”扩展到“优化完整吞吐流程”。

本周重点有 4 个：

1. 理解 streams 的基本概念。
2. 理解 pinned memory 为什么有助于异步传输。
3. 理解 data transfer 与 kernel execution overlap。
4. 实现一个多 stream chunk pipeline，并和单 stream 版本对比。

学完这一周，你至少应该能做到：

- 解释 stream 解决的是调度和重叠问题，而不是让代码看起来更复杂。
- 解释 pageable memory 和 pinned memory 的区别。
- 写出 `cudaMemcpyAsync` + kernel launch + stream 同步的基本流程。
- 通过实验判断当前任务是否被数据传输限制。

## 2. 为什么需要 streams

默认情况下，很多 CUDA 操作会按顺序执行。

如果你的程序流程是：

```text
copy input -> kernel -> copy output
```

那么 GPU 和 PCIe 传输通道可能没有被充分利用。

streams 的作用是让你把任务拆成多个队列：

```text
stream 0: copy chunk 0 -> kernel chunk 0 -> copy chunk 0 back
stream 1: copy chunk 1 -> kernel chunk 1 -> copy chunk 1 back
stream 2: copy chunk 2 -> kernel chunk 2 -> copy chunk 2 back
```

这样不同 chunk 的传输和计算就可能发生重叠。

## 3. pageable memory 与 pinned memory

普通 `std::vector` 使用的是 pageable host memory。

异步传输要真正有效，通常需要 pinned host memory，也就是页锁定内存。

CUDA 中常用：

```cpp
cudaMallocHost(&ptr, bytes);
cudaFreeHost(ptr);
```

pinned memory 的特点：

- 更适合 DMA 传输
- 可以让 `cudaMemcpyAsync` 更有效
- 但不应该无限制申请，因为它会减少操作系统可分页内存

## 4. overlap 的判断

你要通过实验判断：

- 单 stream 是否基本串行
- 多 stream 是否降低总时间
- 传输是否占据明显时间
- kernel 是否足够长，足以和传输形成重叠

注意：

**多 stream 不保证一定更快。**

如果任务太小，或者设备/驱动模式限制了重叠能力，多 stream 可能收益不明显。

## 5. 本周代码说明

## `src/stream_overlap_compare.cu`

这个程序比较 3 种方式：

1. pageable host memory + 单次同步传输
2. pinned host memory + 单次同步传输
3. pinned host memory + 多 stream chunk pipeline

每个 chunk 会经过两个 kernel：

```text
scaleKernel -> biasKernel
```

这样它不仅有数据传输，也有一个简单的多 kernel pipeline。

程序会输出：

- 设备 async engine 数量
- 三种方案的总时间
- 结果校验

## 6. 本周必须完成的练习

### 练习 1：运行 stream 对比程序

记录三种方案时间：

- pageable sequential
- pinned sequential
- pinned multi-stream

### 练习 2：修改 chunk size

尝试：

- `1 << 18`
- `1 << 20`
- `1 << 22`

观察 chunk 太小或太大时表现如何。

### 练习 3：修改 stream 数量

尝试：

- 2
- 4
- 8

观察是否一定越多越好。

### 练习 4：用 Nsight Systems 看 timeline

用 Week 7 学过的方法看：

- H2D copy
- kernel
- D2H copy
- 不同 stream 是否重叠

## 7. 本周完成标准

- 能解释 stream 的作用
- 能解释 pinned memory 的作用
- 能跑通基础 overlap 实验
- 能指出当前实验中传输或计算哪个更像瓶颈

## 8. 本周最该记住的一句话

**Week 9 的重点不是单个 kernel 更快，而是让传输、计算和多阶段任务组织得更有效。**
