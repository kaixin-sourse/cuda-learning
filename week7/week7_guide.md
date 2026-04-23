# CUDA C++ Week 7 详细学习资料

## 1. 本周目标

这一周的重点不是再写一个新算法，而是建立一个新的工程习惯：

**先 profile（性能分析），再谈优化。**

本周重点有 4 个：

1. 理解 Nsight Systems 和 Nsight Compute 看的是两类不同问题。
2. 理解 timeline、kernel 指标和热点分析的区别。
3. 理解 occupancy、register pressure、memory throughput 的基本含义。
4. 用一个实际 target 程序做最基础的 profiling 练习。

学完这一周，你至少应该能做到：

- 分清 `nsys` 和 `ncu` 各自适合做什么。
- 看懂最基本的 timeline。
- 看懂至少几个最常见的 kernel 指标。
- 能写出一份 1 页以内的 profiling 笔记。

## 2. Nsight Systems 和 Nsight Compute 的区别

### Nsight Systems

它更偏向全局时间线视角，主要看：

- CPU 在做什么
- GPU 在做什么
- kernel 是什么时候发起的
- 内存拷贝和 kernel 是否重叠
- 整个程序的执行阶段长什么样

可以把它理解成：

**先看整场比赛录像。**

### Nsight Compute

它更偏向单个 kernel 级别的深入分析，主要看：

- occupancy
- memory throughput
- 指令效率
- warp 执行情况
- 具体热点来源

可以把它理解成：

**再盯住一个关键回合做慢动作分析。**

## 3. 本周核心概念

## 3.1 timeline

timeline 解决的是：

- 程序大致花时间在哪些阶段
- kernel 调用顺序如何
- 有没有频繁的小 kernel
- 数据传输和 kernel 是否被合理组织

如果你连 timeline 都不看，就直接优化某个 kernel，常常会优化错对象。

## 3.2 occupancy

occupancy 可以简单理解为：

- 一个 SM 上活跃 warps 的相对占用程度

但你一定要避免一个误区：

**occupancy 高，不等于程序就一定快。**

occupancy 只是帮助你判断：

- 当前资源使用是否限制了并发度
- block 配置、寄存器、shared memory 是否压住了可并发线程数

## 3.3 register pressure

每个线程都会用寄存器。

如果一个 kernel 使用寄存器过多：

- 每个 SM 能同时驻留的线程块数量会下降
- occupancy 可能降低

所以 register pressure 是影响并发度的重要因素之一。

## 3.4 memory throughput

很多 CUDA 程序的瓶颈并不是“不会算”，而是“读写数据不够快”。

所以你要开始区分：

- 这是 compute-bound 问题
- 还是 memory-bound 问题

Week 7 你先建立直觉，不要求一上来就精通 roofline。

## 4. 你机器上的工具情况

我已经检查过你当前环境：

- `ncu` 可以直接运行
- `nsys.exe` 存在，但不在 PATH 里

你的本机路径是：

```text
C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe
C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\target\windows-desktop-win7-x64\ncu.exe
```

这意味着：

- Nsight Compute 命令行分析你现在就能做
- Nsight Systems 建议先按完整路径调用

## 5. 本周代码说明

## 5.1 `src/profile_reduction_target.cu`

    这是一个专门为了 profiling 准备的 reduction target。

它会：

- 分配较大数组
- 多次重复运行 reduction kernel
- 输出最终结果摘要

这个程序适合拿来做：

- Nsight Systems timeline 观察
- Nsight Compute kernel 指标查看

## 5.2 `src/profile_matmul_target.cu`

这是一个 tiled matmul profiling target。

它会：

- 多次运行 shared memory tiled matmul
- 生成较稳定的 kernel 行为
- 方便你观察 matmul 的 occupancy 和 memory 特征

## 6. 本周必须完成的练习

### 练习 1：运行 profiling targets

先分别运行：

- `week7_profile_reduction`
- `week7_profile_matmul`

确认程序本身正常工作。

### 练习 2：做一次 Nsight Systems timeline

建议先对 reduction target 做。

你要观察：

- kernel 是否连续发射
- 有没有明显的数据传输段
- CPU 是否在等待 GPU

### 练习 3：做一次 Nsight Compute 分析

建议先对 reduction target 或 matmul target 做。

你要记录：

- occupancy
- memory throughput
- 哪些指标看起来像主要瓶颈

### 练习 4：写 profiling 笔记

把结果整理到 `profile_notes_template.md` 对应格式里，控制在 1 页内。

## 7. 本周完成标准

- 能说清 `nsys` 和 `ncu` 的区别
- 能做一次 timeline 分析
- 能看懂至少几个基础 kernel 指标
- 能写出一份基础 profiling 笔记

## 8. 本周最该记住的一句话

**Week 7 的核心不是会点工具，而是建立“证据优先”的优化习惯。**
