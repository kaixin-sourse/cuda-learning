# CUDA C++ Week 3 详细学习资料

## 1. 本周目标

这一周你要从“理解基本数据流和错误检查”升级到“理解 block 内线程如何协作”。

本周重点有 3 个：

1. 理解 shared memory 为什么比 global memory 更适合 block 内复用数据。
2. 理解 `__syncthreads()` 的作用、边界和常见误区。
3. 完成一个入门版 tiled matrix multiplication，并和朴素版做对比。

学完这一周，你至少应该能做到：

- 解释 global memory 和 shared memory 的差别。
- 说清楚 `__syncthreads()` 为什么能保证 block 内线程协同安全。
- 写出一个基础版 tiled matmul，并完成正确性校验。
- 对比朴素版和 tiled 版的计时结果，并说出 tiled 版通常更快的原因。

## 2. Week 3 和前两周的关系

Week 1 解决的是：

- CUDA 程序最小闭环
- thread / block / grid
- 第一个 kernel

Week 2 解决的是：

- host/device 内存分配
- 数据拷贝
- 错误检查
- 基础 benchmark

Week 3 解决的是：

- 一个 block 内线程如何合作
- 为什么要把一部分数据搬进 shared memory
- 为什么矩阵乘法这类任务适合做 tile 化

所以 Week 3 的核心不是“再学几个 API”，而是开始理解真正的并行协作。

## 3. 本周核心概念

## 3.1 global memory 和 shared memory

### global memory

- 位于 GPU 全局显存中
- 容量大
- 所有线程都可以访问
- 延迟高于 shared memory

### shared memory

- 位于每个 block 关联的片上高速存储中
- 容量小很多
- 只有同一个 block 内的线程能访问
- 适合做 block 内的数据复用

你现在最该记住的一句话是：

**如果同一块数据会被同一个 block 的多个线程反复使用，那么 shared memory 往往值得考虑。**

## 3.2 为什么矩阵乘法适合 shared memory

在朴素矩阵乘法里，计算 `C[row][col]` 时，线程会不断从 global memory 读取：

- `A[row][k]`
- `B[k][col]`

问题在于：

- 同一个 block 内很多线程会重复访问 A 和 B 的某些相邻区域
- 如果每次都从 global memory 取，访存代价会比较高

tile 化的思路是：

1. 每次只处理 A 和 B 的一个小块
2. 先把这两个小块搬进 shared memory
3. block 内线程一起用这两个小块做一段局部乘加
4. 再换下一块

这样可以减少对 global memory 的重复访问。

## 3.3 `__syncthreads()` 是什么

`__syncthreads()` 是 block 级别的同步屏障。

它的作用是：

- 让一个 block 内所有线程都走到这里
- 确保前面的 shared memory 写入对同一个 block 内其他线程可见

在 tile 化矩阵乘法中，常见模式是：

1. 每个线程负责搬一点数据到 shared memory
2. 调一次 `__syncthreads()`
3. 所有线程开始读 shared memory 做计算
4. 再调一次 `__syncthreads()`
5. 准备搬下一轮 tile

## 3.4 `__syncthreads()` 的边界

你必须记住：

- `__syncthreads()` 只对同一个 block 内线程有效
- 它不能同步不同 block
- 如果同一个 block 中有些线程会走到同步点、有些线程不会，程序可能出错或挂起

所以最重要的约束是：

**同一个 block 里的所有线程必须以一致方式到达 `__syncthreads()`。**

## 3.5 warp 的基本概念

warp 是 GPU 调度线程的基本单位。

在大多数 NVIDIA GPU 上：

- 一个 warp = 32 个线程

你现在不需要过早陷入太多 warp 级细节，但要知道两件事：

1. 线程并不是完全独立逐个执行的，而是以 warp 为单位推进。
2. 线程组织方式、分支行为和访存方式，会影响执行效率。

## 4. 本周代码说明

## 4.1 `src/shared_memory_demo.cu`

这个程序用一个很小但直接的例子说明：

- 线程先把数据搬到 shared memory
- 再用 `__syncthreads()` 确保大家都搬完
- 然后再进行 block 内求和

这个例子的重点不是“算法多高级”，而是让你真正理解：

- shared memory 是 block 内共享的
- `__syncthreads()` 是必要的同步点

## 4.2 `src/matmul_compare.cu`

这个程序包含两种矩阵乘法实现：

1. 朴素版 matmul
2. tiled shared memory 版 matmul

程序会做三件事：

- 运行两个 kernel
- 检查结果是否一致
- 对比两者计时

你本周最重要的观察点就是：

- 为什么两个版本结果一样
- 为什么 tiled 版通常更快
- 哪一部分优化来自 shared memory 数据复用

## 5. 本周必须完成的练习

### 练习 1：说清楚 shared memory 的价值

请你用自己的话回答：

- shared memory 和 global memory 最大差别是什么
- 为什么不是所有数据都直接用 shared memory
- 什么情况下 shared memory 值得用

### 练习 2：运行 `week3_shared_memory_demo`

要求：

- 跑通程序
- 看懂 shared memory 数组是怎么被 block 内线程共同使用的
- 说清 `__syncthreads()` 在这里解决了什么问题

### 练习 3：运行 `week3_matmul_compare`

要求：

- 跑通朴素版和 tiled 版
- 观察两者的计时输出
- 确认结果校验通过

### 练习 4：修改矩阵规模

你可以尝试修改：

- 128 x 128
- 256 x 256
- 512 x 512

观察不同规模下两种实现的表现变化。

### 练习 5：修改 tile size

在代码里试着把 tile 大小改成：

- 8
- 16
- 32

然后观察：

- 是否都能编译通过
- 结果是否仍然正确
- 时间是否变化

## 6. 本周完成标准

满足下面这些，就算 Week 3 达标：

- 能解释 shared memory 的价值，而不是只会套模板
- 能说明 `__syncthreads()` 解决了什么问题
- 能写出一个基础版 tiled matmul
- 能做正确性校验
- 能说出 tiled 版比朴素版通常更快的原因

## 7. 学完本周后你应该具备的能力

- 开始理解 block 内线程协作，而不是只会“每个线程各做各的”
- 能把共享数据搬到 shared memory 并正确同步
- 能完成基础矩阵乘法优化实验
- 为后面的 reduction、scan 和更深入优化打下基础

## 8. 你现在最该记住的一句话

**Week 3 的重点不是 shared memory 这个名词，而是“数据复用 + 正确同步”这两个动作。**
