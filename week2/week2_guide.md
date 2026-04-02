# CUDA C++ Week 2 详细学习资料

## 1. 本周目标

这一周你要从“能跑一个 kernel”升级到“能把一个 CUDA 程序写得更像工程代码”。

本周重点有 4 个：

1. 理解 host 和 device 之间的数据流动。
2. 学会给 CUDA API 和 kernel 启动加统一错误检查。
3. 会查询当前 GPU 的基础属性。
4. 会用最基础的方式给 kernel 计时并做小规模 benchmark。

学完这一周，你至少应该能做到：

- 独立写出 `cudaMalloc`、`cudaMemcpy`、`cudaFree` 的完整流程。
- 解释数据如何从 CPU 到 GPU，再从 GPU 回到 CPU。
- 给程序加上统一错误检查，而不是等程序出错后再猜。
- 用 `cudaEvent` 做一次基础计时实验。

## 2. Week 2 和 Week 1 的关系

Week 1 你解决的是：

- 什么是 kernel
- 什么是 thread / block / grid
- 为什么 GPU 适合数据并行

Week 2 要解决的是：

- 数据怎么进入 GPU
- GPU 算完以后结果怎么回来
- 如果中间某一步失败了，怎么第一时间发现
- 怎么知道程序到底快不快

所以 Week 2 的核心不是“写更复杂的算法”，而是把 Week 1 的代码补成一版更规范、更可观察的程序。

## 3. 这一周最重要的执行流程

一个典型的 CUDA 程序，从工程视角看，通常是这个顺序：

1. 在 host 端准备输入数据
2. 在 device 端分配显存
3. 把输入从 host 拷到 device
4. 启动 kernel
5. 检查 kernel 启动是否成功
6. 等待 GPU 执行结束
7. 把结果从 device 拷回 host
8. 做正确性校验
9. 释放显存

Week 1 你已经走通过一次这个流程。Week 2 要做的是把这个流程真正吃透。

## 4. 本周核心概念

## 4.1 host memory 和 device memory

- `host memory`：CPU 侧内存，也就是普通 C++ 程序里常见的内存
- `device memory`：GPU 侧显存

最关键的认识是：

**CPU 指针不能直接当 GPU 指针用，GPU 指针也不能直接在 CPU 上访问。**

所以你必须显式地做拷贝：

```cpp
cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
```

## 4.2 `cudaMalloc`、`cudaFree`、`cudaMemcpy`

这是 Week 2 必须熟到看到就能反应过来的 3 个 API。

### `cudaMalloc`

作用：
- 在 GPU 显存上申请一块内存

典型形式：

```cpp
float* d_a = nullptr;
cudaMalloc(&d_a, bytes);
```

### `cudaFree`

作用：
- 释放之前在 GPU 上申请的显存

典型形式：

```cpp
cudaFree(d_a);
```

### `cudaMemcpy`

作用：
- 在 host 和 device 之间做显式拷贝

常见方向：

```cpp
cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
```

## 4.3 为什么必须做错误检查

CUDA 学习初期最常见的问题不是“程序直接崩溃”，而是：

- 程序看起来跑了
- 结果却是错的
- 甚至有时不会立刻报错

所以你必须养成习惯：

- 每个 CUDA API 调用都检查返回值
- 每次 kernel 启动后都检查一次
- 在关键点调用 `cudaDeviceSynchronize()`

本周你最该形成的习惯就是：

**不是出问题了才查错，而是默认每一步都可能错。**

## 4.4 kernel 启动后的两步检查

通常建议写成：

```cpp
kernel<<<gridSize, blockSize>>>(...);
CHECK_CUDA(cudaGetLastError());
CHECK_CUDA(cudaDeviceSynchronize());
```

两者的作用不同：

- `cudaGetLastError()`：检查 kernel 启动配置有没有明显错误
- `cudaDeviceSynchronize()`：等待 GPU 执行完成，同时让运行时错误更早暴露

## 4.5 `cudaGetDeviceProperties`

这个 API 用来查看当前 GPU 的基本属性，例如：

- 名称
- 计算能力
- global memory 大小
- 每个 block 最大线程数
- warp size
- multiprocessor 数量

你现在不需要把每个字段都背下来，但要开始知道：

**程序能利用什么资源，和设备属性是直接相关的。**

## 4.6 为什么要学计时

很多初学者写完 CUDA 程序以后，只看：

- 程序能不能跑
- 输出是不是对

这还不够。

CUDA 的意义之一是性能提升，所以从 Week 2 开始，你要建立最基础的 benchmark 意识：

- 不同输入规模下多快
- 不同 block size 下多快
- GPU 版本和 CPU 版本差异有多大

本周先不追求复杂 profiling，只做最基础的 event timing。

## 5. 本周代码说明

## 5.1 `src/device_query.cu`

这个程序负责：

- 查询设备数量
- 输出当前 GPU 的基础属性

它的作用不是“写算法”，而是让你认识你当前机器的 GPU 能力边界。

你运行后，建议重点关注：

- GPU 名称
- compute capability
- total global memory
- warp size
- max threads per block
- multiprocessor count

## 5.2 `src/vector_add_benchmark.cu`

这个程序是在 Week 1 `vector add` 基础上做的工程化升级。

它增加了：

- 统一错误检查宏
- 不同输入规模测试
- 不同 block size 测试
- `cudaEvent` 计时
- 正确性校验

你要重点观察：

- block size 改变时，时间如何变化
- 输入规模变大后，时间如何变化
- 为什么程序结果始终还要做正确性校验

## 6. 你本周必须完成的练习

### 练习 1：说清楚完整数据流

请你用自己的话写下来：

- 哪一步是 host 分配内存
- 哪一步是 device 分配内存
- 哪一步是数据从 CPU 到 GPU
- 哪一步是数据从 GPU 回 CPU

如果你说不清这 4 个点，说明 Week 2 还没真正掌握。

### 练习 2：运行 `device_query`

要求：

- 运行程序
- 记录你的 GPU 名称、global memory、max threads per block、warp size
- 写一个 5 行以内的总结

### 练习 3：运行 benchmark

要求：

- 运行 `week2_vector_add_benchmark`
- 观察不同 block size 的计时结果
- 观察不同数组长度的计时结果

你不用追求结论绝对精确，但要开始建立“参数变化会影响性能”的意识。

### 练习 4：修改输入规模

自己在代码里添加或修改一组输入规模，例如：

- `1 << 18`
- `1 << 20`
- `1 << 22`
- `1 << 24`

然后比较结果。

### 练习 5：修改 block size

自己尝试：

- 64
- 128
- 256
- 512

然后思考：

- 为什么都能跑
- 为什么时间未必一样
- 为什么 block size 不是越大越好

## 7. 本周完成标准

满足下面这些，就算 Week 2 达标：

- 能独立写出 `cudaMalloc` / `cudaMemcpy` / `cudaFree`
- 能给 CUDA API 和 kernel 启动补完整错误检查
- 能运行设备查询程序并解释几个关键属性
- 能做一次基础 benchmark，并记录结果
- 能清楚描述 CPU -> GPU -> CPU 的数据流

## 8. 学完本周后你应该具备的能力

- 不再只是“照着模板写 CUDA”
- 开始能看懂一个 CUDA 程序的运行路径
- 对错误检查和基础性能观察建立习惯
- 为 Week 3 的 shared memory 和同步打下工程基础

## 9. 你现在最该记住的一句话

**Week 1 是让程序跑起来，Week 2 是让程序跑得更可控、更可检查。**
