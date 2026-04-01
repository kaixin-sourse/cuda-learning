# CUDA C++ 学习计划

## 1. 学习定位

### 你的起点
- 你已经具备 C++ 基础，能够熟练使用 STL。
- 你的算法基础不错，常见算法题可以独立做出来。
- 你的下一步不是单纯“会写 CUDA 代码”，而是要为后续在 CT 成像、医学影像处理、模型训练底层优化中的工程工作做准备。

### 12 周后的目标
- 能独立搭建并维护一个基础 CUDA C++ 项目。
- 能理解 GPU 执行模型，并写出正确的 kernel。
- 能看懂并分析 block、grid、warp、shared memory、global memory 等核心概念在性能上的影响。
- 能使用 profiling 工具定位瓶颈，而不是只靠“猜测优化”。
- 能完成 2 个项目：
  - 一个通用并行算法项目。
  - 一个贴近医学影像 / 张量处理的 CUDA 小项目。
- 能理解后续如何过渡到 PyTorch CUDA Extension、自定义算子和训练框架底层优化。

### 学习原则
- 先正确，再并行，再优化。
- 每周必须有代码产出，而不是只看资料。
- 每周至少做一次正确性校验、一次 benchmark、一次复盘。
- 看到性能问题时，优先用工具分析，而不是一开始就手调参数。

## 2. 学习节奏总览

### 时间投入
- 总周期：12 周
- 每周投入：12 小时以上
- 推荐拆分：
  - 4 小时：读资料和理解概念
  - 6 小时：编码实践
  - 2 小时：复盘、记录、整理问题

### 阶段划分
- 第 1 阶段（第 1-3 周）：CUDA 基础与开发环境
- 第 2 阶段（第 4-6 周）：典型并行算法与内存模型
- 第 3 阶段（第 7-9 周）：性能分析与 kernel 优化
- 第 4 阶段（第 10-12 周）：通用实战 + 医学影像 / 张量方向实战

### 阶段性检查点
- 第 3 周结束：你应该能独立写出简单 kernel，并完成基础矩阵乘法。
- 第 6 周结束：你应该能写出 reduction、scan 等典型并行算法，并理解常见性能问题来源。
- 第 9 周结束：你应该能使用 profiling 工具并完成一轮像样的优化。
- 第 12 周结束：你应该能完成项目闭环，并知道下一步怎么进入科研 / 工程场景。

## 3. 12 周详细计划

## 第 1 阶段：CUDA 基础与开发环境

### 第 1 周：GPU 架构入门与第一个 CUDA 程序
**学习目标**
- 建立 GPU 与 CPU 的基本区别认知。
- 理解 CUDA 编程模型中的 thread、block、grid。
- 完成开发环境搭建并跑通第一个 kernel。

**核心知识点**
- GPU 为什么适合数据并行。
- CUDA 程序的基本结构：host 代码、device 代码、kernel 调用。
- thread index、block index、global index 的计算方式。
- kernel launch 的基本语法。

**实践任务**
- 安装并验证 CUDA 工具链。
- 写一个 `vector_add.cu`，实现向量加法。
- 写一个 CPU 版本，对比 CUDA 版本结果。
- 给程序补上最基本的输入初始化、输出打印和结果校验。

**验收标准**
- 能清楚解释 thread、block、grid 的关系。
- 能独立写出一个最简单的 kernel。
- 能跑通 vector add，并验证 GPU 结果与 CPU 一致。

**建议资源**
- 必读：CUDA C++ Programming Guide 中关于编程模型的入门章节。
- 选读：NVIDIA Developer Blog 上的 CUDA 入门文章。

### 第 2 周：内存管理、错误检查与基础调试
**学习目标**
- 掌握 host/device 内存分配与数据拷贝。
- 学会加入错误检查，避免“程序能跑但结果错了”。
- 初步建立计时和设备查询意识。

**核心知识点**
- `cudaMalloc`、`cudaFree`、`cudaMemcpy` 的使用。
- host memory 与 device memory 的区别。
- kernel 启动参数选择的基本思路。
- `cudaGetLastError`、`cudaDeviceSynchronize`、错误码检查。
- `cudaGetDeviceProperties` 和基础计时方法。

**实践任务**
- 重构第 1 周的 vector add，补上统一错误检查宏。
- 查询当前设备基本属性并输出。
- 尝试不同 block size，观察运行时间变化。
- 写一个简单脚本或说明文档，记录不同输入规模下的结果。

**验收标准**
- 能独立解释数据从 CPU 到 GPU 再回 CPU 的完整流程。
- 所有 CUDA API 调用和 kernel 启动都带错误检查。
- 能用基础计时方式完成一次小规模 benchmark。

**建议资源**
- 必读：CUDA Best Practices Guide 中关于错误处理和基本性能测量的部分。
- 选读：官方 samples 中与 memory copy 相关的简单例子。

### 第 3 周：Shared Memory、同步与入门版矩阵乘法
**学习目标**
- 理解 shared memory 的用途。
- 理解 block 内同步和 warp 的基本概念。
- 完成 tiled matrix multiplication 的入门实现。

**核心知识点**
- global memory 与 shared memory 的访问差异。
- `__syncthreads()` 的作用和使用边界。
- warp 的基本概念，以及为什么线程组织会影响性能。
- tiled matrix multiplication 的基本思路。

**实践任务**
- 先实现一个朴素版 matrix multiply。
- 再实现一个使用 shared memory 的 tiled 版本。
- 对比两者在中等规模矩阵上的时间差异。
- 写下你对“为什么 tiled 版本更快”的解释。

**验收标准**
- 能解释 shared memory 的价值，而不是只会照抄模板。
- 能完成矩阵乘法的基础正确性校验。
- 能说清 `__syncthreads()` 解决了什么问题。

**建议资源**
- 必读：Programming Guide 中关于 shared memory 和 synchronization 的章节。
- 选读：NVIDIA 关于矩阵乘法优化的博客或 sample。

## 第 2 阶段：典型并行算法与内存模型

### 第 4 周：Reduction 与常见性能问题
**学习目标**
- 掌握 reduction 的基本写法与优化方向。
- 理解分支发散、访存合并、bank conflict 的含义。

**核心知识点**
- reduction 的树形并行思路。
- 分支发散为什么会浪费执行资源。
- coalesced memory access 的意义。
- shared memory bank conflict 的基本概念。

**实践任务**
- 实现一个 sum reduction。
- 从朴素版开始，逐步做 2 到 3 次优化。
- 记录每次优化后的时间变化。
- 用文字总结每次优化针对的是哪类瓶颈。

**验收标准**
- 能写出正确的 reduction。
- 能用自己的话解释 branch divergence、memory coalescing、bank conflict。
- 至少产出一份“优化前后对比记录”。

**建议资源**
- 必读：NVIDIA 关于 reduction 的经典资料。
- 选读：Best Practices Guide 中关于 memory access 的章节。

### 第 5 周：Prefix Sum / Scan 与数据重排
**学习目标**
- 理解 scan 作为并行算法基础模块的重要性。
- 在 scan 与数据重排类任务之间建立联系。

**核心知识点**
- inclusive scan 与 exclusive scan。
- Blelloch scan 的基本思路。
- scan 在 stream compaction、histogram、排序等问题中的作用。

**实践任务**
- 实现 prefix sum / scan。
- 在下面两个任务中二选一：
  - 实现 stream compaction。
  - 实现 histogram。
- 和 CPU 版本对拍，确保正确性。

**验收标准**
- 能讲清 scan 的上扫 / 下扫思路，或至少能清楚说明实现流程。
- 完成一个 scan 的应用任务。
- 能通过随机数据测试验证结果。

**建议资源**
- 必读：GPU 并行算法中关于 scan 的经典材料。
- 选读：相关 sample 或课程笔记。

### 第 6 周：原子操作、排序 / Top-K 与 CPU-GPU 对比
**学习目标**
- 理解原子操作的用途和代价。
- 认识排序 / top-k 这类更复杂任务中的并行化难点。

**核心知识点**
- atomic operation 的适用场景。
- contention 对性能的影响。
- 并行排序和 top-k 的高层思路。
- CPU 与 GPU 的边界：不是所有任务 GPU 都更快。

**实践任务**
- 写一个使用原子操作的简单统计任务。
- 在下面两个方向中二选一：
  - 实现一个简化版并行排序思路。
  - 实现一个 top-k 的近似或分阶段版本。
- 记录 CPU 与 GPU 版本在不同数据规模下的差异。

**验收标准**
- 能解释什么时候应该避免过度依赖 atomic。
- 能总结一个“GPU 不一定更快”的具体案例。
- 能产出一次 CPU-GPU 对比实验记录。

**建议资源**
- 必读：Programming Guide 中关于 atomics 的部分。
- 选读：并行排序、top-k 的工程文章。

## 第 3 阶段：性能分析与 Kernel 优化

### 第 7 周：Profiling 工具入门
**学习目标**
- 建立“先 profile 再优化”的工程习惯。
- 学会使用 Nsight Systems 和 Nsight Compute 看基本信息。

**核心知识点**
- timeline 与 kernel 级指标的区别。
- occupancy、register pressure、memory throughput 的基本含义。
- roofline 的直觉理解：算力受限还是带宽受限。

**实践任务**
- 选择前面写过的一个 kernel 做 profiling。
- 用 Nsight Systems 看 kernel 调度和 CPU-GPU 时间线。
- 用 Nsight Compute 看 occupancy、memory 指标和热点。
- 输出一份 1 页以内的 profiling 笔记。

**验收标准**
- 能看懂最基本的 timeline。
- 能解释 occupancy 不是越高越好，而是要结合实际瓶颈判断。
- 能指出当前 kernel 最可能的性能限制来源。

**建议资源**
- 必读：Nsight Systems / Nsight Compute 官方入门文档。
- 选读：NVIDIA 关于 roofline 和 kernel 分析的博客。

### 第 8 周：基线版 vs 优化版
**学习目标**
- 完成一次比较完整的性能优化闭环。
- 学会把“优化”写成可对比、可复现的结果，而不是模糊描述。

**核心知识点**
- baseline 的定义方式。
- 优化要有假设、有验证、有结论。
- 常见优化手段：调整 block size、减少 global memory 访问、改善访问模式、利用 shared memory。

**实践任务**
- 选择 `matrix multiply` 或 `reduction` 作为对象。
- 整理一个基线版本。
- 做至少 2 次有明确目的的优化。
- 输出一份表格，对比输入规模、运行时间、加速比和你的解释。

**验收标准**
- 形成“基线版 vs 优化版”的清晰对比。
- 优化结论能被 profiling 数据支持。
- 有一份可复现实验记录。

**建议资源**
- 必读：Best Practices Guide 中关于 kernel 优化的章节。
- 选读：矩阵乘法、reduction 优化案例文章。

### 第 9 周：Streams、Overlap 与吞吐优化
**学习目标**
- 理解 kernel 执行之外，数据传输与流水线组织也会影响性能。
- 认识吞吐优化与单 kernel 优化的区别。

**核心知识点**
- streams 的基本概念。
- data transfer 与 kernel execution overlap。
- pinned memory 的用途。
- 多 kernel pipeline 的基础思路。

**实践任务**
- 写一个包含多次数据传输与 kernel 调用的实验程序。
- 尝试将任务拆到多个 stream 中。
- 使用 pinned memory 观察是否有收益。
- 记录吞吐场景下的优化效果。

**验收标准**
- 能解释 streams 解决的不是“让代码更复杂”，而是更好利用设备。
- 能完成一个基础 overlap 实验。
- 能指出当前场景中数据传输是否已经成为瓶颈。

**建议资源**
- 必读：Programming Guide 中关于 streams 和 async execution 的部分。
- 选读：NVIDIA 关于 overlap 和 pipeline 的案例。

## 第 4 阶段：通用实战 + 医学影像 / 张量方向实战

### 第 10 周：通用并行算法 mini library
**学习目标**
- 把前面零散的算法实现整理成一个可复用的小工程。
- 建立“代码结构 + 测试 + benchmark”的工程闭环。

**核心知识点**
- 模块化组织 CUDA 工程。
- 正确性测试与性能测试的分离。
- benchmark 的最小工程规范。

**实践任务**
- 完成一个 mini library，至少包含：
  - `vector add`
  - `matmul`
  - `reduction`
  - `scan`
  - `benchmark`
- 为每个模块补基础测试样例。
- 给 benchmark 输出统一格式。

**验收标准**
- 代码结构清晰，能独立运行和复用。
- 每个模块都有正确性校验。
- benchmark 结果可直接比较。

**建议资源**
- 必读：参考官方 sample 的项目组织方式。
- 选读：简单的 CMake CUDA 项目模板。

### 第 11 周：医学影像 / 张量处理项目
**学习目标**
- 把 CUDA 基础迁移到更贴近 CT 成像和模型训练的数据处理问题。
- 开始建立“科研问题如何拆成 CUDA 任务”的意识。

**推荐题目**
- 下面四个方向优先选一个：
  - 2D / 3D 图像卷积
  - 图像滤波
  - 插值与重采样
  - sinogram / 投影预处理中的一个基础步骤

**核心知识点**
- 图像 / 体数据的内存布局。
- stencil 类操作的局部访存特征。
- 数据边界处理。
- 张量 / 图像任务和通用并行算法之间的联系。

**实践任务**
- 先写一个 CPU 基线版本。
- 再实现 CUDA 版本。
- 对比不同尺寸输入下的正确性和性能。
- 写一段说明：这个任务与 CT 成像或模型训练前后处理有什么关系。

**验收标准**
- 项目能正确运行并完成结果对拍。
- 你能清楚说明这个任务为什么贴近医学影像 / 张量场景。
- 你能指出接下来如果继续做大，应该从哪里优化。

**建议资源**
- 必读：图像卷积 / stencil 优化相关资料。
- 选读：与你未来方向更接近的医学影像预处理论文或工程仓库。

### 第 12 周：总结、迁移与下一阶段入口
**学习目标**
- 总结 12 周内容，形成自己的知识图谱。
- 建立从 CUDA 基础到训练框架底层优化的过渡路径。

**核心知识点**
- 自定义 CUDA op 的基本工作流。
- PyTorch extension 的基本结构。
- “科研任务 -> 算子抽象 -> CUDA 实现 -> profiling -> 优化”的完整链路。

**实践任务**
- 回顾前 11 周的代码和笔记。
- 整理一份个人总结：
  - 学会了什么
  - 还不熟的是什么
  - 哪些问题是后续重点
- 阅读一个 PyTorch CUDA Extension 或自定义算子示例。
- 画出你未来 8-12 周的进阶路线图。

**验收标准**
- 能说清楚后续如果要改模型训练底层逻辑，自己应该先看哪里。
- 能理解 CUDA 工程与深度学习框架之间如何连接。
- 能产出一份不依赖别人解释的个人学习总结。

**建议资源**
- 必读：PyTorch C++ / CUDA Extension 官方文档。
- 选读：CUTLASS、Tensor Core、混合精度相关资料。

## 4. 两个核心项目的设计目标

### 项目 1：通用并行算法项目
**建议内容**
- `vector add`
- `matmul`
- `reduction`
- `scan`
- `benchmark`
- 正确性校验模块

**项目目标**
- 建立从“写代码”到“验证正确性”再到“测性能”的完整闭环。
- 熟悉 CUDA 工程最常见的基础算法模式。
- 为后续进入更复杂算子打底。

### 项目 2：医学影像 / 张量处理项目
**建议内容**
- 图像卷积
- 插值重采样
- sinogram / 投影预处理
- 3D 张量操作中的一个基础算子

**项目目标**
- 把通用 CUDA 技能迁移到你未来更可能接触的真实问题。
- 训练你从“问题定义”到“数据布局”再到“kernel 设计”的能力。
- 为后续做 CT 成像相关加速、数据预处理加速或训练框架底层改造做铺垫。

### 这些项目与未来工作的关系
- 如果未来你在模型训练里需要修改底层逻辑，本质上常常是在做：
  - 自定义算子
  - 数据预处理加速
  - 特定 kernel 的 profiling 与优化
- 通用并行算法项目让你掌握基础构件。
- 医学影像 / 张量项目让你开始接触更贴近科研任务的数据形态和访存模式。
- 两者结合之后，你会更容易理解为什么深度学习框架底层实现会这样设计。

## 5. 每周固定输出模板

为了避免“学了很多但没有沉淀”，建议你每周固定输出下面 5 项：

1. 本周学了哪些概念。
2. 本周写了哪些代码。
3. 哪个实验结果最有价值。
4. 当前最大的理解障碍是什么。
5. 下周要优先补的内容是什么。

你也可以直接用下面这个模板做每周复盘：

```markdown
## 第 X 周复盘
- 本周目标：
- 本周完成：
- 本周代码文件：
- 正确性验证方式：
- benchmark 结果：
- 遇到的问题：
- 我对性能瓶颈的判断：
- 下周计划：
```

## 6. 资源使用建议

### 必读主线
- CUDA C++ Programming Guide
- CUDA Best Practices Guide
- NVIDIA Developer Blog
- Nsight Systems / Nsight Compute 官方文档

### 选读辅助
- 1 本 CUDA 相关教材或系统课程即可，不要同时开太多资料。
- 一些经典博客、sample、课程讲义可以作为“卡点时的补充”。

### 资源使用原则
- 官方文档做主线，视频或课程做辅助。
- 每看完一个核心概念，必须马上写代码验证。
- 避免长期停留在“看懂了，但没做过”的状态。

## 7. 完成 12 周后你应具备的能力

- 能独立写基础 CUDA kernel，并完成编译、调试和运行。
- 能理解常见的性能问题来自哪里，而不是只会盲目调 block size。
- 能对一个小型 CUDA 项目做正确性校验和 benchmark。
- 能使用 profiling 工具做基础分析。
- 能把一个图像 / 张量处理任务拆成可实现的 CUDA 问题。
- 能开始阅读自定义算子、PyTorch extension、图像处理加速相关代码。

## 8. 下一阶段路线

如果你完成了这 12 周，下一步建议按下面顺序继续：

1. PyTorch CUDA Extension 与自定义算子
2. CUTLASS 与高性能 GEMM 思路
3. Tensor Core 与混合精度
4. 更贴近医学影像的 2D / 3D 算子优化
5. 多 GPU 与更大规模吞吐优化

## 9. 常见误区

- 只会写 kernel，不会做 profiling。
- 只看资料，不做 benchmark。
- 只跑一个样例，就以为程序正确。
- 过早追求极致优化，结果基础概念反而不清楚。
- 只盯着单个 kernel，不看完整数据流和吞吐。
- 一上来就研究非常复杂的深度学习框架底层，导致基础不稳。

## 10. 最后的建议

你的目标不是“短期内把 CUDA 所有高级特性都学完”，而是先建立一个稳固、可扩展、能迁移到科研和工程场景的基础。  
这 12 周如果认真执行，最大的收获不是背下多少 API，而是你会真正形成下面这条路径：

**问题定义 -> CUDA 实现 -> 正确性验证 -> profiling -> 优化 -> 迁移到真实任务**

这条路径一旦建立起来，后续无论你是做 CT 成像、医学影像处理，还是模型训练中的底层逻辑优化，都会更顺。
