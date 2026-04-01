# CUDA C++ Week 1 详细学习资料

## 1. 本周目标

这一周你只做三件事：

1. 建立 GPU 与 CPU 的基本区别认知。
2. 理解 CUDA 编程模型中的 `thread`、`block`、`grid`。
3. 跑通第一个 CUDA 程序，并且看懂它为什么能跑。

如果这一周学完，你至少应该能做到：
- 能解释为什么 GPU 适合做大规模并行计算。
- 能看懂一个最简单的 CUDA 程序结构。
- 能独立写出一个 `vector add` 的 kernel。
- 能理解 host 代码、device 代码、数据拷贝、kernel 启动这几个基本步骤。

## 2. 你这台电脑当前的 CUDA 环境检查结果

### 已确认可用的部分
- GPU：`NVIDIA GeForce GTX 1650 Ti`
- 驱动：`566.36`
- `nvidia-smi` 可正常运行
- 驱动侧报告的 CUDA 运行时支持版本：`12.7`
- `nvcc --version` 可正常运行
- 已安装 CUDA Toolkit：`12.6`
- `CUDA_PATH` 已存在：

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

- 已安装 CLion：

```text
E:\softwares\Developments\Clion\CLion 2023.3.4\bin\clion64.exe
```

- CLion 自带的 CMake 和 Ninja 也存在：

```text
E:\softwares\Developments\Clion\CLion 2023.3.4\bin\cmake\win\x64\bin\cmake.exe
E:\softwares\Developments\Clion\CLion 2023.3.4\bin\ninja\win\x64\ninja.exe
```

- 已安装 Visual Studio：

```text
E:\softwares\visualstudio\vs
```

- 已找到 MSVC 编译器 `cl.exe`：

```text
E:\softwares\visualstudio\vs\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe
```

### 当前真正的问题
我已经实际做了三步验证：

1. 直接在当前 PowerShell 里编译  
结果：`nvcc` 找不到 `cl.exe`

```text
nvcc fatal   : Cannot find compiler 'cl.exe' in PATH
```

2. 手动加载 Visual Studio 开发环境后检查 `cl`  
结果：`cl` 可以正常使用

3. 手动加载 Visual Studio 开发环境后再编译 `week1_vector_add.cu`  
结果：CUDA 12.6 会报 Visual Studio 版本不受官方支持

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include\crt/host_config.h(170): fatal error C1189:
-- unsupported Microsoft Visual Studio version!
Only the versions between 2017 and 2022 (inclusive) are supported!
```

然后我继续使用 `-allow-unsupported-compiler` 做了兼容性编译测试，结果如下：

- `week1_vector_add.cu` 编译成功
- 可执行文件成功生成
- 实际运行结果：

```text
gridSize = 4096, blockSize = 256
Result check: PASS
```

这说明你现在的机器上：
- 不是没装 Visual Studio。
- 也不是没有 MSVC。
- 而是有两个实际问题：
  - 当前 PowerShell / CLion 工具链环境没有自动带上 VS 的开发者变量。
  - 你安装的是 `Visual Studio Community 2026`，而 `CUDA 12.6` 官方只支持到 `Visual Studio 2022`。

### 结论
你的电脑当前状态是：

- `GPU 驱动`：可以
- `CUDA Toolkit`：可以
- `CLion`：可以
- `CMake/Ninja`：CLion 自带可用
- `MSVC / cl.exe`：已安装
- `CUDA 代码本地编译运行`：可以跑，但当前属于“非官方支持组合”

换句话说，你现在不是“没环境”，而是：

**环境基本能用，但需要按当前版本组合做兼容配置。**

## 3. 你应该怎么把 CLion 的 CUDA 环境补齐

### 推荐方案
在 Windows + CLion + CUDA 这个组合下，推荐用：

- CLion 作为 IDE
- CUDA Toolkit 作为 CUDA 编译环境
- Visual Studio 提供 `cl.exe`

### 你当前更现实的两条路

#### 路线 A：先用当前环境学习，接受兼容参数
适合你现在就开始 Week 1，不想先折腾版本切换。

你需要做的是：
- 在 CLion 里使用 Visual Studio / MSVC 工具链
- 让 CMake 识别 CUDA
- 在 CUDA 编译参数里带上 `-allow-unsupported-compiler`

这个仓库里的 `CMakeLists.txt` 已经加了这个兼容参数。

#### 路线 B：改成官方支持组合
适合你后面认真做更多 CUDA 开发。

更稳妥的方式是：
- 保留 CLion
- 安装或切换到 Visual Studio 2022 Build Tools / Community
- 让 CUDA 12.6 配合 VS 2022 使用

这条路线更符合官方支持矩阵，后续踩坑会少很多。

### 在 CLion 里怎么配
CLion 中重点看两个地方：

1. `Settings -> Build, Execution, Deployment -> Toolchains`
2. `Settings -> Build, Execution, Deployment -> CMake`

推荐配置思路：
- Toolchain 选择 Visual Studio
- CMake 使用 CLion 自带的 `cmake.exe`
- Build tool 使用 CLion 自带的 `ninja.exe`
- 编译器让 CLion 识别到 MSVC 和 CUDA

你当前机器上可用的路径是：

```text
Visual Studio: E:\softwares\visualstudio\vs
CMake: E:\softwares\Developments\Clion\CLion 2023.3.4\bin\cmake\win\x64\bin\cmake.exe
Ninja: E:\softwares\Developments\Clion\CLion 2023.3.4\bin\ninja\win\x64\ninja.exe
CUDA: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

### 你配置成功后的最小验证标准
只要下面三件事都成功，就说明环境基本通了：

1. `nvcc --version` 正常
2. CLion 能识别 `project(... LANGUAGES CXX CUDA)`
3. `week1_vector_add.cu` 能编译并运行

你当前已经满足了其中第 1 条，并且在命令行兼容模式下也验证通过了第 3 条。

## 4. Week 1 必学概念

## 4.1 GPU 和 CPU 的差别

### CPU 擅长什么
CPU 擅长：
- 复杂控制逻辑
- 分支多、依赖强的任务
- 单线程性能高
- 通用性强

### GPU 擅长什么
GPU 擅长：
- 大量相似操作同时执行
- 数据并行
- 吞吐优先
- 对同一种计算反复作用在大量数据上

### 一个直观例子
如果你要把一个长度为 1000000 的数组里每个元素都乘 2：
- CPU 的常规做法是顺序或少量线程并行处理。
- GPU 的思路是：开很多线程，每个线程负责一个元素。

这就是 CUDA 最基础的思维方式：

**把一个大任务拆成很多相互独立的小任务，让大量线程同时做。**

## 4.2 CUDA 程序由什么组成

一个最简单的 CUDA 程序通常包含四部分：

1. 在 CPU 上准备数据
2. 把数据拷到 GPU
3. 在 GPU 上启动 kernel 做计算
4. 把结果拷回 CPU 并检查

你要先建立这个固定流程感。

## 4.3 什么是 host 和 device

- `host`：通常指 CPU 这一侧
- `device`：通常指 GPU 这一侧

所以：
- `host code` 是在 CPU 上运行的普通 C++ 代码
- `device code` 是在 GPU 上运行的代码
- `kernel` 是一种特殊的 device 函数，由 CPU 发起调用，在 GPU 上并行执行

## 4.4 thread、block、grid 是什么

这是 CUDA 第一周最核心的抽象。

### thread
线程是最小执行单元。

在 `vector add` 里，你可以理解为：
- 一个线程处理一个数组元素

### block
很多线程组成一个 block。

你可以把一个 block 理解成：
- 一小组一起工作的线程
- 同一个 block 内的线程可以协作
- 同一个 block 内可以使用 shared memory

### grid
很多 block 组成一个 grid。

一次 kernel 启动会生成一个 grid。

### 关系图

```text
grid
 ├─ block 0
 │   ├─ thread 0
 │   ├─ thread 1
 │   └─ ...
 ├─ block 1
 │   ├─ thread 0
 │   ├─ thread 1
 │   └─ ...
 └─ ...
```

## 4.5 为什么要计算全局索引

你启动 kernel 时会开很多线程，但每个线程必须知道“自己该处理哪一份数据”。

最常见写法是：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

含义是：
- `threadIdx.x`：当前线程在当前 block 里的编号
- `blockIdx.x`：当前 block 在整个 grid 里的编号
- `blockDim.x`：每个 block 里有多少个线程

最终 `idx` 就是这个线程对应的全局编号。

## 5. Week 1 示例代码讲解

工作区里已经生成了一个示例文件：

- `week1_vector_add.cu`

这个示例做的事情非常简单：

```text
c[i] = a[i] + b[i]
```

但它包含了 CUDA 程序的最小闭环：
- host 数据初始化
- device 显存分配
- host 到 device 拷贝
- kernel 启动
- device 到 host 拷贝
- 结果校验

### 代码结构理解顺序

第一次看代码时，建议按这个顺序理解：

1. `main()` 里先看 CPU 侧准备了什么数据
2. 再看 `cudaMalloc` 和 `cudaMemcpy`
3. 再看 `vectorAddKernel<<<...>>>`
4. 最后看 `__global__` 的 kernel 实现

### 这个 kernel 做了什么

```cpp
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

这段代码的逻辑就是：
- 每个线程先算出自己负责的索引 `idx`
- 如果没越界，就计算 `c[idx] = a[idx] + b[idx]`

你可以把它理解成“把 for 循环拆给大量线程去做”。

## 6. 你这周必须亲手完成的练习

### 练习 1：读懂并复述代码流程
要求你用自己的话写下来：
- 哪些代码在 CPU 上执行
- 哪些代码在 GPU 上执行
- 数据什么时候进入 GPU
- kernel 什么时候开始工作

### 练习 2：改数据规模
把：

```cpp
const int n = 1 << 20;
```

改成：
- `1 << 10`
- `1 << 16`
- `1 << 22`

观察程序是否都能正常工作。

### 练习 3：改 block size
把：

```cpp
const int blockSize = 256;
```

改成：
- 128
- 256
- 512

思考两个问题：
- 为什么 `gridSize` 也会跟着变化？
- 为什么最后一个 block 往往需要边界判断？

### 练习 4：故意制造一个错误
你可以试着做下面其中一件：
- 去掉 `if (idx < n)`
- 去掉 `cudaMemcpy(h_c.data(), d_c, ...)`
- 去掉 `cudaDeviceSynchronize()`

然后观察会发生什么，再恢复代码。

目的不是“写坏代码”，而是建立对正确流程的敏感度。

## 7. 你这周应该掌握到什么程度

如果 Week 1 学完，你应该能回答下面这些问题：

1. CUDA 程序里 host 和 device 分别指什么？
2. `__global__` 函数是什么？
3. 为什么 kernel 要开很多线程？
4. `threadIdx.x`、`blockIdx.x`、`blockDim.x` 分别表示什么？
5. 为什么常常要写 `if (idx < n)`？
6. 为什么数据要先拷到 GPU，再把结果拷回来？

如果这些问题你能稳定回答出来，说明你第一周就过关了。

## 8. 建议你本周这样安排

### 第 1 天
- 理解 GPU vs CPU
- 理解 host / device / kernel
- 看懂 `week1_vector_add.cu` 的结构

### 第 2 天
- 重点理解 thread / block / grid
- 手推几个索引计算例子

### 第 3 天
- 尝试修改 `n` 和 `blockSize`
- 自己解释为什么这样改不会影响正确性

### 第 4 天
- 重新独立打一遍代码
- 不看原文件，自己实现一个最小版

### 第 5 天
- 写本周复盘
- 把自己不理解的概念列成问题清单

## 9. Week 1 完成标准

满足下面 4 条，就算这一周达标：

- 你能说清 CUDA 程序的最小执行流程。
- 你能独立解释 thread / block / grid。
- 你能看懂并修改 `week1_vector_add.cu`。
- 你的 CLion 环境已经补齐到可以编译运行 CUDA。

## 10. 你当前最需要做的下一步

不是继续往下学 shared memory，也不是急着做复杂算子。

你现在最优先的是：

1. 在 CLion 里把 Toolchain 切到 Visual Studio / MSVC
2. 用当前仓库里的 `CMakeLists.txt` 重新加载工程
3. 编译并运行 `week1_vector_add`
4. 成功运行后，再继续做 Week 1 练习

等你把 CLion 工具链切好以后，我可以下一步继续帮你：
- 检查 CLion 的 Toolchain 和 CMake 配置
- 带你逐行讲 `week1_vector_add.cu`
- 再给你出一组 Week 1 小练习题和检查题
