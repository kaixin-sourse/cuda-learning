# CUDA C++ Week 11 详细学习资料

## 1. 本周目标

这一周你要把前面学过的 CUDA 技能迁移到一个更接近医学影像和张量处理的问题上：2D 图像卷积。

本周重点有 4 个：

1. 理解图像/张量在内存中的二维布局。
2. 理解 stencil 类操作的局部访存特征。
3. 正确处理图像边界。
4. 对比 CPU baseline、CUDA global memory 版本、CUDA shared memory 版本。

学完这一周，你至少应该能做到：

- 写出一个 CPU 版 3x3 图像卷积。
- 写出一个 CUDA 版 3x3 图像卷积。
- 用 CPU 结果对拍 CUDA 结果。
- 解释这个任务和 CT 成像/模型输入预处理之间的关系。

## 2. 为什么选择 2D 图像卷积

CT 成像、医学图像处理和深度学习训练前处理里经常会出现类似操作：

- 图像滤波
- 平滑/去噪
- 边缘增强
- 插值前后的局部处理
- 投影数据或重建图像的局部 stencil 计算

3x3 卷积虽然简单，但它具备很多真实任务的共性：

- 输入是二维图像
- 每个输出像素依赖邻域
- 边界需要特殊处理
- 相邻线程会访问重叠数据
- shared memory 有潜在价值

所以这是从通用 CUDA 走向医学影像 CUDA 的合适入口。

## 3. 本周项目结构

示例文件：

```text
src/medical_image_convolution.cu
```

它包含：

- synthetic CT-like image 生成
- CPU 3x3 convolution baseline
- CUDA global memory convolution
- CUDA shared memory convolution
- 正确性校验
- 不同尺寸输入的计时对比

## 4. 核心概念

## 4.1 图像内存布局

二维图像通常在内存中按一维数组存放：

```cpp
image[y * width + x]
```

所以 CUDA kernel 中常见索引是：

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

## 4.2 stencil 操作

3x3 卷积就是典型 stencil：

每个输出像素会读取周围 3x3 邻域。

这意味着相邻线程读取的数据高度重叠，因此 shared memory 可以用于缓存 tile 和 halo。

## 4.3 边界处理

本周示例使用 clamp 边界：

- 左边界越界时使用最左列
- 右边界越界时使用最右列
- 上下边界同理

这种方式简单稳定，适合学习阶段。

## 4.4 shared memory tile + halo

如果 block 大小是 16x16，3x3 卷积需要额外一圈 halo。

shared tile 大小就是：

```text
(16 + 2) x (16 + 2)
```

block 内线程先共同加载 tile 和 halo，再同步，然后每个线程从 shared memory 中计算自己的输出。

## 5. 本周必须完成的练习

### 练习 1：运行项目

运行 `week11_medical_image_convolution`，确认所有尺寸的结果都是 PASS。

### 练习 2：解释三种版本

回答：

- CPU baseline 做了什么
- CUDA global 版本为什么正确
- CUDA shared 版本为什么要加载 halo

### 练习 3：修改图像尺寸

尝试：

- 512 x 512
- 1024 x 1024
- 1536 x 1536

观察 CPU 和 GPU 时间变化。

### 练习 4：修改卷积核

把平滑核改成边缘增强核或锐化核，确认 CPU/GPU 仍然一致。

### 练习 5：写项目说明

使用 `medical_image_project_report_template.md` 写一页说明：

- 任务是什么
- 为什么贴近医学影像
- 数据布局是什么
- 下一步可以怎么优化

## 6. 本周完成标准

- 项目能正确运行
- CPU/GPU 对拍通过
- 能解释 stencil 和 halo
- 能指出后续优化方向，例如 separable filter、texture memory、3D volume、multi-stream pipeline

## 7. 本周最该记住的一句话

**医学影像 CUDA 项目的起点，是把二维/三维数据布局、邻域访问和边界处理先写正确。**
