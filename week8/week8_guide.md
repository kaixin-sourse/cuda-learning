# CUDA C++ Week 8 详细学习资料

## 1. 本周目标

这一周你不再只是“实现某个算法”，而是要完整走一遍优化闭环：

1. 定义一个 baseline。
2. 提出明确优化假设。
3. 实现优化版本。
4. 做正确性验证。
5. 记录计时和加速比。
6. 写出能被数据支持的结论。

本周示例选择 matrix multiplication，因为它能很好体现：

- global memory 访问代价
- shared memory 数据复用
- thread/block 组织方式
- 单线程负责多个输出时的复用思路

## 2. 为什么 baseline 很重要

如果没有 baseline，就没有办法严肃讨论“优化”。

baseline 的作用是：

- 提供一个正确但不一定快的参考版本
- 作为所有优化版本的对照对象
- 帮你避免“我感觉更快了”这种模糊判断

Week 8 里，你要把优化写成表格：

```text
版本 | 输入规模 | 时间 | 相对 baseline 加速比 | 解释
```

## 3. 本周三个版本

## 3.1 Naive matmul

每个线程计算一个 `C[row][col]`。

特点：

- 代码直观
- global memory 读取次数多
- 没有显式数据复用

它是本周 baseline。

## 3.2 Shared memory tiled matmul

优化假设：

**同一个 block 内多个线程会反复使用 A 和 B 的局部区域，所以把 tile 搬进 shared memory 可以减少 global memory 重复访问。**

它会：

- 每轮加载 A 的一个 tile
- 每轮加载 B 的一个 tile
- block 内线程共享这两个 tile
- 用 `__syncthreads()` 保证加载完成后再计算

## 3.3 Tiled two-output matmul

优化假设：

**同一个线程如果计算同一行相邻两个输出，可以复用从 A tile 中读出的值，减少一部分重复读取。**

这个版本让一个线程计算两个相邻列的输出：

- `C[row][col0]`
- `C[row][col1]`

它不是工业级 GEMM，但很适合教学，因为它能展示“单线程负责多个输出”的基本思想。

## 4. 本周代码说明

## `src/matmul_optimization_compare.cu`

程序会做：

- 生成输入矩阵
- 运行 naive 版本
- 运行 tiled 版本
- 运行 two-output tiled 版本
- 验证三个版本结果一致
- 输出计时表格和加速比

你要重点观察：

- tiled 版是否比 naive 快
- two-output 版是否在你的机器上有收益
- 如果某个优化不明显，原因可能是什么

## 5. 本周必须完成的练习

### 练习 1：运行对比程序

记录不同矩阵规模下三个版本的时间。

### 练习 2：改矩阵规模

尝试：

- 256
- 512
- 768

观察加速比是否稳定。

### 练习 3：改 tile size

尝试把 `kTileSize` 从 16 改成 8 或 32。

注意：

- 32x32 block 有 1024 个线程，可能达到设备上限
- tile size 变大不一定更快

### 练习 4：写优化报告

使用 `optimization_report_template.md` 写一页实验报告。

至少包含：

- baseline 是什么
- 做了哪些优化
- 每个优化的假设
- 表格结果
- 最终结论

## 6. 本周完成标准

- 能定义 baseline
- 能完成至少 2 个优化版本
- 能验证结果正确
- 能输出时间和加速比
- 能写出不是“凭感觉”的优化结论

## 7. 本周最该记住的一句话

**优化不是改代码，而是“提出假设 -> 实现 -> 测量 -> 解释”的闭环。**
