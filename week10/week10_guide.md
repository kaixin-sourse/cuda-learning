# CUDA C++ Week 10 详细学习资料

## 1. 本周目标

这一周的重点不是再单独学习某一个 CUDA 技巧，而是把前面学过的内容整理成一个“小型工程”。

你要完成的目标是：

1. 把 `vector add`、`reduction`、`scan`、`matmul` 放到同一个工程里。
2. 每个模块都有正确性校验。
3. 每个模块都有基础 benchmark。
4. 输出格式统一，便于对比和记录。

这周你要开始从“写练习代码”过渡到“组织可复用 CUDA 工程”。

## 2. 为什么要做 mini library

前 9 周你已经写过很多独立示例：

- vector add
- reduction
- scan
- matmul
- histogram
- stream overlap
- profiling target

这些代码如果一直散落在不同文件里，会出现几个问题：

- 很难复用
- 很难统一测试
- 很难统一 benchmark
- 很难作为后续项目基础

mini library 的意义就是：

**把零散练习整理成一个可维护、可测试、可对比的小工程。**

## 3. 本周工程结构

本周示例文件是：

```text
src/mini_library_benchmark.cu
```

它包含 4 个模块：

- `vector add`
- `reduction`
- `exclusive scan`
- `tiled matmul`

每个模块都包含：

- CUDA kernel
- host 端数据准备
- 结果校验
- 基础计时
- 统一输出

## 4. 正确性测试和 benchmark 要分开想

你要养成一个工程习惯：

- 正确性验证回答“结果对不对”
- benchmark 回答“速度怎么样”

这两个问题不要混在一起。

例如：

- vector add 可以全量对拍
- reduction 可以和 CPU sum 对比
- scan 可以和 CPU exclusive scan 对比
- matmul 可以和 CPU matmul 或 baseline GPU matmul 对比

## 5. 本周必须完成的练习

### 练习 1：运行 mini library benchmark

运行 `week10_mini_library_benchmark`，记录每个模块的：

- 输入规模
- 时间
- 正确性结果

### 练习 2：改一个模块参数

可以选择：

- 修改 vector add 的元素数量
- 修改 reduction 的 block size
- 修改 scan 输入数据
- 修改 matmul 的矩阵规模

观察结果和时间变化。

### 练习 3：补一个小模块

建议你后续尝试补一个模块，例如：

- scale
- axpy
- histogram
- simple stencil

并按照同样格式输出。

### 练习 4：写 mini library 总结

使用 `mini_library_report_template.md` 写一页总结。

## 6. 本周完成标准

- mini library 能独立编译运行
- 至少包含 vector add、reduction、scan、matmul
- 每个模块都有正确性校验
- benchmark 输出格式统一
- 你能说清楚代码结构为什么这样组织

## 7. 本周最该记住的一句话

**Week 10 的重点是工程闭环：代码结构、正确性测试、benchmark 和可复用性。**
