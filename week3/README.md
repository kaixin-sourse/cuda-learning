# Week 3

## 主题

- shared memory 的基本用途
- block 内同步与 `__syncthreads()`
- 入门版 tiled matrix multiplication

## 当前文件

- `week3_guide.md`：Week 3 详细讲义
- `CMakeLists.txt`：Week 3 工程配置
- `src/shared_memory_demo.cu`：shared memory 与 block 内同步示例
- `src/matmul_compare.cu`：朴素矩阵乘法与 tiled 矩阵乘法对比

## 运行方式

在 CLion 中选择本目录下的 `CMakeLists.txt`，然后运行：

- `week3_shared_memory_demo`
- `week3_matmul_compare`
