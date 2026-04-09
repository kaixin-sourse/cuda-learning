# Week 6

## 主题

- 原子操作与 contention
- 两阶段 top-k 思路
- CPU 与 GPU 的边界

## 当前文件

- `week6_guide.md`：Week 6 详细讲义
- `CMakeLists.txt`：Week 6 工程配置
- `src/atomic_histogram_compare.cu`：global atomic 与 shared atomic histogram 对比
- `src/topk_two_stage_demo.cu`：两阶段 top-k 示例与 CPU-GPU 总耗时对比

## 运行方式

在 CLion 中选择本目录下的 `CMakeLists.txt`，然后运行：

- `week6_atomic_histogram_compare`
- `week6_topk_two_stage_demo`
