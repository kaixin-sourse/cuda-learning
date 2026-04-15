# Week 8

## 主题

- baseline 的定义方式
- 有假设、有验证、有结论的优化闭环
- matrix multiplication 的多版本优化对比

## 当前文件

- `week8_guide.md`：Week 8 详细讲义
- `optimization_report_template.md`：优化实验报告模板
- `CMakeLists.txt`：Week 8 工程配置
- `src/matmul_optimization_compare.cu`：朴素版、shared memory tiled 版、双输出 tiled 版 matmul 对比

## 运行方式

在 CLion 中选择本目录下的 `CMakeLists.txt`，然后运行：

- `week8_matmul_optimization_compare`
