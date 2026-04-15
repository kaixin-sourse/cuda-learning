# Week 9

## 主题

- CUDA streams
- pageable memory 与 pinned memory
- data transfer 和 kernel execution overlap
- 多阶段 pipeline 的吞吐优化

## 当前文件

- `week9_guide.md`：Week 9 详细讲义
- `stream_experiment_notes_template.md`：stream 实验记录模板
- `CMakeLists.txt`：Week 9 工程配置
- `src/stream_overlap_compare.cu`：pageable / pinned / multi-stream pipeline 对比

## 运行方式

在 CLion 中选择本目录下的 `CMakeLists.txt`，然后运行：

- `week9_stream_overlap_compare`
