# Week 7

## 主题

- Nsight Systems 基础使用
- Nsight Compute 基础使用
- timeline、occupancy、memory throughput 的初步理解

## 当前文件

- `week7_guide.md`：Week 7 详细讲义
- `profile_notes_template.md`：profiling 笔记模板
- `profile_commands_windows.md`：Windows 下的命令示例
- `CMakeLists.txt`：Week 7 工程配置
- `src/profile_reduction_target.cu`：用于 Nsight Systems / Nsight Compute 的 reduction target
- `src/profile_matmul_target.cu`：用于 profiling 的 tiled matmul target

## 运行方式

在 CLion 中选择本目录下的 `CMakeLists.txt`，然后运行：

- `week7_profile_reduction`
- `week7_profile_matmul`
