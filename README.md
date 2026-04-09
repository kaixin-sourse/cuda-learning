# CUDA Learning

这个仓库用于系统学习 CUDA C++，按照“每周一个主题 + 后续独立项目”的方式组织。

## 目录结构

- `study.md`：12 周总学习计划
- `week1/`：第 1 周内容，包含讲义、示例代码和可运行工程
- `week2/`：第 2 周骨架目录
- `week3/`：第 3 周骨架目录
- `week4/`：第 4 周内容，reduction 与常见性能问题
- `week5/`：第 5 周内容，scan 与 histogram
- `week6/`：第 6 周内容，原子操作、top-k 与 CPU-GPU 对比
- `week7/`：第 7 周内容，profiling 工具入门与分析目标程序
- `projects/`：后续独立项目，例如 CT 成像或医学影像相关实验

## 使用方式

当前可直接运行的示例在 `week1/` 中。后续每一周都建议遵循同样结构：

- `README.md`：本周主题说明
- `CMakeLists.txt`：本周构建入口
- `src/`：本周 CUDA 源码
