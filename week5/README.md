# Week 5

## 主题

- inclusive scan 与 exclusive scan
- Blelloch scan 的上扫 / 下扫思路
- scan 在 histogram 等数据重排任务中的作用

## 当前文件

- `week5_guide.md`：Week 5 详细讲义
- `CMakeLists.txt`：Week 5 工程配置
- `src/blelloch_scan_demo.cu`：单 block Blelloch exclusive scan 示例
- `src/histogram_shared_atomic.cu`：shared memory + atomic 的 histogram 示例

## 运行方式

在 CLion 中选择本目录下的 `CMakeLists.txt`，然后运行：

- `week5_blelloch_scan_demo`
- `week5_histogram_shared_atomic`
