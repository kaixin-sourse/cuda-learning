# CUDA C++ Week 6 详细学习资料

## 1. 本周目标

这一周你要开始接触两类更“工程化”的现实问题：

1. 多线程同时更新共享结果时怎么办。
2. 更复杂任务为什么不一定适合直接、粗暴地搬到 GPU 上。

本周重点有 4 个：

1. 理解 atomic operation 的作用和代价。
2. 理解 contention 为什么会让 atomic 方案变慢。
3. 理解 top-k / 排序类任务为什么更难并行化。
4. 通过 CPU-GPU 对比建立“GPU 不一定更快”的判断意识。

学完这一周，你至少应该能做到：

- 解释什么时候需要 atomic。
- 解释为什么 atomic 过多会导致严重竞争。
- 看懂一个两阶段 top-k 思路。
- 用实际计时总结一个“GPU 不一定更快”的例子。

## 2. 本周核心概念

## 2.1 什么是 atomic operation

atomic operation 的意义是：

- 多个线程可能同时修改同一块数据
- 你希望这次修改不会互相覆盖
- 更新过程必须是“不可分割”的

最常见例子就是计数：

```cpp
atomicAdd(&counter, 1);
```

如果不用 atomic，多线程同时做 `counter += 1`，结果可能丢失更新。

## 2.2 为什么 atomic 会慢

atomic 并不是“免费线程安全”。

一旦很多线程同时更新同一个地址，就会产生 contention：

- 大家都在抢同一份数据
- 更新无法真正并行展开
- 硬件需要序列化这些访问

所以本周你要开始形成一个意识：

**atomic 解决的是正确性问题，不保证高性能。**

## 2.3 histogram 为什么能体现 contention

如果输入分布很偏，比如很多值都落在同一个 bin 上，那么大量线程会同时对同一个 bin 做 `atomicAdd`。

这时就会出现明显竞争。

因此 histogram 是 Week 6 很好的教学例子，因为它能直接展示：

- global atomic 的直接写法
- shared memory 局部统计再合并的改进思路
- contention 对时间的影响

## 2.4 top-k 为什么更难并行

像 vector add、matmul 这类问题，数据并行结构很直接：

- 每个元素或每个输出位置可以比较自然地分配给线程

但 top-k / 排序更复杂，因为它们涉及：

- 全局顺序关系
- 元素之间大量比较
- 多轮重排
- 结果不是“每个线程独立算一个输出”这么简单

所以 Week 6 你先不要追求“工业级高性能 top-k”，而是先理解一个**分阶段方案**：

1. 每个 block 先筛出自己的局部 top-k 候选
2. 再把所有候选合并成全局 top-k

这能帮助你建立 top-k 的基本分治思路。

## 2.5 CPU 与 GPU 的边界

这周最关键的工程观念是：

**不是所有问题搬到 GPU 都会更快。**

例如：

- 数据量小
- 算法结构不够规则
- 线程间依赖强
- 大量原子竞争
- 数据搬运开销很大

这些都会让 GPU 的优势变小，甚至让 CPU 更合适。

## 3. 本周代码说明

## 3.1 `src/atomic_histogram_compare.cu`

这个程序实现了三种 histogram 统计路径：

1. CPU histogram
2. GPU global atomic histogram
3. GPU shared atomic histogram

它会做：

- 生成带偏斜分布的数据
- 分别统计 CPU 与 GPU 结果
- 对拍正确性
- 打印时间

这个例子重点不是“谁绝对更快”，而是让你观察：

- global atomic 在高 contention 下会有什么表现
- shared memory 局部统计为什么常常更合理

## 3.2 `src/topk_two_stage_demo.cu`

这个程序实现的是一个教学版两阶段 top-k：

1. 每个 block 从自己的数据块里挑出局部 top-k 候选
2. host 端把候选再做一次合并，得到最终 top-k

它会做：

- CPU 参考 top-k
- GPU 两阶段 top-k
- CPU 与 GPU 总耗时对比
- 结果校验

这个例子要传达的不是“这就是最强 top-k”，而是：

**top-k 这类问题常常需要分阶段拆解，而不是一个简单 kernel 直接做完。**

## 4. 本周必须完成的练习

### 练习 1：运行 histogram 对比程序

观察：

- CPU 时间
- GPU global atomic 时间
- GPU shared atomic 时间

然后回答：

- 为什么两种 GPU 方案都正确
- 为什么它们时间会不同

### 练习 2：修改输入分布

把 histogram 输入分布改得更平均，或者更偏向单一 bin，观察 contention 强弱变化。

### 练习 3：运行 top-k 示例

观察：

- CPU 总时间
- GPU 总时间
- 结果是否一致

然后思考：

- 为什么 GPU 两阶段 top-k 不一定比 CPU 快
- 开销主要花在了哪里

### 练习 4：修改数据规模

你可以尝试：

- `1 << 14`
- `1 << 18`
- `1 << 20`

观察不同规模下 CPU/GPU 边界是否变化。

## 5. 本周完成标准

- 能解释 atomic 的作用
- 能解释 contention 的含义
- 能说清一个两阶段 top-k 思路
- 能举出一个 GPU 不一定更快的具体实验结果

## 6. 本周最该记住的一句话

**Week 6 的关键不是“把所有东西都并行化”，而是学会判断：哪些任务值得放到 GPU，哪些代价可能抵消收益。**
