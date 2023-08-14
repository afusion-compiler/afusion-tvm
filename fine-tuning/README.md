# Fine-Tuning
| [relax]()

## schedule
### Transpose
1. 从 PrimFunc 中找到转置的 block。
2. 将迭代轴 i, j 分别 split 为多级循环,并绑定到 thread/block 索引上。
3. 将计算提前到 shared cache_read block 中,并在内层循环后面做向量化。
4. 对外层块索引循环 bi 做 unroll 优化。
5. 如果存在前导块,内联计算提高性能。

### Reduction
| softmax, layer norm, RMS norm

1. 对 PrimFunc 进行 normalize,提取 BlockInfo。
2. 分析每个 Block 的循环类型,检查其正确性。
3. 获取空间循环数量 num_leading_s 和归约循环数量 num_trailing_r。
4. 生成 outer block,将空间循环 fuse 到 bx,分割最后一个循环到 tx。
5. 对于内部块:
- 将 Buffer 置于 shared scope 内
- 计算至 outer block 的 bx 循环
- 将归约循环 fuse 到 tx
6. 对 bx 循环进行 unroll
7. 返回生成的 Schedule

### DecodeGEMV(密集矩阵乘以密集向量)
| y = alpha * A * x + beta * y

1. 对 PrimFunc 进行标准化转换,提取出相关的 BlockInfo。
2. 检查 reduction block 的正确性。
3. 将空间循环和 reduction 循环进行归一化,获取 is_inner_reduction 和 c_factor 信息。
4. 根据 is_inner_reduction 的不同,生成不同的 Schedule:
- 如果 reduction 在内层,调用 _sch_inner_reduction 生成 Schedule。
- 如果 reduction 不在内层,调用 _sch_inner_spatial 生成 Schedule。
5. _sch_inner_reduction 主要步骤:
- 将 reduction 轴 split 并unroll
- 生成 rf block 进行 reduction factor 出
- 添加复用写回块
- 处理 epilogue
6. _sch_inner_spatial 主要步骤:
- 将迭代轴 split 并 bind 到 threadIdx
- 生成 rf block 进行 reduction factor 出
- 添加复用写回块
- 处理 epilogue
7. 返回生成的 Schedule。

### fallback(default)
1. 尝试通过try_inline进行所有的算子内联。
2. 对剩余的每个block:
- 提取空间循环(s_loops)、reduction循环(r_loops)、其他循环(o_loops)
- 重新排序循环为 s_loops -> r_loops -> o_loops
- 将s_loops fuse并split成bx(blockIdx)和tx(threadIdx)
- 为bx和tx绑定线程/块索引
- 如果有reduction循环,记录下来待后处理
3. 对记录的reduction block,调用decompose_reduction进行reduce factor提取。
4. 这样就生成了一个默认的GPU block/thread schedule。