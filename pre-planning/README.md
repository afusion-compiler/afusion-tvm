# Pre-Planning
| [ref](collage)

解决图层级的融合问题，使用AOT的library划分图

1. CollagePartition接受CompilationConfig和CostEstimator作为输入参数。
2. 内部定义了一个pass_func函数,它接受IRModule和PassContext作为参数。
3. pass_func首先从CompilationConfig中收集PartitionSpec。
4. 然后针对输入模块中的每一个函数:
- 检索函数的虚拟设备(virtual device)映射信息
- 用收集到的PartitionSpec、CostEstimator和虚拟设备映射构造Partitioner对象
- 用Partitioner的Partition()方法对函数进行分区
- 用分区后的新函数替换模块中的原函数
- 在所有函数分区完成后,调用OutlineCompilerFunctions pass进一步优化
CollagePartition通过创建并返回一个ModulePass,将pass_func封装成一个TVM pass