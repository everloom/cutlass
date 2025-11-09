# B2B GEMM FP16 Sm80 完整分析和实现

## 概述

这是对 CUTLASS `examples/13_two_tensor_op_fusion/fused_two_gemms_f16_sm80_shmem.cu` 的**完整追踪和扁平化实现**。

## 文件列表

### 分析文档

1. **B2B_GEMM_ANALYSIS.md** - 初步分析
   - 模板参数
   - 指令概述
   - 执行流程

2. **B2B_GEMM_COMPLETE_TRACE.md** - 完整追踪（重要！）
   - 每一层模板的详细展开
   - 所有底层 PTX 指令的定义位置
   - 完整的执行流程和数据流
   - 指令使用统计

3. **B2B_SUMMARY.md** - 总结文档
   - 追踪结果汇总
   - 关键发现
   - 实现注意事项

### 代码实现

4. **b2b_gemm_f16_sm80_complete.cu** - 扁平化实现
   - 包含所有 PTX 指令宏
   - 融合的 B2B GEMM kernel
   - 直接使用底层指令

5. **build_b2b.sh** - 编译脚本

## 追踪到的所有底层指令

### Tensor Core 计算

```ptx
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 
    {%d0, %d1},                    # D[2] - 输出
    {%a0, %a1, %a2, %a3},          # A[4] - A矩阵
    {%b0, %b1},                    # B[2] - B矩阵
    {%c0, %c1};                    # C[2] - 累加器
```

**位置**: `include/cutlass/arch/mma_sm80.h` Line 311  
**使用**: GEMM0 和 GEMM1 的计算核心

### 优化的内存加载

```ptx
# 加载 16x16 矩阵 (4 个 8x8 块)
ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r0, %r1, %r2, %r3}, [%addr];

# 加载 16x8 矩阵 (2 个 8x8 块)
ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%r0, %r1}, [%addr];
```

**位置**: `include/cutlass/arch/memory_sm75.h` Line 131, 107  
**使用**: 从 shared memory 加载数据到寄存器

### 异步内存拷贝

```ptx
# 异步拷贝 16 bytes
cp.async.ca.shared.global [%smem], [%gmem], 16;

# 提交一组拷贝
cp.async.commit_group;

# 等待完成
cp.async.wait_group 1;
```

**位置**: `include/cutlass/arch/memory_sm80.h` Line 131, 436, 446  
**使用**: 从 global memory 预取数据到 shared memory

## 配置参数（完全准确）

### GEMM 0
```
ThreadblockShape: 64 x 64 x 32
WarpShape: 32 x 32 x 32
InstructionShape: 16 x 8 x 16
WarpCount: 2 x 2 x 1 = 4 warps
Threads: 128
MMA per warp per K-tile: 2 * 4 * 2 = 16
```

### GEMM 1
```
ThreadblockShape: 64 x 256 x 32
WarpShape: 64 x 64 x 32
InstructionShape: 16 x 8 x 16
WarpCount: 1 x 4 x 1 = 4 warps
Threads: 128
MMA per warp per K-tile: 4 * 8 * 2 = 64
```

### Shared Memory
```
s_A0: 3 stages x 64x32 FP16 = 12 KB
s_B0: 3 stages x 32x64 FP16 = 12 KB
s_Accum: 64x64 FP16 = 8 KB (GEMM0→GEMM1)
s_B1: 3 stages x 32x256 FP16 = 48 KB
Total: ~80 KB
```

## 编译和运行

```bash
# 编译
./build_b2b.sh

# 运行
./b2b_gemm_f16_sm80
```

**要求**: 
- GPU: A100 (sm_80) 或 RTX 3090 (sm_86)
- CUDA: 11.0+
- 支持 Tensor Core 和 cp.async

## 关键学习点

### 1. Tensor Core 的使用模式

```
1. 使用 cp.async 预取数据到 shared memory
2. 使用 ldmatrix 加载数据到寄存器 (特殊格式)
3. 调用 mma.sync 执行矩阵乘 (一个 warp 协作)
4. 重复步骤 1-3 多次
5. 写回结果
```

### 2. B2B Fusion 的关键

```
传统方式:
GEMM0 → DRAM → GEMM1
耗时: ~200ns 读 + ~200ns 写 = ~400ns

融合方式:
GEMM0 → Shared Memory → GEMM1
耗时: ~20ns 读 + ~20ns 写 = ~40ns

加速: ~10x 内存访问
```

### 3. 3-Stage Pipeline

```
Cycle 0:  Load[0]  |         |
Cycle 1:  Load[1]  | Wait[0] |
Cycle 2:  Load[2]  | Wait[1] | Compute[0]
Cycle 3:  Load[3]  | Wait[2] | Compute[1]
...

隐藏内存延迟，保持计算单元饱和
```

## 性能预期

在 RTX 3090 上:
- GEMM0 (64x64x576): ~0.05 ms
- GEMM1 (64x256x64): ~0.08 ms
- Total fused: ~0.10 ms
- 非融合: ~0.15 ms
- **加速比: ~1.5x**

在 A100 上:
- 预期加速比: ~2-3x

## 总结

这个完整的追踪分析展示了：

✅ **完整的模板调用链** - 从 device API 到 PTX 指令  
✅ **所有底层指令** - mma.sync, ldmatrix, cp.async  
✅ **Tensor Core 工作原理** - 数据格式、布局、协作模式  
✅ **融合技术** - Shared memory staging 避免 DRAM 访问  
✅ **Pipeline 优化** - 3-stage 隐藏延迟  
✅ **生产级特性** - ReLU, scaling, 异步拷贝

这是一个**真实生产环境**中使用的高性能 GEMM kernel 的完整剖析！

