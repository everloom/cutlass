# CUTLASS 模板展开分析 - 完整索引

## 概述

本目录包含了对 CUTLASS 库的两个关键示例的**完整模板追踪分析**和**扁平化实现**：

1. **basic_gemm.cu** - 基础 SGEMM (SIMT 模式，FP32)
2. **fused_two_gemms_f16_sm80_shmem.cu** - 融合 B2B GEMM (Tensor Core 模式，FP16)

## 文件组织

### 📁 基础 SGEMM (SIMT 模式)

#### 分析文档
- **COMPLETE_ANALYSIS.md** - basic_gemm.cu 的完整分析（正确版本）
- **ARCH_TAG_PROOF.md** - ArchTag 默认值的证据
- **INSTRUCTION_USAGE.md** - 指令使用说明
- **TEMPLATE_TO_CUDA_MAPPING.md** - 模板到 CUDA 的映射关系

#### 代码实现
- **sgemm_cutlass_correct.cu** - 正确的扁平化实现
  - 配置: 256 threads, WarpShape 32x64x8
  - 使用指令: fma.rn.f32, ld.global, ld.shared
  - 不使用: mma.sync, ldmatrix, cp.async

#### 构建脚本
- **build_correct.sh** - 编译脚本
- **README_CORRECT.md** - 使用说明

#### 废弃文件（早期错误版本，可忽略）
- ~~sgemm_cutlass_unrolled.cu~~
- ~~build_sgemm_unrolled.sh~~
- ~~README_UNROLLED.md~~
- ~~TEMPLATE_ANALYSIS.md~~

### 📁 融合 B2B GEMM (Tensor Core 模式)

#### 分析文档
- **B2B_GEMM_ANALYSIS.md** - 初步分析和指令介绍
- **B2B_GEMM_COMPLETE_TRACE.md** - 完整追踪（重要！）★★★
  - 所有模板层次的详细展开
  - 每个 PTX 指令的定义和使用位置
  - 完整的执行流程
  - 数据流图
- **B2B_SUMMARY.md** - 总结和学习要点
- **B2B_README.md** - 使用指南和性能分析

#### 代码实现
- **b2b_gemm_f16_sm80_complete.cu** - 扁平化实现★
  - 融合的两个 GEMM
  - 包含所有 PTX 指令宏
  - Shared memory accumulator
  - 3-stage pipeline

#### 构建脚本
- **build_b2b.sh** - 编译脚本

## 快速开始

### 基础 SGEMM (SIMT)

```bash
# 编译
./build_correct.sh

# 运行
./sgemm_cutlass_correct 1024 1024 1024
```

**学习重点**:
- 标量 FMA 指令
- Shared memory tiling
- Double buffering
- Warp 和 thread 组织

### 融合 B2B GEMM (Tensor Core)

```bash
# 编译 (需要 Sm80+ GPU)
./build_b2b.sh

# 运行
./b2b_gemm_f16_sm80
```

**学习重点**:
- Tensor Core `mma.sync` 指令
- `ldmatrix` 优化加载
- `cp.async` 异步拷贝
- Shared memory staging
- Kernel fusion 技术

## 底层指令对比

| 指令类型 | basic_gemm.cu | fused_b2b_gemm.cu | 用途 |
|---------|--------------|-------------------|------|
| **计算指令** | `fma.rn.f32` | `mma.sync.m16n8k16.f16` | 矩阵乘累加 |
| **加载指令** | `ld.shared.f32` | `ldmatrix.x4/x2` | 从 SMEM 加载 |
| **拷贝指令** | `ld.global` + `st.shared` | `cp.async.ca` | GMEM → SMEM |
| **同步指令** | `__syncthreads()` | `cp.async.wait_group` | 同步 |
| **性能** | 基准 | **8-16x 更快** | - |

## 配置对比

### basic_gemm.cu (SIMT)

```
ThreadblockShape: 128 x 128 x 8
WarpShape: 32 x 64 x 8
Threads: 256 (8 warps)
OpClass: OpClassSimt
ArchTag: Sm70 (默认)
Stages: 2
DataType: float (FP32)
```

### fused_b2b_gemm.cu (TensorOp)

```
GEMM0: 64 x 64 x 32
GEMM1: 64 x 256 x 32
WarpShape0: 32 x 32 x 32
WarpShape1: 64 x 64 x 32
Threads: 128 (4 warps)
OpClass: OpClassTensorOp
ArchTag: Sm80
Stages: 3
DataType: half (FP16)
Special: Shared memory accumulator
```

## 模板追踪方法论

### 成功的分析步骤

1. **确定入口模板参数** - 查看用户代码
2. **查找默认参数** - 追踪 Default*Configuration
3. **展开每一层** - Device → Kernel → Threadblock → Warp → Arch
4. **找到 PTX 指令** - arch/*.h 文件中的 asm volatile
5. **追踪使用位置** - grep 搜索指令调用
6. **理解数据流** - 从 GMEM 到 SMEM 到 REG 到 GMEM
7. **验证配置** - 计算 warp count, thread count 等

### 关键教训

❌ **错误**: 假设默认值（如 WarpShape = 64x64）  
✅ **正确**: 查看源码确认每个默认值

❌ **错误**: 忽略架构差异（Sm70 vs Sm80）  
✅ **正确**: 追踪 ArchTag 并理解其影响

❌ **错误**: 简化复杂的索引计算  
✅ **正确**: 完整计算 warp/thread 的位置映射

## 进一步学习

### 1. 深入 Tensor Core

- 阅读 NVIDIA Tensor Core 白皮书
- 理解 wmma API 和 mma.sync 的区别
- 学习 Tensor Core 的数据分布模式

### 2. 深入 cp.async

- 理解异步拷贝的原理
- 学习 pipeline 优化技术
- 掌握 wait_group 的使用时机

### 3. 深入 CUTLASS

- 研究 CuTe (CUTLASS 3.0 的核心)
- 理解 Epilogue fusion
- 学习 Group GEMM 和 Sparse GEMM

## 参考资料

### CUTLASS 源码
```
include/cutlass/gemm/device/gemm.h          - Device API
include/cutlass/gemm/kernel/gemm.h          - Kernel 实现
include/cutlass/gemm/threadblock/           - Threadblock MMA
include/cutlass/gemm/warp/                   - Warp MMA
include/cutlass/arch/                        - PTX 指令

examples/00_basic_gemm/                      - 基础示例
examples/13_two_tensor_op_fusion/            - 融合示例
```

### NVIDIA 文档
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Tensor Core Programming](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/)

### 论文
- "CUTLASS: Fast Linear Algebra in CUDA C++" (2020)
- "Dissecting the Ampere GPU Architecture" (2020)

## 总结

这个分析项目完成了：

✅ 两个完整的 CUTLASS 示例的追踪  
✅ 从最高层 API 到最底层 PTX 指令的完整链路  
✅ 所有关键指令的定义和使用位置  
✅ 扁平化的可执行代码实现  
✅ 详细的文档和学习材料  

这些材料可以帮助你：
1. 理解 CUTLASS 的设计和实现
2. 学习高性能 CUDA 编程技术
3. 掌握 Tensor Core 编程
4. 理解 GEMM 优化的各个层次

**注意**: 生成的扁平化代码是教学实现，展示了核心概念。生产环境请使用 CUTLASS 或 cuBLAS，它们包含更多优化和边界情况处理。

