# fused_two_gemms_f16_sm80_shmem.cu 完整分析

## 任务说明

这个文档追踪 `fused_two_gemms_f16_sm80_shmem.cu` 中所有模板调用，一直到最底层的 PTX 指令：
- `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
- `ldmatrix.sync.aligned.x4.m8n8.shared.b16`
- `cp.async.cg.shared.global`

## 1. 入口模板参数

```cpp
// Line 167-187
using B2bGemm = cutlass::gemm::device::B2bGemm<
    cutlass::half_t,                          // ElementA
    cutlass::layout::RowMajor,                // LayoutA
    cutlass::half_t,                          // ElementB
    cutlass::layout::ColumnMajor,             // LayoutB
    cutlass::half_t,                          // ElementC (output)
    cutlass::layout::RowMajor,                // LayoutC
    cutlass::half_t,                          // ElementAccumulator
    cutlass::arch::OpClassTensorOp,           // ← Tensor Core!
    cutlass::arch::Sm80,                      // ← Ampere!
    cutlass::gemm::GemmShape<64, 64, 32>,     // ThreadblockShape0
    cutlass::gemm::GemmShape<64, 256, 32>,    // ThreadblockShape1
    cutlass::gemm::GemmShape<32, 32, 32>,     // WarpShape0
    cutlass::gemm::GemmShape<64, 64, 32>,     // WarpShape1
    cutlass::gemm::GemmShape<16, 8, 16>,      // InstructionShape (mma.m16n8k16)
    EpilogueOutputOp0,                        // LinearCombinationRelu
    EpilogueOutputOp1,                        // LinearCombinationRelu
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3,                                        // Stages (3-stage pipeline)
    true                                      // SmemAccumulator
>;
```

## 2. 关键配置参数

### GEMM 0 (第一个矩阵乘法)
- **ThreadblockShape**: 64 x 64 x 32
- **WarpShape**: 32 x 32 x 32
- **InstructionShape**: 16 x 8 x 16 (Tensor Core mma.m16n8k16)
- **Warps per block**: (64/32) x (64/32) x (32/32) = 2 x 2 x 1 = **4 warps**
- **Threads per block**: 4 x 32 = **128 threads**

### GEMM 1 (第二个矩阵乘法)
- **ThreadblockShape**: 64 x 256 x 32
- **WarpShape**: 64 x 64 x 32
- **InstructionShape**: 16 x 8 x 16
- **Warps per block**: (64/64) x (256/64) x (32/32) = 1 x 4 x 1 = **4 warps**
- **Threads per block**: 4 x 32 = **128 threads**

### Tensor Core 参数
- **Instruction**: `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
- **每个 warp**: 32 threads
- **每个 instruction**: 处理 16x8 输出，消耗 16 个 K
- **数据类型**: FP16 输入和累加

### Pipeline 参数
- **Stages**: 3 (三级流水线)
- **Shared memory buffering**: 3 x (ThreadblockTile_A + ThreadblockTile_B)

## 3. DefaultGemmConfiguration for Sm80 TensorOp FP16

位置：`include/cutlass/gemm/device/default_gemm_configuration.h`

```cpp
template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct DefaultGemmConfiguration<
    arch::OpClassTensorOp,   // ← TensorOp
    arch::Sm80,              // ← Ampere
    ElementA, ElementB, ElementC, ElementAccumulator> {
    
    static int const kAlignmentA = 128 / sizeof_bits<ElementA>::value;  // 128/16 = 8
    static int const kAlignmentB = 128 / sizeof_bits<ElementA>::value;  // 8
    
    using ThreadblockShape = GemmShape<128, 256, 64>;
    using WarpShape = GemmShape<64, 64, 64>;
    using InstructionShape = GemmShape<16, 8, 16>;  // ← m16n8k16
    static int const kStages = 3;  // ← 3 stages for Sm80
    
    using EpilogueOutputOp = epilogue::thread::LinearCombination<...>;
    
    using Operator = arch::OpMultiplyAdd;
};
```

**注意**: 实际代码中使用了自定义的 ThreadblockShape 和 WarpShape，覆盖了默认值。

## 4. DefaultMmaCore for Sm80 TensorOp

位置：`include/cutlass/gemm/threadblock/default_mma_core_sm80.h`

关键计算（以 GEMM0 为例）：

```cpp
ThreadblockShape = GemmShape<64, 64, 32>
WarpShape = GemmShape<32, 32, 32>
InstructionShape = GemmShape<16, 8, 16>

// Warp 数量
WarpCount = GemmShape<
    64 / 32,   // M: 2
    64 / 32,   // N: 2
    32 / 32    // K: 1
>;
// Total: 2 x 2 x 1 = 4 warps

// Threads
kThreads = 4 * 32 = 128

// Warp 内的 MMA 迭代次数
MmaIterationsM = WarpShape::kM / InstructionShape::kM  // 32 / 16 = 2
MmaIterationsN = WarpShape::kN / InstructionShape::kN  // 32 / 8 = 4
MmaIterationsK = WarpShape::kK / InstructionShape::kK  // 32 / 16 = 2

// 每个 warp 执行: 2 x 4 x 2 = 16 个 mma.sync 指令
```

## 5. Tensor Core MMA 指令

### 5.1 mma.sync 指令格式

```cpp
// 位置: include/cutlass/arch/mma_sm80.h

template <>
struct Mma<
    gemm::GemmShape<16, 8, 16>,  // Shape
    32,                          // kThreads
    half_t,                      // ElementA
    layout::RowMajor,            // LayoutA
    half_t,                      // ElementB
    layout::ColumnMajor,         // LayoutB
    float,                       // ElementC
    layout::RowMajor,            // LayoutC
    OpMultiplyAdd> {
    
    using Shape = gemm::GemmShape<16, 8, 16>;
    using ElementA = half_t;
    using FragmentA = Array<half_t, 4>;  // 每个线程 4 个 half
    using FragmentB = Array<half_t, 2>;  // 每个线程 2 个 half
    using FragmentC = Array<float, 4>;   // 每个线程 4 个 float (或 2 个 half)
    
    CUTLASS_HOST_DEVICE
    void operator()(
        FragmentC &d,
        FragmentA const &a,
        FragmentB const &b,
        FragmentC const &c
    ) const {
        // PTX 指令
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0, %1}, "              // D (输出, 4个寄存器)
            "{%2, %3, %4, %5}, "      // A (输入, 4个寄存器)
            "{%6, %7}, "              // B (输入, 2个寄存器)
            "{%8, %9};\n"             // C (累加器, 4个寄存器)
            : "=r"(d[0]), "=r"(d[1])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(c[0]), "r"(c[1])
        );
    }
};
```

### 5.2 mma.sync 的数据分布

对于 `mma.m16n8k16.f16`:
- **线程参与**: 一个 warp (32 threads)
- **输出**: 16 x 8 = 128 个 FP16 结果
- **每个线程**: 128 / 32 = 4 个输出元素
- **矩阵 A**: 16 x 16，每个线程 4 个元素
- **矩阵 B**: 16 x 8，每个线程 2 个元素

线程到元素的映射（simplified）:
```
Thread layout in warp (for output):
  lane_id  |  Output positions
  ---------|------------------
    0-3    |  row 0, cols 0-3 and row 8, cols 0-3
    4-7    |  row 1, cols 0-3 and row 9, cols 0-3
    ...
```

## 6. ldmatrix 指令

### 6.1 ldmatrix 用途

`ldmatrix` 指令专门用于从 shared memory 加载数据到寄存器，以供 Tensor Core 使用。

位置: `include/cutlass/arch/memory_sm75.h` (Sm75+都支持)

```cpp
template <int SizeInBytes>
CUTLASS_DEVICE
void ldsm(void *D, void const *ptr) {
    unsigned addr = cutlass::arch::cutlass_get_smem_pointer(ptr);
    
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(addr)
    );
    #endif
}
```

### 6.2 ldmatrix 变体

```cpp
// x1: 加载 1 个 8x8 矩阵 (1个寄存器)
ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%r0}, [%addr];

// x2: 加载 2 个 8x8 矩阵 (2个寄存器)
ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%r0, %r1}, [%addr];

// x4: 加载 4 个 8x8 矩阵 (4个寄存器)
ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r0, %r1, %r2, %r3}, [%addr];

// .trans: 转置加载
ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {...}, [%addr];
```

对于 `mma.m16n8k16`：
- **A 矩阵** (16x16): 需要 `ldmatrix.x4` (4个8x8块)
- **B 矩阵** (16x8): 需要 `ldmatrix.x2` (2个8x8块)

## 7. cp.async 指令

### 7.1 cp.async 用途

异步地从 global memory 拷贝数据到 shared memory，与计算重叠。

位置: `include/cutlass/arch/memory_sm80.h`

```cpp
template <int SizeInBytes>
struct cp_async<SizeInBytes, CacheOperation::Always> {
    
    CUTLASS_DEVICE
    static void copy(
        void *smem_ptr,
        void const *global_ptr,
        bool mask = true) {
        
        unsigned smem_addr = cutlass_get_smem_pointer(smem_ptr);
        
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %0, 0;\n"
            "@p cp.async.ca.shared.global [%1], [%2], %3;\n"
            "}\n" ::"r"((int)mask),
            "r"(smem_addr),
            "l"(global_ptr),
            "n"(SizeInBytes));
        #endif
    }
};
```

### 7.2 cp.async 控制指令

```cpp
// 提交当前组的所有 cp.async
CP_ASYNC_COMMIT_GROUP()
→ asm volatile("cp.async.commit_group;\n" ::)

// 等待所有 cp.async 完成
CP_ASYNC_WAIT_ALL()
→ asm volatile("cp.async.wait_all;\n" ::)

// 等待除了最近 N 组外的所有 cp.async 完成
CP_ASYNC_WAIT_GROUP(N)
→ asm volatile("cp.async.wait_group %0;\n" ::"n"(N))
```

### 7.3 3-Stage Pipeline 使用

```cpp
// Prologue: 预加载前两个 tiles
for (int k = 0; k < 2; ++k) {
    cp.async tile[k] from GMEM to SMEM[k]
    CP_ASYNC_COMMIT_GROUP()
}

// Mainloop
for (int k = 2; k < num_k_tiles; ++k) {
    CP_ASYNC_WAIT_GROUP(1)  // 等待 tile[k-2] 完成
    __syncthreads()
    
    // 计算 tile[k-2]
    compute_mma(tile[k-2])
    
    // 异步加载 tile[k]
    cp.async tile[k] from GMEM to SMEM[k % 3]
    CP_ASYNC_COMMIT_GROUP()
}

// Epilogue: 处理最后两个 tiles
CP_ASYNC_WAIT_GROUP(0)
__syncthreads()
compute_mma(tile[num_k_tiles-2])
compute_mma(tile[num_k_tiles-1])
```

## 8. B2B GEMM with Shared Memory Accumulator

### 8.1 架构

```
┌─────────────────────────────────────────────────┐
│            Thread Block                         │
│                                                 │
│  ┌──────────────┐      ┌──────────────┐        │
│  │   GEMM 0     │      │   GEMM 1     │        │
│  │              │      │              │        │
│  │  A0 × B0     │──┐   │  Temp × B1   │        │
│  │              │  │   │              │        │
│  │  Output: Temp│  │   │  Output: D1  │        │
│  └──────────────┘  │   └──────────────┘        │
│                    │                            │
│              Shared Memory                      │
│         ┌──────────────────┐                    │
│         │   Accumulator    │◄───┘               │
│         │   (Temp matrix)  │                    │
│         │   64 x 32 FP16   │                    │
│         └──────────────────┘                    │
└─────────────────────────────────────────────────┘
```

### 8.2 执行流程

```cpp
// 同一个 thread block 执行两个 GEMM

// GEMM 0: A0 × B0 → Temp (存储在 shared memory)
for k_tile in K0:
    ldmatrix A0_tile, B0_tile from shared memory
    mma.sync ... (累加到寄存器)

// Epilogue 0: 寄存器 → Shared Memory
// 使用特殊的 epilogue 写入 shared memory 而不是 global memory
store_to_shared_memory(accum0, shared_accum)

__syncthreads()

// GEMM 1: Temp × B1 → D1 (输出到 global memory)
for k_tile in K1:
    ldmatrix Temp_tile from shared memory  // ← 从 shared memory 读取 GEMM0 的结果!
    ldmatrix B1_tile from shared memory
    mma.sync ... (累加到寄存器)

// Epilogue 1: 寄存器 → Global Memory
store_to_global_memory(accum1, D1)
```

## 9. Warp-Level MMA 实现

位置: `include/cutlass/gemm/warp/mma_tensor_op.h`

```cpp
template <
    typename Shape_,           // 32x32x32 for GEMM0
    typename ElementA_,        // half_t
    typename LayoutA_,         // RowMajor
    typename ElementB_,        // half_t
    typename LayoutB_,         // ColumnMajor
    typename ElementC_,        // half_t
    typename LayoutC_,         // RowMajor
    typename Policy_>
class MmaTensorOp {
public:
    using Shape = Shape_;
    using Policy = Policy_;
    
    // MMA 迭代次数
    static int const kMmaIterationsM = Shape::kM / Policy::Operator::Shape::kM;  // 32/16=2
    static int const kMmaIterationsN = Shape::kN / Policy::Operator::Shape::kN;  // 32/8=4
    static int const kMmaIterationsK = Shape::kK / Policy::Operator::Shape::kK;  // 32/16=2
    
    using MmaOperator = typename Policy::Operator;  // mma.m16n8k16
    
    CUTLASS_DEVICE
    void operator()(
        FragmentC &D,
        FragmentA const &A,
        FragmentB const &B,
        FragmentC const &C) {
        
        // 执行多个 mma.sync 指令
        #pragma unroll
        for (int k = 0; k < kMmaIterationsK; ++k) {
            #pragma unroll
            for (int m = 0; m < kMmaIterationsM; ++m) {
                #pragma unroll
                for (int n = 0; n < kMmaIterationsN; ++n) {
                    
                    // 调用底层的 mma.sync 指令
                    MmaOperator mma_op;
                    mma_op(
                        D[m][n],      // 输出 fragment
                        A[m][k],      // A fragment
                        B[k][n],      // B fragment
                        D[m][n]       // 累加器
                    );
                    // ↑ 这里会展开为实际的 PTX mma.sync 指令
                }
            }
        }
    }
};
```

## 10. 完整的数据流

### 10.1 Global Memory → Shared Memory

```cpp
// 使用 cp.async (Sm80)
cp.async.ca.shared.global [smem_A], [gmem_A], 16;  // 16 bytes
cp.async.ca.shared.global [smem_B], [gmem_B], 16;
cp.async.commit_group;

// 或者使用普通加载 (Sm75)
// ldgsts: ld.global + st.shared
ld.global.f16x8 {data}, [gmem_A];
st.shared.f16x8 [smem_A], {data};
```

### 10.2 Shared Memory → Registers (ldmatrix)

```cpp
// 加载 A tile (16x16 需要 4 个 8x8)
ldmatrix.sync.aligned.x4.m8n8.shared.b16 
    {%r0, %r1, %r2, %r3}, [smem_A_addr];

// 加载 B tile (16x8 需要 2 个 8x8)
ldmatrix.sync.aligned.x2.m8n8.shared.b16 
    {%r4, %r5}, [smem_B_addr];
```

### 10.3 Tensor Core Computation (mma.sync)

```cpp
// 执行矩阵乘累加
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 
    {%r8, %r9},           // D[2]
    {%r0, %r1, %r2, %r3}, // A[4]
    {%r4, %r5},           // B[2]
    {%r8, %r9};           // C[2] (累加器)
```

### 10.4 Registers → Global Memory (Epilogue)

```cpp
// 线性组合: D = alpha * accum + beta * C
// 加上 ReLU: max(D, 0)

// 读取 C
ld.global.f16 c_val, [gmem_C];

// 计算
d_val = alpha * accum + beta * c_val;
d_val = max(d_val, 0);  // ReLU

// 写回
st.global [gmem_D], d_val;
```

## 11. 总结：所有使用的底层指令

### 数据移动指令
```ptx
# Global → Shared (Sm80)
cp.async.ca.shared.global [%smem], [%gmem], 16;
cp.async.commit_group;
cp.async.wait_group N;

# Shared → Registers (Sm75+)
ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r0,%r1,%r2,%r3}, [%addr];
ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%r0,%r1}, [%addr];

# Registers → Global
st.global.f16 [%gmem], %data;
```

### 计算指令
```ptx
# Tensor Core MMA (Sm80, FP16)
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 
    {%d0, %d1},           # D[2]
    {%a0, %a1, %a2, %a3}, # A[4]
    {%b0, %b1},           # B[2]
    {%c0, %c1};           # C[2]

# 标量运算 (ReLU, scaling)
fma.rn.f16 %d, %a, %b, %c;  # D = A * B + C
max.f16 %d, %a, 0.0;        # D = max(A, 0)
```

### 同步指令
```ptx
__syncthreads();
→ bar.sync 0;
```

## 下一步

基于这个完整的分析，我将生成一个扁平化的 .cu 文件，直接使用这些 PTX 指令实现融合的 B2B GEMM。

