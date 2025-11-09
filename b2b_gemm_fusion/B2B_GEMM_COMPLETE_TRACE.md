# fused_two_gemms_f16_sm80_shmem.cu 完整追踪分析

## 执行摘要

这个文档记录了对 `fused_two_gemms_f16_sm80_shmem.cu` 的**完整**模板追踪，从最高层的 API 调用到最底层的 PTX 指令。

## 1. 模板参数配置

### 入口参数 (Line 167-187)

```cpp
using B2bGemm = cutlass::gemm::device::B2bGemm<
    cutlass::half_t,                          // ElementA
    cutlass::layout::RowMajor,                // LayoutA
    cutlass::half_t,                          // ElementB
    cutlass::layout::ColumnMajor,             // LayoutB
    cutlass::half_t,                          // ElementC
    cutlass::layout::RowMajor,                // LayoutC
    cutlass::half_t,                          // ElementAccumulator
    cutlass::arch::OpClassTensorOp,           // OperatorClass (使用 Tensor Core!)
    cutlass::arch::Sm80,                      // ArchTag (Ampere 架构)
    cutlass::gemm::GemmShape<64, 64, 32>,     // ThreadblockShape0
    cutlass::gemm::GemmShape<64, 256, 32>,    // ThreadblockShape1
    cutlass::gemm::GemmShape<32, 32, 32>,     // WarpShape0
    cutlass::gemm::GemmShape<64, 64, 32>,     // WarpShape1
    cutlass::gemm::GemmShape<16, 8, 16>,      // InstructionShape (m16n8k16)
    EpilogueOutputOp0,                        // LinearCombinationRelu
    EpilogueOutputOp1,                        // LinearCombinationRelu
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3,                                        // Stages (3-stage pipeline)
    true                                      // SmemAccumulator
>;
```

## 2. GEMM 0 配置（第一个矩阵乘法）

### 基本参数
```cpp
ThreadblockShape0 = GemmShape<64, 64, 32>
WarpShape0 = GemmShape<32, 32, 32>
InstructionShape = GemmShape<16, 8, 16>
```

### 计算得出的参数
```cpp
// Warp数量
WarpCount0 = GemmShape<
    64 / 32,   // M: 2 warps
    64 / 32,   // N: 2 warps
    32 / 32    // K: 1
>;
// Total: 2 * 2 * 1 = 4 warps

// 总线程数
kThreads = 4 * 32 = 128 threads

// Warp 内的 MMA 迭代次数
kMmaIterationsM = 32 / 16 = 2
kMmaIterationsN = 32 / 8 = 4
kMmaIterationsK = 32 / 16 = 2
// 每个 warp 执行: 2 * 4 * 2 = 16 个 mma.sync 指令

// Warp GEMM 迭代次数（在 K 维度）
kWarpGemmIterations0 = 32 / 16 = 2
// 所以内层循环会迭代 2 次，每次执行 16 个 mma.sync
```

### Shared Memory 布局
```cpp
// 3 stages for A0
s_A0[3][64 * 32] = 3 * 2048 half = 12 KB

// 3 stages for B0
s_B0[3][32 * 64] = 3 * 2048 half = 12 KB

// Accumulator (GEMM0 的输出，GEMM1 的输入)
s_Accum[64 * 64] = 4096 half = 8 KB

// Total for GEMM0: ~32 KB
```

## 3. GEMM 1 配置（第二个矩阵乘法）

### 基本参数
```cpp
ThreadblockShape1 = GemmShape<64, 256, 32>
WarpShape1 = GemmShape<64, 64, 32>
InstructionShape = GemmShape<16, 8, 16>
```

### 计算得出的参数
```cpp
// Warp数量
WarpCount1 = GemmShape<
    64 / 64,   // M: 1 warp
    256 / 64,  // N: 4 warps
    32 / 32    // K: 1
>;
// Total: 1 * 4 * 1 = 4 warps

// 使用相同的 128 threads

// Warp 内的 MMA 迭代次数
kMmaIterationsM = 64 / 16 = 4
kMmaIterationsN = 64 / 8 = 8
kMmaIterationsK = 32 / 16 = 2
// 每个 warp 执行: 4 * 8 * 2 = 64 个 mma.sync 指令

// Warp GEMM 迭代次数
kWarpGemmIterations1 = 32 / 16 = 2
```

### Shared Memory 布局
```cpp
// 输入 A1 来自 s_Accum (GEMM0 的输出)
// 已经在 shared memory 中，不需要额外分配

// 3 stages for B1
s_B1[3][32 * 256] = 3 * 8192 half = 48 KB

// Total for GEMM1: ~48 KB

// 总的 Shared Memory: ~80 KB
```

## 4. 底层 PTX 指令定义

### 4.1 mma.sync.m16n8k16.f16 (Tensor Core)

**位置**: `include/cutlass/arch/mma_sm80.h` Line 277-327

```cpp
template <>
struct Mma<
    gemm::GemmShape<16, 8, 16>,  // Shape
    32,                          // kThreads (一个 warp)
    half_t,                      // ElementA
    layout::RowMajor,            // LayoutA
    half_t,                      // ElementB
    layout::ColumnMajor,         // LayoutB
    half_t,                      // ElementC
    layout::RowMajor,            // LayoutC
    OpMultiplyAdd> {
    
    using FragmentA = Array<half_t, 4>;  // 4 个 uint32_t (8 个 half)
    using FragmentB = Array<half_t, 2>;  // 2 个 uint32_t (4 个 half)
    using FragmentC = Array<half_t, 4>;  // 4 个 uint32_t (8 个 half)
    
    CUTLASS_HOST_DEVICE
    void operator()(
        FragmentC &d,
        FragmentA const &a,
        FragmentB const &b,
        FragmentC const &c) const {
        
        uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
        uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
        uint32_t const *C = reinterpret_cast<uint32_t const *>(&c);
        uint32_t *D = reinterpret_cast<uint32_t *>(&d);
        
        // 关键 PTX 指令！Line 311
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0,%1}, "              // D[2] - 输出 (2 个 uint32_t = 4 个 half)
            "{%2,%3,%4,%5}, "        // A[4] - A矩阵 (4 个 uint32_t = 8 个 half)
            "{%6,%7}, "              // B[2] - B矩阵 (2 个 uint32_t = 4 个 half)
            "{%8,%9};\n"             // C[2] - 累加器 (2 个 uint32_t = 4 个 half)
            : "=r"(D[0]), "=r"(D[1])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
              "r"(B[0]), "r"(B[1]),
              "r"(C[0]), "r"(C[1])
        );
    }
};
```

**工作原理**:
- 一个 warp (32 threads) 协作执行一个 mma 指令
- 输入: A[16x16] + B[16x8] (FP16)
- 输出: D[16x8] (FP16)
- K dimension: 16 个元素
- 每个 thread 提供部分输入，接收部分输出

### 4.2 ldmatrix (从 Shared Memory 加载)

**位置**: `include/cutlass/arch/memory_sm75.h` Line 122-141

```cpp
// ldmatrix.x4 - 加载 4 个 8x8 矩阵块
template <>
inline __device__ void ldsm<layout::RowMajor, 4>(
    Array<unsigned, 4> & D,
    void const* ptr) {
    
    unsigned addr = cutlass_get_smem_pointer(ptr);
    
    int x, y, z, w;
    
    // 关键 PTX 指令！Line 131
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
        : "r"(addr)
    );
    
    reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
}

// ldmatrix.x2 - 加载 2 个 8x8 矩阵块
template <>
inline __device__ void ldsm<layout::RowMajor, 2>(
    Array<unsigned, 2> & D,
    void const* ptr) {
    
    unsigned addr = cutlass_get_smem_pointer(ptr);
    
    int x, y;
    
    // 关键 PTX 指令！Line 107
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];"
        : "=r"(x), "=r"(y)
        : "r"(addr)
    );
    
    reinterpret_cast<int2 &>(D) = make_int2(x, y);
}
```

**工作原理**:
- 从 shared memory 加载数据到寄存器
- 数据布局专门优化给 Tensor Core 使用
- `.x4` 加载 4 个 8x8 块 (用于 A 矩阵的 16x16)
- `.x2` 加载 2 个 8x8 块 (用于 B 矩阵的 16x8)

### 4.3 cp.async (异步拷贝)

**位置**: `include/cutlass/arch/memory_sm80.h` Line 115-145

```cpp
template <int SizeInBytes, CacheOperation::Kind cache_op>
struct cp_async {
    
    CUTLASS_DEVICE
    static void copy(
        void *smem_ptr,
        void const *global_ptr,
        bool pred_guard) {
        
        unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
        
        // 关键 PTX 指令！Line 131
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %0, 0;\n"
            "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
            "}\n"
            ::"r"((int)pred_guard),
              "r"(smem_int_ptr),
              "l"(global_ptr),
              "n"(SizeInBytes)
        );
    }
};

// 提交 cp.async 组
CUTLASS_DEVICE
void cp_async_fence() {
    // Line 436
    asm volatile("cp.async.commit_group;\n" ::);
}

// 等待 cp.async 完成
template <int N>
CUTLASS_DEVICE void cp_async_wait() {
    // Line 446
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
```

**工作原理**:
- 异步地从 global memory 拷贝到 shared memory
- 不阻塞当前线程，可以与计算重叠
- `cp.async.commit_group` 提交一组拷贝
- `cp.async.wait_group N` 等待除最近 N 组外的所有拷贝完成

## 5. 完整执行流程

### 5.1 Kernel 启动

```cpp
// device/b2b_gemm.h Line 320
cutlass::Kernel<B2bGemmKernel><<<grid, block, smem_size, stream>>>(params_);

// 其中
grid = GemmCoord(
    div_ceil(M, 64),      // GEMM0 的 M
    div_ceil(N1, 256),    // GEMM1 的 N
    1
);
block = dim3(128, 1, 1);  // 128 threads (4 warps)
smem_size = ~80 KB
```

### 5.2 Kernel 内部 (kernel/b2b_gemm.h Line 692-702)

```cpp
// 1. 构造 B2bMma 对象
B2bMma b2bMma(
    shared_storage.main_loop,
    thread_idx,  // 0-127
    warp_idx,    // 0-3
    lane_idx,    // 0-31
    params.problem_size_0.n()  // GEMM0 的 N 维度
);

// 2. 分配累加器
typename B2bMma::FragmentC0 src_accum;       // GEMM0 的输入累加器
typename B2bMma::FragmentC1 accumulators;    // GEMM1 的输出累加器

src_accum.clear();
accumulators.clear();

// 3. 执行融合的两个 GEMM
b2bMma(
    gemm_k_iterations_0,  // K0 / 32
    accumulators,         // 最终输出
    iterator_A0,          // A0 迭代器
    iterator_B0,          // B0 迭代器
    iterator_Scale0,      // Scale 向量
    iterator_Bias0,       // Bias 向量
    iterator_B1,          // B1 迭代器
    src_accum,            // 初始累加器
    output_op_0           // GEMM0 的 epilogue 操作 (ReLU)
);

// 4. 执行 GEMM1 的 Epilogue (写回到 global memory)
epilogue(output_op_1, iterator_D1, accumulators, iterator_C1);
```

### 5.3 B2bMma::operator() 执行流程

**位置**: `threadblock/b2b_mma_multistage_smem_accumulator.h` Line 431-888

#### Phase 1: Prologue (预取前 Stages-1 个 tiles)

```cpp
// Line 456-526
for (int stage = 0; stage < Stages - 1; ++stage) {  // stage = 0, 1
    // 使用 cp.async 加载 A0
    for (int j = 0; j < TBLoadIterationsA0; ++j) {
        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA0>(
            dst_ptr + v, iterator_A0.get(), iterator_A0.valid());
        // PTX: cp.async.ca.shared.global [smem], [gmem], 16;
    }
    
    // 使用 cp.async 加载 B0
    for (int j = 0; j < TBLoadIterationsB0; ++j) {
        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB0>(
            dst_ptr + v, iterator_B0.get(), iterator_B0.valid());
        // PTX: cp.async.ca.shared.global [smem], [gmem], 16;
    }
    
    iterator_A0.add_tile_offset({0, 1});  // 移动到下一个 K tile
    iterator_B0.add_tile_offset({1, 0});
    
    cutlass::arch::cp_async_fence();  // 提交这一组 cp.async
    // PTX: cp.async.commit_group;
}

// 等待前 Stages-2 组完成
cutlass::arch::cp_async_wait<Stages - 2>();  // wait_group(1)
// PTX: cp.async.wait_group 1;

__syncthreads();
```

#### Phase 2: GEMM0 Mainloop

```cpp
// Line 566-655
for (; gemm_k_iterations_0 > (-Stages + 1);) {
    
    // 对于每个 K tile，执行 kWarpGemmIterations0 次 warp-level GEMM
    for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations0; ++warp_mma_k) {
        
        // 从 shared memory 加载到寄存器 (使用 ldmatrix)
        warp_tile_iterator_A0_.load(warp_loaded_frag_A0[(warp_mma_k + 1) % 2]);
        // ↓ 展开为
        // ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r0,%r1,%r2,%r3}, [addr];
        
        warp_tile_iterator_B0_.load(warp_loaded_frag_B0[(warp_mma_k + 1) % 2]);
        // ↓ 展开为
        // ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%r4,%r5}, [addr];
        
        // 转换数据格式（如果需要）
        if (warp_mma_k > 0)
            warp_mma0.transform(warp_transformed_frag_A0[warp_mma_k % 2],
                               warp_transformed_frag_B0[warp_mma_k % 2],
                               warp_loaded_frag_A0[warp_mma_k % 2],
                               warp_loaded_frag_B0[warp_mma_k % 2]);
        
        // 执行 Warp-level MMA (包含多个 mma.sync)
        warp_mma0(
            accum0,  // 累加器
            warp_transformed_frag_A0[warp_mma_k % 2],
            warp_transformed_frag_B0[warp_mma_k % 2],
            accum0
        );
        // ↓ 展开为 (在 warp_mma0 内部，执行 2*4*2=16 次)
        // mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 
        //     {%d0,%d1}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1};
        
        // 异步加载下一个 tile (cp.async)
        if (warp_mma_k == 0) {
            copy_tiles_and_advance_0(iterator_A0, iterator_B0, 0, 0);
            // 内部使用 cp.async
            cutlass::arch::cp_async_fence();
        }
    }
    
    --gemm_k_iterations_0;
    
    // 等待 cp.async 完成
    cutlass::arch::cp_async_wait<Stages - 2>();
    __syncthreads();
    
    // 切换 stage
    smem_write_stage_idx = (smem_write_stage_idx + 1) % Stages;
    smem_read_stage_idx = (smem_read_stage_idx + 1) % Stages;
}
```

#### Phase 3: Epilogue0 (GEMM0 结果写入 Shared Memory)

```cpp
// Line 657-686
// 加载 scale 和 bias
FragmentA1ScaleBias tb_frag_scale;
FragmentA1ScaleBias tb_frag_bias;
iterator_accum0_scale.load(tb_frag_scale);
iterator_accum0_bias.load(tb_frag_bias);

// 应用 scale/bias/ReLU，写入 shared memory
epilogue0(
    output_op_0,              // ReLU + scaling
    tb_frag_scale,
    tb_frag_bias,
    smem_iterator_D0,         // 目标: shared memory
    accum0,                   // 源: 寄存器累加器
    warp_tile_iterator_A0_    // 用于获取位置信息
);
// 结果现在在 s_Accum[] 中

__syncthreads();  // 确保所有线程完成 GEMM0
```

#### Phase 4: GEMM1 Setup

```cpp
// Line 688-703
// 初始化 GEMM1 的 warp iterator
warp_tile_iterator_A1_.set_kgroup_index(0);
warp_tile_iterator_B1_.set_kgroup_index(0);

// 从 shared memory 加载 GEMM0 的结果作为 GEMM1 的 A 矩阵
warp_tile_iterator_A1_.load(warp_loaded_frag_A1[0]);
// ↓ 从 s_Accum 加载，使用 ldmatrix

// 预取 B1 的前 Stages-1 个 tiles
for (int stage = 0; stage < Stages - 1; ++stage) {
    copy_tiles_and_advance_1(iterator_B1, 0);
    // 使用 cp.async 加载 B1
    cutlass::arch::cp_async_fence();
}

cutlass::arch::cp_async_wait<Stages - 2>();
__syncthreads();
```

#### Phase 5: GEMM1 Mainloop

```cpp
// Line 712-788
int gemm_k_iterations_1 = (problem_size_1_n + Shape1::kK - 1) / Shape1::kK;

for (; gemm_k_iterations_1 > (-Stages + 1);) {
    
    for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations1; ++warp_mma_k) {
        
        // 从 shared memory 加载 A1 (来自 GEMM0 的输出)
        warp_tile_iterator_A1_.load(warp_loaded_frag_A1[(warp_mma_k + 1) % 2]);
        // ldmatrix.sync.aligned.x4.m8n8.shared.b16 {...}, [s_Accum];
        
        // 从 shared memory 加载 B1 (从 global memory 预取的)
        warp_tile_iterator_B1_.load(warp_loaded_frag_B1[(warp_mma_k + 1) % 2]);
        // ldmatrix.sync.aligned.x2.m8n8.shared.b16 {...}, [s_B1];
        
        // 转换
        if (warp_mma_k > 0)
            warp_mma1.transform(...);
        
        // 执行 Warp-level MMA
        warp_mma1(
            accum1,  // GEMM1 的累加器
            warp_transformed_frag_A1[warp_mma_k % 2],
            warp_transformed_frag_B1[warp_mma_k % 2],
            accum1
        );
        // ↓ 内部执行 4*8*2=64 个 mma.sync 指令
        
        // 异步加载下一个 B1 tile
        if (warp_mma_k == 0) {
            copy_tiles_and_advance_1(iterator_B1, 0);
            cutlass::arch::cp_async_fence();
        }
    }
    
    --gemm_k_iterations_1;
    
    cutlass::arch::cp_async_wait<Stages - 2>();
    __syncthreads();
}
```

## 6. Warp-Level MMA 实现

**位置**: `include/cutlass/gemm/warp/mma_tensor_op.h`

```cpp
template <typename Shape, typename ElementA, typename LayoutA,
          typename ElementB, typename LayoutB,
          typename ElementC, typename LayoutC,
          typename Policy>
class MmaTensorOp {
    
    static int const kMmaIterationsM = Shape::kM / InstructionShape::kM;
    static int const kMmaIterationsN = Shape::kN / InstructionShape::kN;
    static int const kMmaIterationsK = Shape::kK / InstructionShape::kK;
    
    using MmaOperator = typename Policy::Operator;
    
    CUTLASS_DEVICE
    void operator()(
        FragmentC &D,
        FragmentA const &A,
        FragmentB const &B,
        FragmentC const &C) {
        
        // 三重循环，执行多个 mma.sync
        #pragma unroll
        for (int k = 0; k < kMmaIterationsK; ++k) {
            #pragma unroll
            for (int m = 0; m < kMmaIterationsM; ++m) {
                #pragma unroll
                for (int n = 0; n < kMmaIterationsN; ++n) {
                    
                    // 调用底层 mma.sync 指令
                    MmaOperator mma_op;
                    mma_op(
                        D[m*kMmaIterationsN + n],      // 输出片段
                        A[m*kMmaIterationsK + k],      // A 片段
                        B[k*kMmaIterationsN + n],      // B 片段
                        D[m*kMmaIterationsN + n]       // 累加器
                    );
                    // ↑ 这里会调用 arch::Mma::operator()
                    // 最终展开为 PTX mma.sync 指令
                }
            }
        }
    }
};
```

### GEMM0 的 Warp MMA 迭代

```cpp
WarpShape0 = 32x32x32
InstructionShape = 16x8x16

kMmaIterationsM = 32 / 16 = 2
kMmaIterationsN = 32 / 8 = 4
kMmaIterationsK = 32 / 16 = 2

// 三重循环: 2 * 4 * 2 = 16 个 mma.sync 指令
```

### GEMM1 的 Warp MMA 迭代

```cpp
WarpShape1 = 64x64x32
InstructionShape = 16x8x16

kMmaIterationsM = 64 / 16 = 4
kMmaIterationsN = 64 / 8 = 8
kMmaIterationsK = 32 / 16 = 2

// 三重循环: 4 * 8 * 2 = 64 个 mma.sync 指令
```

## 7. 数据流总结

### 完整的数据流

```
GEMM 0:
┌─────────────────────────────────────────────────────────┐
│ Global Memory                                           │
│   A0[M x K0]  ────cp.async────> s_A0[3][64x32]         │
│   B0[K0 x N0] ────cp.async────> s_B0[3][32x64]         │
└─────────────────────────────────────────────────────────┘
                    │
                    │ ldmatrix.x4 (A0)
                    │ ldmatrix.x2 (B0)
                    ↓
           ┌────────────────┐
           │ Registers      │
           │ FragmentA[4]   │
           │ FragmentB[2]   │
           │ FragmentC[4]   │
           └────────────────┘
                    │
                    │ mma.sync.m16n8k16 (16 次)
                    ↓
           ┌────────────────┐
           │ Accumulator0   │
           │ accum0[寄存器] │
           └────────────────┘
                    │
                    │ Epilogue0: scale + bias + ReLU
                    │ store to shared memory
                    ↓
        ┌──────────────────────┐
        │ s_Accum[64x64] FP16  │  ← 中间结果
        └──────────────────────┘
                    │
                    │ __syncthreads()
                    ↓

GEMM 1:
        ┌──────────────────────┐
        │ s_Accum[64x64]       │ ← A1 输入
        └──────────────────────┘
                    │
                    │ ldmatrix.x4 (A1 from s_Accum)
┌───────────────────┴─────────────────────────────────────┐
│ Global Memory                                           │
│   B1[K1 x N1] ────cp.async────> s_B1[3][32x256]        │
└─────────────────────────────────────────────────────────┘
                    │
                    │ ldmatrix.x2 (B1)
                    ↓
           ┌────────────────┐
           │ Registers      │
           │ FragmentA[4]   │
           │ FragmentB[2]   │
           │ FragmentC[4]   │
           └────────────────┘
                    │
                    │ mma.sync.m16n8k16 (64 次)
                    ↓
           ┌────────────────┐
           │ Accumulator1   │
           │ accum1[寄存器] │
           └────────────────┘
                    │
                    │ Epilogue1: ReLU
                    │ store to global memory
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Global Memory                                           │
│   D1[M x N1]  ◄─── st.global ─── accum1                │
└─────────────────────────────────────────────────────────┘
```

## 8. 关键指令使用统计

### 每个 Threadblock 执行的指令数量

#### GEMM 0
```cpp
K0_tiles = K0 / 32
warp_gemm_iterations = 2

Per K tile:
  - cp.async: ~数十次 (加载 A0, B0)
  - ldmatrix: 2 * (4 + 2) = 12 per warp per warp_iteration
  - mma.sync: 16 per warp per warp_iteration
  
Total for GEMM0:
  - cp.async: ~数十次 * K0_tiles
  - ldmatrix: 12 * 4 warps * 2 iterations * K0_tiles
  - mma.sync: 16 * 4 warps * 2 iterations * K0_tiles
```

#### GEMM 1
```cpp
K1_tiles = K1 / 32 = N0 / 32 = 64 / 32 = 2

Per K tile:
  - cp.async: ~数十次 (只加载 B1，A1 已在 shared memory)
  - ldmatrix: 2 * (4 + 2) = 12 per warp per warp_iteration
  - mma.sync: 64 per warp per warp_iteration
  
Total for GEMM1:
  - ldmatrix: 12 * 4 warps * 2 iterations * 2
  - mma.sync: 64 * 4 warps * 2 iterations * 2
```

## 9. 总结

这个 B2B GEMM kernel 的完整技术栈：

### 高层特性
- ✅ 融合的两个 GEMM (避免中间结果写回 DRAM)
- ✅ Shared memory accumulator (GEMM0 → GEMM1)
- ✅ FP16 Tensor Core 加速
- ✅ 3-stage pipeline (隐藏内存延迟)
- ✅ Asynchronous copy (cp.async)
- ✅ ReLU activation

### 底层指令
- ✅ `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` - Tensor Core 计算
- ✅ `ldmatrix.sync.aligned.x4.m8n8.shared.b16` - 优化的 shared memory 加载
- ✅ `ldmatrix.sync.aligned.x2.m8n8.shared.b16` - 优化的 shared memory 加载
- ✅ `cp.async.ca.shared.global` - 异步内存拷贝
- ✅ `cp.async.commit_group` - 提交异步拷贝组
- ✅ `cp.async.wait_group N` - 等待异步拷贝完成
- ✅ `bar.sync 0` (__syncthreads) - 线程块同步

### 性能优化技术
1. **Tensor Core**: 使用 mma.sync，相比 SIMT 快 8-16x
2. **Async Copy**: cp.async 与计算重叠，隐藏内存延迟
3. **ldmatrix**: 专门优化的加载指令
4. **3-stage Pipeline**: 保持计算单元持续工作
5. **Shared Memory Staging**: 避免中间结果写回 DRAM，节省 ~10x 带宽
6. **Fusion**: 两个 kernel 合并为一个，减少启动开销
7. **Register Blocking**: 每个线程维护多个累加器

这是一个**生产级别**的高性能 GEMM kernel！

