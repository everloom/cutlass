# B2B GEMM Fusion 的尺寸限制

## 你的问题

> "如果 gemm1 需要算的是 81920×1024 和 1024×256，那么 s_Accum 只存储 64×64 的 threadblock tile 的结果是不是就不适用了？"

**✅ 完全正确！你发现了 B2B GEMM fusion 的核心限制。**

---

## 当前配置（可以 fusion）

### 矩阵尺寸

```cpp
// Line 49-50
GEMM0: A0[81920 × 576] @ B0[576 × 64]   = Temp[81920 × 64]
GEMM1: Temp[81920 × 64] @ B1[64 × 256] = D1[81920 × 256]

关键: N0 = 64, K1 = 64 (N0 = K1 ✓)
```

### Threadblock 划分

```cpp
GEMM0:
  ThreadblockShape0 = 64 × 64 × 32
  Grid: (81920/64, 64/64, 1) = (1280, 1, 1)
  
  每个 block 计算 Temp 的一个 64×64 区域
  
Block(0,0):   Temp[0:64, 0:64]       → s_Accum[64×64]
Block(1,0):   Temp[64:128, 0:64]     → s_Accum[64×64]
...
Block(1279,0): Temp[81856:81920, 0:64] → s_Accum[64×64]

关键: gridDim.y = 1 (N 方向只有 1 个 block!)
```

### 为什么可以 fusion？

```
每个 threadblock:
1. 完整计算 Temp 的 [某64行, 全部64列]
2. 将完整的 64 列存入 s_Accum[64×64]
3. GEMM1 可以直接使用这 64 列作为完整的 A1 矩阵
4. 计算 GEMM1: Temp[64×64] @ B1[64×256] = D1[64×256]

✓ Fusion 成功！
```

---

## 你的场景（无法 fusion）

### 矩阵尺寸

```cpp
GEMM0: A0[81920 × ?] @ B0[? × 1024] = Temp[81920 × 1024]
GEMM1: Temp[81920 × 1024] @ B1[1024 × 256] = D1[81920 × 256]

关键: N0 = 1024, K1 = 1024
```

### Threadblock 划分会怎样？

```cpp
ThreadblockShape0 = 64 × 64 × 32  (假设不变)

GEMM0 Grid:
  gridDim.x = 81920 / 64 = 1280
  gridDim.y = 1024 / 64 = 16  ← 关键！N 方向需要 16 个 blocks!
  
Block(0, 0):    Temp[0:64, 0:64]
Block(0, 1):    Temp[0:64, 64:128]
Block(0, 2):    Temp[0:64, 128:192]
...
Block(0, 15):   Temp[0:64, 960:1024]
Block(1, 0):    Temp[64:128, 0:64]
...
```

### 问题出现了！

```
Block(0, 0) 的情况:
  - GEMM0: 计算 Temp[0:64, 0:64]
  - s_Accum[64×64] 存储这 64 列
  
  - GEMM1 需要: Temp[0:64, 0:1024] @ B1[1024×256]
                完整的 1024 列！
  
  但是:
  - s_Accum 只有 64 列 (0:64)
  - 其他 960 列在 Block(0,1) 到 Block(0,15) 中
  - 这些 blocks 的 s_Accum 无法访问！
  
  ❌ Fusion 失败！
```

### 为什么无法 fusion？

**根本原因**: **Shared Memory 无法跨 Threadblock 访问**

```
Block(0, 0) 的 s_Accum[64×64]   ← 只能被 Block(0,0) 访问
Block(0, 1) 的 s_Accum[64×64]   ← 只能被 Block(0,1) 访问
...
Block(0, 15) 的 s_Accum[64×64]  ← 只能被 Block(0,15) 访问

如果要在 Block(0, 0) 中执行 GEMM1:
  需要访问 Block(0, 0-15) 的所有 s_Accum
  → 这是不可能的！(CUDA 限制)
```

---

## CUTLASS 的解决方案

### 方案1: 限制 N0 ≤ ThreadblockShape0::kN

**当前做法**:
```cpp
// 代码中的检查 (kernel/b2b_gemm.h:539-540)
if(problem_size_0.n() > B2bMma::Shape0::kN)
    return Status::kErrorInvalidProblem;

// Shape0::kN = ThreadblockShape0::kN = 64

// 所以 N0 必须 ≤ 64
```

**原理**: 
- 如果 N0 ≤ 64，只需要 1 个 block 在 N 方向
- 这个 block 的 s_Accum 可以存储完整的 N0 列
- GEMM1 可以使用完整的数据

### 方案2: 使用更大的 ThreadblockShape

**如果需要 N0 = 128**:
```cpp
ThreadblockShape0 = GemmShape<64, 128, 32>  // N 改为 128
WarpShape0 = GemmShape<32, 64, 32>          // 调整
s_Accum[64 × 128]                           // 增大

Grid: (1280, 1024/128, 1) = (1280, 8, 1)

每个 block 存储 64×128
仍然无法 fusion! (需要 8 个 blocks 在 N 方向)
```

**限制**: ThreadblockShape 也不能无限大
- Shared Memory 限制 (~100 KB)
- 寄存器限制
- 实际上 ThreadblockShape::kN 很少超过 256

### 方案3: 如果 N0 > 256，无法用这种 fusion

**必须回退到非 fusion 版本**:
```cpp
// 非 fusion
GEMM0: A0 @ B0 = Temp → 写回 Global Memory
GEMM1: Temp @ B1 = D1 (从 Global Memory 读取 Temp)

// 损失:
- 额外的 Global Memory 访问
- 带宽浪费
- 性能降低
```

---

## 你的场景分析

### 场景: N0 = 1024

```cpp
GEMM0: [81920 × ?] @ [? × 1024] = Temp[81920 × 1024]
GEMM1: Temp[81920 × 1024] @ B1[1024 × 256] = D1[81920 × 256]
```

### 如果强行用当前的 fusion 代码？

```cpp
ThreadblockShape0 = 64 × 64 × 32

GEMM0 Grid: (1280, 16, 1)  // 16 blocks 在 N 方向

Block(0, 0):
  - 计算 Temp[0:64, 0:64]
  - s_Accum[64×64] 存储
  
  GEMM1 需要计算:
    Temp[0:64, 0:1024] @ B1[1024×256]
    
  但只有 Temp[0:64, 0:64] 在 s_Accum!
  其他 15 个 64-列段在其他 blocks 的 s_Accum 中，无法访问!
  
  ❌ 无法完成 GEMM1 的计算
```

### 正确的处理方式

**方案A: 非 fusion（标准做法）**
```cpp
// Kernel 1: GEMM0
GEMM0<<<(1280, 16), 128>>>(A0, B0, Temp_global);
// Temp_global[81920 × 1024] 在 Global Memory

// Kernel 2: GEMM1  
GEMM1<<<(1280, 4), 128>>>(Temp_global, B1, D1);
```

**方案B: 修改问题规模**
```cpp
// 将 1024 分解为多个小问题
for (int chunk = 0; chunk < 16; ++chunk) {
    // GEMM0: A0 @ B0[:, chunk*64:(chunk+1)*64] = Temp_chunk[81920×64]
    // GEMM1: Temp_chunk @ B1[chunk*64:(chunk+1)*64, :] = D1_partial
    // 累加 D1_partial
}
```

但这已经不是真正的 fusion 了。

**方案C: 使用超大的 ThreadblockShape（不现实）**
```cpp
ThreadblockShape0 = GemmShape<64, 1024, 32>
s_Accum[64 × 1024] = 64 × 1024 half = 128 KB

问题: 超过 Shared Memory 限制 (~100 KB)
```

---

## CUTLASS 中的实际限制

### 代码中的检查

**文件**: `examples/13_two_tensor_op_fusion/kernel/b2b_gemm.h`  
**行号**: Line 539-540

```cpp
// 检查 fusion 的有效性
if(problem_size_0.n() > B2bMma::Shape0::kN)
    return Status::kErrorInvalidProblem;

// Shape0::kN = ThreadblockShape0::kN
```

**对于你的配置**:
```cpp
if (1024 > 64)  // True!
    return Status::kErrorInvalidProblem;

// CUTLASS 会拒绝这个配置！
```

### 为什么有这个限制？

**Line 532-537 的另一个检查**:
```cpp
// Determine if fusion sizes are valid
if(problem_size_0.m() != problem_size_1.m())
    return Status::kErrorInvalidProblem;

if(problem_size_0.n() != problem_size_1.k())  // ← 关键!
    return Status::kErrorInvalidProblem;

if(problem_size_0.n() > B2bMma::Shape0::kN)   // ← 你问的问题
    return Status::kErrorInvalidProblem;
```

**第三个检查的含义**: 
- `problem_size_0.n()` = N0 (GEMM0 的输出列数)
- `B2bMma::Shape0::kN` = ThreadblockShape0::kN = 64
- **N0 必须 ≤ 64**

---

## 实际的尺寸限制

### SHMEM 版本

```cpp
限制: N0 ≤ ThreadblockShape0::kN

常见配置:
  ThreadblockShape0::kN = 64, 128, 或 256
  
最大 N0:
  如果 ThreadblockShape0::kN = 256:
    s_Accum[64 × 256] = 32 KB
    其他 buffers ~50 KB
    Total ~82 KB ✓ 可行
    
  如果 N0 = 512:
    s_Accum[64 × 512] = 64 KB
    Total > 100 KB ✗ 超限
```

**实际限制**: N0 ≤ 256 (对于 SHMEM 版本)

### RF 版本

```cpp
限制: N0 ≤ WarpShape0::kN (寄存器容量)

当前配置:
  WarpShape0::kN = 64
  每个 warp 的累加器: 16 × 64 = 1024 half
  每个 thread: 1024 / 32 = 32 half = 16 uint32_t
  
如果 N0 = 128:
  每个 warp: 16 × 128 = 2048 half
  每个 thread: 64 half = 32 uint32_t
  ⚠️ 寄存器压力很大！
  
如果 N0 = 1024:
  每个 thread: 512 half = 256 uint32_t
  ✗ 超过寄存器限制 (~255)
```

**实际限制**: N0 ≤ 64-128 (对于 RF 版本)

---

## 你的场景: N0 = 1024

### 问题分析

```cpp
GEMM0: [81920 × K0] @ [K0 × 1024] = Temp[81920 × 1024]
GEMM1: Temp[81920 × 1024] @ B1[1024 × 256] = D1[81920 × 256]

N0 = 1024, K1 = 1024
```

### 如果用 ThreadblockShape0::kN = 64

```cpp
GEMM0 Grid:
  gridDim.x = 81920 / 64 = 1280 (M 方向)
  gridDim.y = 1024 / 64 = 16  (N 方向) ← 关键！
  
16 个 blocks 在 N 方向:

Block(0, 0):  Temp[0:64, 0:64]       → s_Accum_00[64×64]
Block(0, 1):  Temp[0:64, 64:128]     → s_Accum_01[64×64]
Block(0, 2):  Temp[0:64, 128:192]    → s_Accum_02[64×64]
...
Block(0, 15): Temp[0:64, 960:1024]   → s_Accum_015[64×64]
```

### 问题

```
Block(0, 0) 想执行 GEMM1:
  需要: Temp[0:64, 0:1024] (完整的 1024 列)
  
  有的: s_Accum_00[64×64] (只有 0:64 列)
  
  缺失: 列 64:1024 在 Block(0,1) 到 Block(0,15) 的 s_Accum 中
  
  能访问吗? ❌ 不能！
  
  原因: Shared Memory 是 per-threadblock 的
        Block(0,0) 无法访问 Block(0,1) 的 shared memory
```

**结论**: **无法 fusion！**

---

## 解决方案

### 方案1: 回退到非 fusion 版本

```cpp
// Kernel 1: GEMM0
gemm0<<<grid0, block>>>(A0, B0, Temp_global);
// Temp_global[81920 × 1024] 在 Global Memory

cudaDeviceSynchronize();  // 确保 GEMM0 完成

// Kernel 2: GEMM1
gemm1<<<grid1, block>>>(Temp_global, B1, D1);
```

**优点**: 
- ✅ 支持任意 N0
- ✅ 简单直接

**缺点**: 
- ❌ Temp 需要写回和读取 Global Memory
- ❌ 浪费带宽: 81920×1024×2 bytes = ~160 MB
- ❌ 两次 kernel 启动开销
- ❌ 性能损失 ~30-50%

### 方案2: 修改 Threadblock 和 Grid 策略

**不切分 N 维度**:
```cpp
// 让一个 threadblock 在 N 方向覆盖所有 1024 列
ThreadblockShape0 = GemmShape<64, 1024, 32>  // ✗ 不可行！

s_Accum[64 × 1024] = 128 KB  // ✗ 超过 SMEM 限制
```

**不现实**: Shared Memory 只有 ~100 KB。

### 方案3: 分块 fusion

```cpp
// 将 1024 列分成 16 个 64-列的块
// 每个块独立 fusion

for (int chunk = 0; chunk < 16; ++chunk) {
    int n_offset = chunk * 64;
    
    // Fused B2B for this chunk
    b2b_gemm_fused<<<grid, block>>>(
        A0,
        B0 + n_offset * ldb0,           // B0[:, n_offset:n_offset+64]
        B1 + n_offset,                   // B1[n_offset:n_offset+64, :]
        D1_partial + n_offset * ldd1,
        ...
    );
}

// 累加所有 D1_partial
```

**优点**:
- ✅ 可以利用部分 fusion 优势
- ✅ 每个 chunk 可以 fusion

**缺点**:
- ⚠️ 需要多次 kernel 启动
- ⚠️ 累加开销
- ⚠️ 不是真正的单次 fusion

---

## 对比总结

### 可以 Fusion 的配置

| N0 | ThreadblockShape::kN | s_Accum | Fusion? |
|----|---------------------|---------|---------|
| 32 | 64 | 64×32 = 4KB | ✅ 可以 |
| 64 | 64 | 64×64 = 8KB | ✅ 可以 |
| 128 | 128 | 64×128 = 16KB | ✅ 可以 |
| 256 | 256 | 64×256 = 32KB | ✅ 可以 (需大SMEM) |
| 512 | ? | 64×512 = 64KB | ⚠️ 勉强 (SMEM紧张) |
| **1024** | ? | 64×1024 = **128KB** | **❌ 不行** |

### 根本限制

```
B2B GEMM Fusion 的必要条件:

1. N0 ≤ ThreadblockShape0::kN
   ├─> 保证 gridDim.y = 1 (N 方向只有 1 个 block)
   └─> 保证一个 block 计算完整的 N0 列

2. ThreadblockShape0::kN × BM0 × sizeof(half) ≤ SMEM_limit
   └─> 保证 s_Accum 能容纳

3. N0 = K1
   └─> 保证矩阵维度匹配
```

对于你的场景 (N0 = 1024):
- ❌ 条件1 不满足 (1024 > 64)
- ❌ 条件2 不满足 (128 KB > 100 KB)
- ✅ 条件3 满足 (如果 K1 = 1024)

**结论: 无法使用这种 fusion 策略！**

---

## CUTLASS 的设计哲学

### B2B Fusion 的目标场景

**设计用于**:
- 小到中等的 N0 (≤ 256)
- 深度学习中的特定模式:
  - MLP 层: [batch, 512] → [512, 128] → [128, 256]
  - Transformer FFN: [batch, 768] → [768, 3072] → [3072, 768]
    - 需要拆分或使用不同策略

**不设计用于**:
- 非常大的 N0 (> 256)
- 这种情况 fusion 的收益有限

### 为什么不支持大 N0？

1. **Shared Memory 是瓶颈**
   - 每个 SM 只有 ~100 KB
   - s_Accum 太大会降低 occupancy

2. **收益递减**
   - N0 越大，中间结果的内存访问占总体比例越小
   - Fusion 的加速效果降低

3. **寄存器压力**
   - 大 N0 需要更多寄存器存储累加器
   - 降低 occupancy

---

## 总结

### 你的理解 ✅

**完全正确！**

如果 N0 = 1024:
- s_Accum[64×64] **确实不够用**
- 一个 threadblock 只能存储 1024 列中的 64 列
- 其他列在其他 blocks 的 s_Accum 中
- 无法跨 block 访问 shared memory
- **这种 fusion 策略不适用**

### B2B GEMM Fusion 的核心约束

**N0 必须 ≤ ThreadblockShape0::kN**

原因:
1. 保证 gridDim.y = 1 (N 方向单个 block)
2. s_Accum 能存储完整的 N0 列
3. GEMM1 能访问完整的 A1 矩阵

### 实际限制

```
SHMEM 版本: N0 ≤ 64-256 (取决于 ThreadblockShape 和 SMEM 容量)
RF 版本: N0 ≤ 64-128 (取决于寄存器容量)

超出这个范围: 必须使用非 fusion 版本
```

**你完全抓住了 B2B GEMM fusion 的本质限制！** 🎯
