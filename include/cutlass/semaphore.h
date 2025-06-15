/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
 ===============================================================================
                          Split-K GEMM 详细解析（以下内容为markdown格式，建议粘贴到markdown中进行查看）
 ===============================================================================

 问题：为什么split-K中多个threadblock需要处理同一个tile，不应该不同的threadblock处理不同的tile吗？

 ## 1. 常规GEMM vs Split-K GEMM的区别

 ### 常规GEMM（每个threadblock处理不同的输出tile）
 ```
 输出矩阵C (M×N):
 ┌─────┬─────┬─────┬─────┐
 │TB-0 │TB-1 │TB-2 │TB-3 │  ← 每个TB处理不同的输出tile
 ├─────┼─────┼─────┼─────┤
 │TB-4 │TB-5 │TB-6 │TB-7 │
 ├─────┼─────┼─────┼─────┤
 │TB-8 │TB-9 │TB-10│TB-11│
 └─────┴─────┴─────┴─────┘

 每个threadblock: 处理完整的K维度 (A[M×K] * B[K×N])
 ```

 ### Split-K GEMM（多个threadblock处理同一个输出tile）
 ```
 同一个输出tile的K维度分片处理:

 A矩阵的K维度分片:        B矩阵的K维度分片:
 ┌─────┬─────┬─────┐      ┌─────────────┐
 │ K0  │ K1  │ K2  │  ×  │ K0 │ K1 │ K2 │ = 输出tile C[i,j]
 └─────┴─────┴─────┘      │────┼────┼────│
   ↓     ↓     ↓           └─────────────┘
 TB-K0  TB-K1  TB-K2

 结果累加: C[i,j] = C_K0 + C_K1 + C_K2
 ```

 ## 2. 为什么需要Split-K？

 ### 问题场景：K维度过大导致的性能问题

 ```cpp
 // 假设矩阵乘法: A(5120×16384) × B(16384×4096) = C(5120×4096)
 // Threadblock tile: 128×256×64

 // 常规方法的问题：
 K_iterations = 16384 / 64 = 256次迭代  // 每个threadblock需要256次K维度迭代
 ```

 **性能问题**：
 1. **寄存器压力过大**：需要维持256次迭代的累加器状态
 2. **共享内存占用过多**：双缓冲需要存储大量中间数据  
 3. **SM利用率低**：少数长时间运行的threadblock无法充分利用GPU

 ### Split-K的解决方案

 ```cpp
 // Split-K方法：
 split_k_slices = 4  // 将K维度分为4片
 每片K_iterations = 256 / 4 = 64次迭代

 // 现在每个threadblock只需要64次迭代，但需要4个threadblock协作
 ```

 ## 3. Split-K的具体工作流程

 ### 阶段1：并行计算各个K分片
 ```cpp
 // 4个threadblock并行处理同一个输出tile C[i,j]
 TB-K0: 计算 C_partial0 = A[i, 0:4096]     × B[0:4096,     j]  // K=0~4095
 TB-K1: 计算 C_partial1 = A[i, 4096:8192]  × B[4096:8192,  j]  // K=4096~8191  
 TB-K2: 计算 C_partial2 = A[i, 8192:12288] × B[8192:12288, j]  // K=8192~12287
 TB-K3: 计算 C_partial3 = A[i, 12288:16384]× B[12288:16384,j]  // K=12288~16383

 // 这些计算可以完全并行进行！
 ```

 ### 阶段2：串行累加得到最终结果
 ```cpp
 // 使用信号量确保按顺序累加
 C[i,j] = 0
 C[i,j] += C_partial0  // TB-K0完成后写入
 C[i,j] += C_partial1  // TB-K1等待TB-K0完成后累加
 C[i,j] += C_partial2  // TB-K2等待TB-K1完成后累加  
 C[i,j] += C_partial3  // TB-K3等待TB-K2完成后累加
 ```

 ## 4. 信号量的作用：确保正确的累加顺序

 ```cpp
 // 没有信号量的话会出现竞争条件：
 TB-K1可能在TB-K0还没写入C[i,j]时就开始读取和累加
 → 导致最终结果错误

 // 有信号量的正确流程：
 TB-K0: 计算完成 → 写入C[i,j] = C_partial0 → release(semaphore=1)
 TB-K1: wait(semaphore==1) → 读取C[i,j] → 累加 → 写入C[i,j] += C_partial1 → release(semaphore=2)
 TB-K2: wait(semaphore==2) → 读取C[i,j] → 累加 → 写入C[i,j] += C_partial2 → release(semaphore=3)
 TB-K3: wait(semaphore==3) → 读取C[i,j] → 累加 → 写入C[i,j] += C_partial3 → release(semaphore=0)
 ```

 ## 5. Split-K的优势

 ### 🚀 **提高并行度**
 ```
 常规方法: 1个threadblock处理1个输出tile
 Split-K:  4个threadblock同时处理1个输出tile（不同K分片）
 → 4倍的计算并行度！
 ```

 ### 💾 **减少资源占用**  
 - **寄存器使用**：从256次累加减少到64次累加
 - **共享内存**：每个threadblock处理更小的K块
 - **执行时间**：更短的kernel执行时间，更好的调度

 ### 📈 **提高SM利用率**
 ```
 常规方法: 少数长时间运行的threadblock → SM空闲时间多
 Split-K:  更多短时间运行的threadblock → 更好的负载均衡
 ```

 ## 6. 何时使用Split-K？

 Split-K适用于以下场景：
 - **K维度很大**：导致单个threadblock迭代次数过多
 - **M×N较小**：输出tile数量有限，需要更多并行度
 - **内存受限**：需要减少单个threadblock的资源占用

 总结：Split-K是一种**将串行的K维度计算转换为并行+串行累加**的优化技术，
 多个threadblock协作处理同一个输出tile是为了充分利用GPU的并行计算能力。

 ===============================================================================

 ## CUTLASS中Split-K的Threadblock线程数计算

 在CUTLASS中，使用Split-K后，每个threadblock的线程数量计算与常规GEMM相同，
 Split-K主要影响的是threadblock的数量和工作分配，而不是每个threadblock内部的线程组织。

 ### 1. Threadblock线程数计算公式

 在CUTLASS中，每个threadblock的线程数由以下公式确定：

 ```cpp
 // 基础计算公式
 threads_per_threadblock = WarpCount::kM × WarpCount::kN × WarpCount::kK × 32

 // 其中：
 // WarpCount::kM = ceil(ThreadblockShape::kM / WarpShape::kM)
 // WarpCount::kN = ceil(ThreadblockShape::kN / WarpShape::kN) 
 // WarpCount::kK = ceil(ThreadblockShape::kK / WarpShape::kK)
 // 32 = 每个warp的线程数
 ```

 ### 2. 具体计算示例

 以你的配置为例：

 ```cpp
 // 配置参数
 using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>;  // M=128, N=256, K=64
 using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;           // M=64, N=64, K=64

 // 计算各维度的warp数量
 WarpCount::kM = ceil(128 / 64) = 2
 WarpCount::kN = ceil(256 / 64) = 4  
 WarpCount::kK = ceil(64 / 64) = 1

 // 总线程数
 threads_per_threadblock = 2 × 4 × 1 × 32 = 256个线程
 ```

 ### 3. Split-K对线程数的影响

 #### 常规GEMM vs Split-K GEMM的线程组织

 ```cpp
 // 常规GEMM
 每个threadblock: 256个线程
 总threadblock数: (M/128) × (N/256) 

 // Split-K GEMM (假设split_k_slices=4)
 每个threadblock: 256个线程 (相同！)
 总threadblock数: (M/128) × (N/256) × 4
 ```

 **关键点：Split-K不改变单个threadblock的线程数，只增加threadblock的总数量！**

 ### 4. Split-K中的线程工作分配

 #### 线程层次结构保持不变

 ```cpp
 每个Threadblock (256线程):
 ├── Warp 0 (32线程) → 处理输出tile的一部分
 ├── Warp 1 (32线程) → 处理输出tile的一部分
 ├── Warp 2 (32线程) → 处理输出tile的一部分
 ├── Warp 3 (32线程) → 处理输出tile的一部分
 ├── Warp 4 (32线程) → 处理输出tile的一部分
 ├── Warp 5 (32线程) → 处理输出tile的一部分
 ├── Warp 6 (32线程) → 处理输出tile的一部分
 └── Warp 7 (32线程) → 处理输出tile的一部分

 // 在Split-K中，每个threadblock仍然有这8个warp，
 // 但现在处理的是K维度的一个分片，而不是完整的K维度
 ```

 ### 5. 具体的Split-K工作示例

 ```cpp
 // 假设矩阵: A(5120×4096) × B(4096×4096) = C(5120×4096)
 // Threadblock tile: 128×256×64
 // split_k_slices = 4

 // 常规GEMM的threadblock网格
 grid_dim = (ceil(4096/256), ceil(5120/128), 1)
          = (16, 40, 1)
 总threadblock数 = 16 × 40 × 1 = 640个

 // Split-K的threadblock网格  
 grid_dim = (ceil(4096/256), ceil(5120/128), 4)
          = (16, 40, 4) 
 总threadblock数 = 16 × 40 × 4 = 2,560个

 // 每个threadblock的线程数都是256，没有变化！
 ```

 ### 6. Split-K中的线程具体工作

 ```cpp
 // 对于输出位置C[i,j]，现在有4个threadblock协作处理：

 // Threadblock-K0 (256个线程)
 for (int k = 0; k < 1024; ++k) {  // 处理K的第1个分片 (0-1023)
     // 256个线程协作计算 A[i, 0:1023] × B[0:1023, j]
 }

 // Threadblock-K1 (256个线程)  
 for (int k = 1024; k < 2048; ++k) {  // 处理K的第2个分片 (1024-2047)
     // 256个线程协作计算 A[i, 1024:2047] × B[1024:2047, j]
 }

 // Threadblock-K2 (256个线程)
 for (int k = 2048; k < 3072; ++k) {  // 处理K的第3个分片 (2048-3071)
     // 256个线程协作计算 A[i, 2048:3071] × B[2048:3071, j]
 }

 // Threadblock-K3 (256个线程)
 for (int k = 3072; k < 4096; ++k) {  // 处理K的第4个分片 (3072-4095)
     // 256个线程协作计算 A[i, 3072:4095] × B[3072:4095, j]
 }
 ```

 ### 7. 资源使用对比

 ```cpp
 // 常规GEMM
 单个threadblock的K迭代次数: 4096/64 = 64次
 寄存器压力: 需要维持64次迭代的累加器状态
 执行时间: 较长

 // Split-K (split_k_slices=4)
 单个threadblock的K迭代次数: (4096/4)/64 = 16次  
 寄存器压力: 只需维持16次迭代的累加器状态
 执行时间: 较短，但需要4个threadblock协作
 ```

 ### 总结

 **Split-K不改变每个threadblock的线程数量**，它只是：

 1. **增加了threadblock的总数量**：从处理一个完整K维度变为多个threadblock处理K的不同分片
 2. **减少了每个threadblock的工作量**：每个threadblock处理更少的K迭代
 3. **保持了线程组织结构**：warp数量、每个warp的线程数都保持不变

 这样既充分利用了GPU的并行性，又降低了单个threadblock的资源压力。

 ===============================================================================
*/
/*! \file
    \brief Implementation of a CTA-wide semaphore for inter-CTA synchronization.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// CTA-wide semaphore for inter-CTA synchronization.
/// 
/// CUTLASS信号量类：用于多threadblock间的同步协调
/// 
/// 在CUTLASS的split-K优化中，多个threadblock需要协作处理同一个输出tile：
/// - 每个threadblock处理K维度的一个分片
/// - 需要按顺序将结果累加到最终的输出矩阵
/// - 信号量确保后续threadblock能读取到前序threadblock的中间结果
///
/// 工作原理：
/// 1. 每个输出tile位置对应一个全局内存中的信号量
/// 2. threadblock按K分片顺序依次获取信号量
/// 3. 完成计算后释放信号量，唤醒下一个等待的threadblock
/// 4. 最后一个threadblock重置信号量，为下轮计算准备
class Semaphore { 
public:

  /// 指向全局内存中信号量值的指针
  /// 多个threadblock通过这个共享的内存位置进行同步
  int *lock;
  
  /// 标识当前线程是否需要参与等待操作
  /// 通常只有thread_id==0的线程负责信号量操作，其他线程跟随
  /// 这样可以减少不必要的全局内存访问和同步开销
  bool wait_thread;
  
  /// 本地缓存的信号量状态值
  /// 避免每次检查都访问全局内存，提高性能
  /// -1表示未初始化状态
  int state;

public:

  /// Implements a semaphore to wait for a flag to reach a given value
  /// 
  /// 构造函数：初始化信号量对象
  /// 
  /// @param lock_ 指向全局内存中信号量的指针，多个threadblock共享
  /// @param thread_id 当前线程的ID，用于确定是否参与同步操作
  /// 
  /// 设计说明：
  /// - 只有特定线程(thread_id<=0)参与信号量操作，减少内存竞争
  /// - 其他线程通过__syncthreads()与参与线程保持同步
  /// - state初始化为-1，表示尚未从全局内存读取实际值
  CUTLASS_HOST_DEVICE
  Semaphore(int *lock_, int thread_id): 
    lock(lock_),                                    // 保存信号量的全局内存地址
    wait_thread(thread_id < 0 || thread_id == 0),  // 确定是否为等待线程（通常是thread 0）
    state(-1) {                                     // 初始状态设为-1，表示未初始化

  }

  /// Permit fetching the synchronization mechanism early
  /// 
  /// 预取信号量值：从全局内存异步加载当前信号量状态
  /// 
  /// 这个函数实现了"预取"机制，允许在实际需要等待之前就开始加载信号量值。
  /// 这样可以隐藏全局内存访问的延迟，在等待期间可以并行执行其他初始化工作。
  /// 
  /// 内存模型说明：
  /// - acquire语义(Volta+)：确保后续的内存操作不会重排到这次加载之前
  /// - cg(cache global)语义(Pre-Volta)：绕过L1缓存，确保读取到最新值
  CUTLASS_DEVICE
  void fetch() {
    // 只有指定的等待线程执行实际的内存操作
    if (wait_thread) {
      #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      // Volta及更新架构：使用acquire内存排序，确保内存一致性
      // ld.global.acquire.gpu.b32: 从全局内存加载32位值，具有acquire语义
      // %0: 输出操作数，将加载的值存储到state变量
      // %1: 输入操作数，lock指针的地址
      asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));  
      #else
      // Pre-Volta架构：使用缓存全局(cache global)语义
      // ld.global.cg.b32: 绕过L1缓存直接从L2/全局内存加载
      asm volatile ("ld.global.cg.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));  
      #endif
    }
    // 非等待线程不执行内存操作，通过后续的__syncthreads()获得同步状态
  }

  /// Gets the internal state
  /// 
  /// 获取本地缓存的信号量状态值
  /// 
  /// @return 当前缓存的信号量值，-1表示尚未初始化
  /// 
  /// 注意：这个值可能不是最新的全局状态，需要调用fetch()更新
  CUTLASS_DEVICE
  int get_state() const {
    return state;
  }

  /// Waits until the semaphore is equal to the given value
  /// 
  /// 等待信号量达到指定值：split-K同步的核心实现
  /// 
  /// @param status 期望的信号量值，默认为0
  /// 
  /// 工作流程：
  /// 1. 检查当前状态是否等于期望值
  /// 2. 如果不等，所有线程继续等待
  /// 3. 等待线程重新获取信号量值
  /// 4. 重复直到条件满足
  /// 5. 最后进行threadblock同步
  /// 
  /// __syncthreads_and()的妙用：
  /// - 只有等待线程的state会更新，其他线程state保持-1
  /// - __syncthreads_and()确保只有当ALL线程的条件都为真时才退出
  /// - 由于非等待线程的state始终是-1（≠status），循环会持续
  /// - 直到等待线程获取到正确的status值，所有线程才同时退出
  CUTLASS_DEVICE
  void wait(int status = 0) {
    // 使用__syncthreads_and()实现条件等待：
    // - 等待线程：检查state是否等于期望的status
    // - 非等待线程：state保持-1，条件始终为false
    // - 只有当等待线程获取到正确状态时，所有线程才能退出循环
    while( __syncthreads_and(state != status) ) {
      fetch();  // 重新从全局内存获取信号量值
    }

    // 最终同步：确保所有线程都已退出等待状态
    __syncthreads();
  }

  /// Updates the lock with the given result
  /// 
  /// 释放信号量：更新全局信号量值，通知下一个等待者
  /// 
  /// @param status 要设置的新信号量值，默认为0
  /// 
  /// 释放流程：
  /// 1. 先进行threadblock内同步，确保所有计算都已完成
  /// 2. 等待线程将新值写入全局内存
  /// 3. 使用release语义确保之前的所有内存操作对后续读取者可见
  /// 
  /// 内存排序语义：
  /// - release语义确保在写入信号量之前的所有内存操作都已完成
  /// - 这保证了下一个threadblock能看到当前threadblock的计算结果
  CUTLASS_DEVICE
  void release(int status = 0) {
    // 首先进行threadblock内同步，确保：
    // 1. 所有线程的计算都已完成
    // 2. 共享内存的写入都已完成
    // 3. 全局内存的写入都已完成
    __syncthreads();

    // 只有等待线程执行实际的释放操作
    if (wait_thread) {
      #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      // Volta及更新架构：使用release内存排序
      // st.global.release.gpu.b32: 将32位值存储到全局内存，具有release语义
      // 确保之前的所有内存操作在这次写入完成前都已对其他线程可见
      asm volatile ("st.global.release.gpu.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
      #else
      // Pre-Volta架构：使用缓存全局语义
      // st.global.cg.b32: 绕过L1缓存直接写入L2/全局内存
      asm volatile ("st.global.cg.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
      #endif
    }
    // 非等待线程无需执行内存操作，通过之前的__syncthreads()已经保证同步
  }
  
  // ========== Split-K工作流程示例 ==========
  //
  // 假设K维度被分为3个分片，处理同一个输出tile：
  //
  // 时间轴：  T0    T1    T2    T3    T4    T5
  // TB-K0:   [计算] release(1) ------> 完成
  // TB-K1:   wait(0) [计算] release(2) ------> 完成  
  // TB-K2:   ------> wait(1) [计算] release(0) -> 完成
  //
  // 信号量值变化: 0 -> 1 -> 2 -> 0 (重置为下轮使用)
  //
  // 1. TB-K0: 直接开始计算，完成后设置semaphore=1
  // 2. TB-K1: 等待semaphore==1，读取TB-K0结果，累加后设置semaphore=2  
  // 3. TB-K2: 等待semaphore==2，读取TB-K1结果，累加后重置semaphore=0
  //
  // 这样确保了：
  // - 串行的reduce语义：D = D0 + D1 + D2
  // - 内存一致性：每个TB都能读取到前序TB的正确结果
  // - 资源复用：信号量重置后可用于下一组计算
};

// ========== 设计要点总结 ==========
//
// 🔒 **线程协作模式**：
//    - 只有thread-0参与信号量操作，减少内存竞争
//    - 其他线程通过__syncthreads()保持同步
//
// 🚀 **性能优化**：  
//    - fetch()实现预取，隐藏内存访问延迟
//    - 本地缓存state，避免重复的全局内存访问
//
// 🔄 **内存一致性**：
//    - acquire/release语义确保正确的内存排序
//    - 防止编译器和硬件的内存操作重排
//
// ⚡ **Split-K支持**：
//    - 支持任意数量的K分片
//    - 自动处理信号量的循环使用
//    - 保证串行reduce的正确语义

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
