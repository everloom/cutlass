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

/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"
#include "cutlass/arch/arch.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial               ///! If true, code supporting split-K via serial reduction is enabled.
>
struct Gemm {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;
    int *semaphore;
    int gemm_k_size;
    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), semaphore(0), gemm_k_size(0) { }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D,
      typename OutputOp::Params output_op = typename OutputOp::Params(),
      int *workspace = nullptr,
      int const *gather_A_indices = nullptr,
      int const *gather_B_indices = nullptr,
      int const *scatter_D_indices = nullptr
    ):
      problem_size(problem_size),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(ref_A.layout()),
      ref_A(ref_A),
      params_B(ref_B.layout()),
      ref_B(ref_B),
      params_C(ref_C.layout()),
      ref_C(ref_C),
      params_D(ref_D.layout()),
      ref_D(ref_D),
      output_op(output_op),
      gather_A_indices(gather_A_indices),
      gather_B_indices(gather_B_indices),
      scatter_D_indices(scatter_D_indices) {

      int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      
      gemm_k_size = gemm_k_iterations * Mma::Shape::kK;

    semaphore = workspace;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Gemm() { } 

  /// Determines whether kernel satisfies alignment
  CUTLASS_HOST_DEVICE
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size,
    typename Mma::IteratorA::TensorRef ref_A,
    typename Mma::IteratorB::TensorRef ref_B,
    typename Epilogue::OutputTileIterator::TensorRef ref_C,
    typename Epilogue::OutputTileIterator::TensorRef ref_D) {

    static int const kAlignmentA = (platform::is_same<typename Mma::IteratorA::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorA::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB =  (platform::is_same<typename Mma::IteratorB::Layout,
                                                       layout::RowMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorB::Layout,
                                                        layout::RowMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_A, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  /// Executes one GEMM
  /// CUTLASS GEMM Kernel的主入口函数
  /// 
  /// 这是CUTLASS GEMM kernel的核心执行函数，负责整个矩阵乘法的完整流程：
  /// 1. 计算threadblock在全局矩阵中的位置和偏移
  /// 2. 设置迭代器访问全局内存中的A、B矩阵
  /// 3. 执行多层次流水线的矩阵乘累加（MMA）
  /// 4. 执行epilogue后处理（如alpha*A*B + beta*C）
  /// 5. 将结果写回全局内存
  ///
  /// @param params 包含所有kernel参数的结构体
  /// @param shared_storage 共享内存存储空间
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // ========== 第一阶段：计算Threadblock的全局位置 ==========
    
    // Compute threadblock location
    // 创建threadblock交换策略对象，用于计算当前threadblock在全局矩阵中的位置
    // Swizzle策略可以改善内存访问模式，提高缓存效率
    ThreadblockSwizzle threadblock_swizzle;

    // 获取当前threadblock在3D tile坐标系中的位置 (M, N, K)
    // threadblock_tile_offset.m(): 在M维度上的tile索引
    // threadblock_tile_offset.n(): 在N维度上的tile索引  
    // threadblock_tile_offset.k(): 在K维度上的tile索引（用于split-K优化）
    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    // 边界检查：如果当前threadblock超出了问题规模的范围，直接退出
    // 这种情况在最后几个threadblock中可能发生，当矩阵大小不能被tile大小整除时
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    // ========== 第二阶段：计算矩阵A和B的访问偏移 ==========
    
    // Compute initial location in logical coordinates
    // 计算矩阵A在全局内存中的起始偏移坐标
    // tb_offset_A.row: 当前threadblock负责处理的A矩阵的起始行 = tile_idx_m * tile_size_m
    // tb_offset_A.column: 当前threadblock负责处理的A矩阵的起始列 = tile_idx_k * gemm_k_size
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,    // M维度偏移：tile索引 × tile大小
      threadblock_tile_offset.k() * params.gemm_k_size, // K维度偏移：用于split-K
    };

    // 计算矩阵B在全局内存中的起始偏移坐标
    // tb_offset_B.row: 当前threadblock负责处理的B矩阵的起始行 = tile_idx_k * gemm_k_size  
    // tb_offset_B.column: 当前threadblock负责处理的B矩阵的起始列 = tile_idx_n * tile_size_n
    cutlass::MatrixCoord tb_offset_B{
      threadblock_tile_offset.k() * params.gemm_k_size, // K维度偏移：用于split-K
      threadblock_tile_offset.n() * Mma::Shape::kN      // N维度偏移：tile索引 × tile大小
    };

    // ========== 第三阶段：计算K维度的处理范围和迭代次数 ==========
    
    // Problem size is a function of threadblock index in the K dimension
    // 计算当前threadblock需要处理的K维度范围的实际大小
    // 这在split-K优化中特别重要，每个threadblock只处理K维度的一部分
    // min确保不会超出原始矩阵的K维度边界
    int problem_size_k = min(
      params.problem_size.k(),  // 原始矩阵的K维度大小
      (threadblock_tile_offset.k() + 1) * params.gemm_k_size); // 当前threadblock的K维度上界

    // Compute threadblock-scoped matrix multiply-add
    // 计算当前threadblock需要执行多少次K维度的迭代
    // 这个值决定了主循环(gemm_iters)中外层for循环的迭代次数
    // 公式：(实际K范围 - 起始K偏移 + tile_k_size - 1) / tile_k_size （向上取整除法）
    int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // ========== 第四阶段：设置线程索引和迭代器 ==========
    
    // Compute position within threadblock  
    // 获取当前线程在threadblock中的索引（0-255对于256线程的block）
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    // 构造访问矩阵A的全局内存迭代器
    // 迭代器负责从全局内存加载数据，支持各种内存布局和优化访问模式
    typename Mma::IteratorA iterator_A(
      params.params_A,        // A矩阵的布局参数（stride等）
      params.ref_A.data(),    // A矩阵在全局内存中的起始地址
      {params.problem_size.m(), problem_size_k}, // A矩阵的逻辑大小 (M×K)
      thread_idx,             // 当前线程索引
      tb_offset_A,            // 当前threadblock在A矩阵中的偏移
      params.gather_A_indices); // 支持间接索引的gather操作（可选）

    // 构造访问矩阵B的全局内存迭代器
    // 与iterator_A类似，但处理B矩阵 (K×N)
    typename Mma::IteratorB iterator_B(
      params.params_B,        // B矩阵的布局参数
      params.ref_B.data(),    // B矩阵在全局内存中的起始地址  
      {problem_size_k, params.problem_size.n()}, // B矩阵的逻辑大小 (K×N)
      thread_idx,             // 当前线程索引
      tb_offset_B,            // 当前threadblock在B矩阵中的偏移
      params.gather_B_indices); // gather操作支持

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    // 计算当前线程所属的warp索引（0-7对于8个warp的threadblock）
    // 使用同步函数确保所有线程获得一致的warp_idx，保证编译器优化的正确性
    int warp_idx = canonical_warp_idx_sync();
    // 计算当前线程在warp内的lane索引（0-31）
    int lane_idx = threadIdx.x % 32;

    // ========== 第五阶段：主要计算阶段 - 矩阵乘累加 ==========
    
    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    // 构造threadblock级别的矩阵乘法对象
    // 这个对象封装了所有的双缓冲流水线逻辑、共享内存管理、warp级MMA操作
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    // 创建累加器fragment，存储在寄存器中
    // 这是整个计算过程中的最终输出累积器
    typename Mma::FragmentC accumulators; // 看到fragment就要想到这个是寄存器

    // 初始化累加器为0，为后续的累加操作准备
    accumulators.clear();

    // 执行矩阵乘累加计算的条件检查
    // !kSplitKSerial: 如果不是串行split-K模式，直接执行
    // gemm_k_iterations > 0: 如果有K维度迭代需要处理，则执行
    if (!kSplitKSerial || gemm_k_iterations > 0) {
      // Compute threadblock-scoped matrix multiply-add
      // 🔥 核心计算：执行多层次流水线的矩阵乘累加
      // 这里会调用之前分析的gemm_iters函数，包含：
      // - 外层循环：遍历K维度的threadblock tiles (gemm_k_iterations次)
      // - 内层循环：遍历每个tile内的warp级MMA操作 (kWarpGemmIterations次)
      // - 三层双缓冲：全局内存↔共享内存↔寄存器，计算与数据传输重叠
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
    }

    // ========== 第六阶段：Epilogue后处理阶段 ==========
    
    //
    // Epilogue
    //
    // 到了这里，mma的结果就已经计算完了，并且结果存在了FragmentC中，这里是对FragmentC的结果进行一些后处理操作，例如计算D=alpha*A*B + beta*C，前面的mma就是A*B，这里epilogue就是算alpha*A*B + beta*C
    
    // 创建输出操作对象，负责执行 D = alpha * (A * B) + beta * C 的线性组合
    // output_op封装了alpha、beta参数以及各种激活函数（如ReLU）
    OutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    // 重新计算threadblock偏移（为epilogue阶段准备）
    // 确保与主计算阶段使用相同的tile位置
    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    // 计算当前threadblock在输出矩阵C/D中的实际内存偏移
    // 将tile坐标转换为实际的矩阵行列坐标
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,  // M维度的实际行偏移
      threadblock_tile_offset.n() * Mma::Shape::kN   // N维度的实际列偏移
    );

    // 计算当前threadblock的全局块索引，用于同步机制
    // 采用行主序的索引计算方式
    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    // ========== 第七阶段：Split-K同步机制设置 ==========
    
    // Construct the semaphore.
    // 为split-K优化创建信号量对象，用于多个threadblock之间的同步
    // 每个输出tile位置有一个对应的信号量
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    // 如果启用了split-K优化并且K维度被分割成多个部分
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      // Fetch the synchronization lock initially but do not block.
      // 获取同步锁，但不阻塞当前线程
      // 这允许在等待同步的同时继续执行其他初始化工作
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      // 告诉output_op当前处理的是第几个K分区，以及总共有多少个K分区
      // 这影响最终结果的缩放和累加方式
      output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }

    // ========== 第八阶段：设置输出矩阵的迭代器 ==========
    
    // Tile iterator loading from source tensor.
    // 这里的iterator_C就是上提到的D=alpha*A*B + beta*C中的C的迭代器，此时C还在gmem中
    // 构造访问输入矩阵C的迭代器，C是线性组合中的加法项
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,        // C矩阵的布局参数
      params.ref_C.data(),    // C矩阵在全局内存中的地址
      params.problem_size.mn(), // C矩阵的大小 (M×N)
      thread_idx,             // 线程索引
      threadblock_offset,     // 当前threadblock在C矩阵中的偏移
      params.scatter_D_indices // scatter操作支持（可选）
    );

    // Tile iterator writing to destination tensor.
    // 这里D就是D=alpha*A*B + beta*C中的结果D，D需要写到gmem中，D的迭代器就是iterator_D
    // 构造写入输出矩阵D的迭代器，D是最终的计算结果
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,        // D矩阵的布局参数
      params.ref_D.data(),    // D矩阵在全局内存中的地址
      params.problem_size.mn(), // D矩阵的大小 (M×N)
      thread_idx,             // 线程索引  
      threadblock_offset,     // 当前threadblock在D矩阵中的偏移
      params.scatter_D_indices // scatter操作支持
    );

    // 构造epilogue处理器，负责执行线性组合和结果写回
    // epilogue也有自己的共享内存需求和线程协作模式
    Epilogue epilogue(
      shared_storage.epilogue,  // epilogue专用的共享内存空间
      thread_idx,              // 线程索引
      warp_idx,                // warp索引
      lane_idx);               // lane索引

    // ========== 第九阶段：Split-K同步等待 ==========
    
    // Wait on the semaphore - this latency may have been covered by iterator construction
    // 等待同步信号量，确保前序的K分区已经完成计算
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      // 对于非第一个K分区的threadblock，需要从D矩阵读取之前的中间结果
      // 这实现了串行reduce的累加效果：D0 + D1 + D2 + ... + Dk
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;  // 将C迭代器重定向到D，读取之前的累加结果
      }

      // 等待轮到当前K分区执行，确保split-K的串行语义
      semaphore.wait(threadblock_tile_offset.k());

    }

    // ========== 第十阶段：执行Epilogue计算 ==========
    
    // Execute the epilogue operator to update the destination tensor.
    // 🔥 执行epilogue操作：
    // 1. 从寄存器fragment读取MMA的计算结果 (accumulators)
    // 2. 从全局内存读取矩阵C的数据 (iterator_C) 
    // 3. 执行线性组合：D = alpha * accumulators + beta * C
    // 4. 应用激活函数（如果有）
    // 5. 将最终结果写入全局内存D (iterator_D)
    epilogue(output_op, iterator_D, accumulators, iterator_C); 
    
    // ========== 第十一阶段：释放Split-K同步资源 ==========
    
    //
    // Release the semaphore
    //

    // 释放信号量，允许下一个K分区的threadblock开始执行
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      int lock = 0;
      // 如果这是最后一个K分区，重置信号量为0，为下一轮计算准备
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;   // 最后一个threadblock重置信号量
      }
      else {
        // Otherwise, the semaphore is incremented
        // 否则，递增信号量值，表示当前K分区已完成，下一个可以开始
        lock = threadblock_tile_offset.k() + 1;
      }

      // 释放信号量并设置新的值
      semaphore.release(lock);
    }
  } // operator() 函数结束
  
  // ========== 总结：整个Kernel的执行流程 ==========
  //
  // 1. 🎯 位置计算：确定当前threadblock负责的矩阵区域
  // 2. 🔧 迭代器设置：创建访问全局内存A、B矩阵的迭代器  
  // 3. 🚀 主要计算：执行多层次流水线的矩阵乘累加(A×B)
  //    - 多重双缓冲：全局内存↔共享内存↔寄存器
  //    - 计算与数据传输完美重叠，隐藏内存延迟
  // 4. ⚡ Epilogue：执行线性组合(alpha*A*B + beta*C)和激活函数
  // 5. 💾 结果写回：将最终结果存储到全局内存D
  // 6. 🔄 Split-K同步：支持K维度分割的大规模矩阵优化
  //
  // 这个设计体现了CUTLASS的核心理念：
  // - 硬件感知的优化：充分利用Tensor Core、共享内存、寄存器层次
  // - 内存带宽优化：通过流水线技术隐藏内存访问延迟  
  // - 可扩展性：支持任意大小的矩阵和各种优化策略
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

