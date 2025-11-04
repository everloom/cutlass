/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/mma_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Iterates over tiles of A operand in global memory 
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorA_,
  /// Iterates over tiles of A operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorA_,
  /// Iterates over tiles of B operand in global memory
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorB_,
  /// Iterates over tiles of B operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorB_,
  /// Data type of accumulator matrix
  typename ElementC_,
  /// Data type of accumulator matrix
  typename LayoutC_,
  /// Policy describing tuning details (concept: MmaPolicy)
  typename Policy_,
  /// Transformation applied to A operand
  typename TransformA_ = NumericArrayConverter<
    typename SmemIteratorA_::Element, 
    typename IteratorA_::Element, 
    IteratorA_::Fragment::kElements>,
  ///
  /// Transformation applied to B operand
  typename TransformB_ = NumericArrayConverter<
    typename SmemIteratorB_::Element, 
    typename IteratorB_::Element, 
    IteratorB_::Fragment::kElements>,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaPipelined : public MmaBase<Shape_, Policy_, 2> {
public:

  ///< Base class
  using Base = MmaBase<Shape_, Policy_, 2>;

  using Shape = Shape_;             ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using IteratorA = IteratorA_;     ///< Iterates over tiles of A operand in global memory
  using IteratorB = IteratorB_;     ///< Iterates over tiles of B operand in global memory
  using ElementC = ElementC_;       ///< Data type of accumulator matrix
  using LayoutC = LayoutC_;         ///< Layout of accumulator matrix
  using Policy = Policy_;           ///< Policy describing tuning details

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  using TransformA = TransformA_;
  using TransformB = TransformB_;

  //
  // Dependent types
  //

  /// Fragment of operand A loaded from global memory
  using FragmentA = typename IteratorA::Fragment;

  /// Fragment of operand B loaded from global memory
  using FragmentB = typename IteratorB::Fragment;

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Obtain the arch tag from the warp-level operator
  using ArchTag = typename Policy::Operator::ArchTag;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  // staticaly assert kStages for MmaPipelined is two (Double-buffered pipeline)
  static_assert((Base::kStages==2), "MmaPipelined requires kStages set to value 2");

protected:

  //
  // Data members
  //

  /// Warp-level MMA operator
  Operator warp_mma;

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  ///< transformation applied to A fragment
  TransformA transform_A_;

  ///< transformation applied to B fragment
  TransformB transform_B_;

  /// Shared memory write stage index
  int smem_write_stage_idx;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  MmaPipelined(
    typename Base::SharedStorage &shared_storage,       ///< Shared storage needed for internal use by threadblock-scoped GEMM
    int thread_idx,                                     ///< ID within the threadblock
    int warp_idx,                                       ///< ID of warp
    int lane_idx,                                       ///< ID of each thread within a warp
    TransformA transform_A = TransformA(),              ///< transformation applied to A fragment
    TransformB transform_B = TransformB()               ///< transformation applied to B fragment
  ):
    Base(shared_storage, thread_idx, warp_idx, lane_idx),
    smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
    smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
    transform_A_(transform_A),
    transform_B_(transform_B),
    smem_write_stage_idx(0)
  {

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }


  /// Advance shared memory write-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_write_stage()
  {
    // 这里其实是++(this->smem_iterator_A_)的意思
    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
    if (smem_write_stage_idx == 1) {
      this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
    }

    smem_write_stage_idx ^= 1;
  }

  /// Advance shared memory read- and write-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_stages()
  {
    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
    if (smem_write_stage_idx == 1) {
      // wrap write stage
      this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
    }
    else
    {
      // wrap read stage
      this->warp_tile_iterator_A_.add_tile_offset(
        {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset(
        {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
    }

    smem_write_stage_idx ^= 1;
  }


  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
  // GEMM前序阶段。通过预取前kStages-1个线程块主循环迭代所需的全局内存片段来启动全局内存到共享内存的流水线。
  CUTLASS_DEVICE
  void prologue(
    IteratorA &iterator_A,      ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B,      ///< [in|out] iterator over B operand in global memory
    int &gemm_k_iterations)     ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // The last kblock is loaded in the prolog

    // Load A fragment from global A
    FragmentA tb_frag_A; // 表示寄存器
    tb_frag_A.clear();
    iterator_A.load(tb_frag_A); // 这里表示把global mem的数据放到寄存器。ppt上也说了，这里cutlass的gemm的数据流动是，数据从gmem到reg再到smem(和常规的数据直接从gmem到smem有点区别)
    /*
    2025 1028更新：上面说了数据流动是gmem到reg再到smem是有误的，这个在reed大佬的这篇博客里面有讲https://zhuanlan.zhihu.com/p/666232173，
    对于ampere之前的架构，确实gmem到smem中间必须过reg。但从ampere开始，可以通过两条路径实现gmem到smem
    第一条：gmem到l2再到smem（l1 bypass）；第二条：gmem到l2再到l1再到smem
    关于ampere使用第一条路径还是第二条路径，貌似可以通过控制l1和l2的bypass与否进行控制，这块我还没仔细研究过
    // */

    ++iterator_A;

    // Load B fragment from global B
    FragmentB tb_frag_B;
    tb_frag_B.clear();
    iterator_B.load(tb_frag_B);
    ++iterator_B;

    // Store A and B fragments to shared
    this->smem_iterator_A_.store(transform_A_(tb_frag_A)); // 这里就是将reg中的数据搬到smem，这里transform A主要做强制类型转换
    // 这里问cursor为什么使用this指针，cursor说其实这里也可以不使用this指针，因为smem_iterator_B_就是MmaPipelined的一个成员变量，所以直接smem_iterator_B_.store也可以
    // 但这里之所以还要加上this指针，应该是为了代码风格统一，因为后面还有个warp_tile_iterator_A_使用了this指针，而warp_tile_iterator_A_使用this指针的原因和smem_iterator_B_不一样，但这里
    // 应该是为了和warp_tile_iterator_A_风格保持统一，所以加上了this指针
    this->smem_iterator_B_.store(transform_B_(tb_frag_B));

    // Advance write stage
    advance_smem_write_stage();
  }

  /// Wait until we have at least one completed global fetch stage
  CUTLASS_DEVICE
  void gmem_wait()
  {
    __syncthreads();
  }


  /// Perform the specified number of threadblock mainloop iterations of matrix
  /// multiply-accumulate.  Assumes prologue has been initiated.
  CUTLASS_DEVICE
  void gemm_iters(
    int gemm_k_iterations,        ///< number of threadblock mainloop iterations
    FragmentC &accum,             ///< [in|out] accumulator tile
    IteratorA &iterator_A,        ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B)        ///< [in|out] iterator over B operand in global memory
  {
    using WarpFragmentA = typename Operator::FragmentA;
    using WarpFragmentB = typename Operator::FragmentB;

    // Pair of fragments used to overlap shared memory loads and math instructions
    // 创建warp级别的寄存器双缓冲数组，用于重叠共享内存加载和数学计算指令
    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];

    /**
    说一下这里为什么使用this指针（注意和smem_iterator_A_使用this指针的情况区分），cursor是这样说的，因为warp_tile_iterator_A_是定义在基类MmaBase中的，而MmaBase是一个模板类，其中warp_tile_iterator_A_也使用到了模板传参
    所以在MmaBase的模板未实例化之前，是不知道warp_tile_iterator_A_是什么类型的，即这里的warp_tile_iterator_A_是一个依赖名称，在派生类中使用基类中定义的依赖名称，需要加上this指针
   */
    // Load A fragment from shared A
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    ++this->warp_tile_iterator_A_;

    // Load B fragment from shared B
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);
    ++this->warp_tile_iterator_B_;

    // Pair of fragments used to overlap global memory loads and math instructions;
    // 创建threadblock级别的fragment，用于重叠全局内存加载和数学计算（threadblock就是block的意思）
    // 这些fragment存储从全局内存加载的原始数据，后续会转换并存储到共享内存
    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

  /**
   这里插播一下，关于warp级别的fragment和threadblock级别的fragment的区别：
   对于threadblock级别（block级别）的FragmentA tb_frag_A：
   作用范围：整个threadblock的tile。数据量：包含整个threadblock tile的数据。使用者：所有threads协作处理
   对于warp级别的warp_frag_A[2]:
   作用范围：单个warp的tile。数据量：只包含一个warp处理的数据片段。使用者：单个warp内的32个线程

   kernel中数据的流动和threadblock级别的fragment和warp级别的fragment的关系：
        Global Memory
            ↓ (load)
        tb_frag_A/B ←── Threadblock级别fragment (临时中转站)
            ↓ (transform + store)
        Shared Memory
            ↓ (load)
        warp_frag_A/B ←── Warp级别fragment (计算用)
            ↓ (compute)
        Accumulator

    对于threadblock级别的fragment的数据流动：
     Global Memory → Registers (tb_frag) → Shared Memory
    对于warp级别的fragment的数据流动：
     Shared Memory → Registers (warp_frag) → Compute Units

     假设threadblock tile = 128x128, warp tile = 32x32
      ┌─────────────────────────────────┐
      │        Threadblock Tile         │ ← tb_frag_A/B 覆盖整个区域
      │  ┌───┐ ┌───┐ ┌───┐ ┌───┐      │
      │  │ W │ │ W │ │ W │ │ W │      │ ← 每个W是一个warp tile
      │  │ 0 │ │ 1 │ │ 2 │ │ 3 │      │   warp_frag_A/B 只覆盖一个W
      │  └───┘ └───┘ └───┘ └───┘      │
      │  ┌───┐ ┌───┐ ┌───┐ ┌───┐      │
      │  │ W │ │ W │ │ W │ │ W │      │
      │  │ 4 │ │ 5 │ │ 6 │ │ 7 │      │
      │  └───┘ └───┘ └───┘ └───┘      │
      └─────────────────────────────────┘     
   */


    // Avoid reading out of bounds
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    //
    // Mainloop 主循环：多层次流水线的核心
    //

    // Note: The main loop does not support Base::kWarpGemmIterations == 2.
    // 外层循环：遍历所有K维度的threadblock tile
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // Loop over GEMM K dimension
      //
      // 内层循环：遍历每个threadblock tile内的warp级MMA操作
      CUTLASS_PRAGMA_UNROLL
      /***
     这里我得着重说一下这里的Base::kWarpGemmIterations
     在b站up的cutlass课程的v3的代码里面，计算5120*4096和4096*4096两个矩阵的乘，然后每个threadblock tile的大小是128*256*64，warp tile的大小是64*64*64，mma使用的是8*8*16
     在这种设置下，上面的gemm_k_iterations的大小是64，然后下面这里的Base::kWarpGemmIterations的大小是4
     gemm_k_iterations的大小是64，这个好理解，因为threadblock tile的K维度大小是64，原始矩阵乘的K维度大小是4096，4096 / 64 = 64，表示K维度需要64次迭代
     然后warp tile的m*n的大小是64*64，但mma的m*n的大小是8*8，所以在m*n这两个维度上，需要调用8*8次mma
     但这8*8次mma不是在这里调用的，是下面warp_mma()调用的，也就是在warp层级进行调用的
     但是因为warp tile的k的维度是64，而mma的k的维度是16
     所以64 / 16 = 4，所以Base::kWarpGemmIterations为4，表示调用4次warp_mma()，总共调用了4*8*8次mma，然后将所有结果累加到accum上
     还有个问题，就是这里一个threadblock里面可以分8个warp，但这8个warp是并行执行的，所以不需要写关于warp的循环
     记住，一个block中的warp都是并行的，不需要写关于warp的循环
    */
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
        // as the case may be.
        // === 共享内存双缓冲管理：在最后一个warp MMA迭代时处理 ===
        // 类似于b站的那个sgemm v8的double buffer的最后一块的数据的处理
        if (warp_mma_k == Base::kWarpGemmIterations - 1) {

          // Write fragments to shared memory
          // 将之前从全局内存加载的数据写入共享内存
          // transform_A_和transform_B_进行类型转换（如int8到half等）
          this->smem_iterator_A_.store(transform_A_(tb_frag_A));

          this->smem_iterator_B_.store(transform_B_(tb_frag_B));

          // Wait until we have at least one completed global fetch stage

        // 等待至少一个全局内存获取阶段完成
        // 确保异步内存操作完成，维持流水线同步
          gmem_wait();

          // Advance smem read and write stages
          advance_smem_stages();
        }

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;
        // === 全局内存预取：在第一个warp MMA迭代时进行 ===
        if (warp_mma_k == 0) {

          // Load fragment from global A
          tb_frag_A.clear();
          iterator_A.load(tb_frag_A);
          ++iterator_A;

          // Load fragment from global B
          tb_frag_B.clear();
          iterator_B.load(tb_frag_B);
          ++iterator_B;

          // Avoid reading out of bounds if this was the last loop iteration
          iterator_A.clear_mask(gemm_k_iterations <= 2);
          iterator_B.clear_mask(gemm_k_iterations <= 2);
        }
        // === 核心计算：执行warp级矩阵乘累加 ===
        warp_mma(
          accum,
          warp_frag_A[warp_mma_k % 2],
          warp_frag_B[warp_mma_k % 2],
          accum);
      }
    }

  }


  /// Prepares the class for another prologue.
  CUTLASS_DEVICE
  void wind_down()
  {
    // First, increment remaining warp tiles to catch it up with the write stage.
    #pragma unroll
    for (int warp_mma_k = 1; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k)
    {
      this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k);
      this->warp_tile_iterator_B_.set_kgroup_index(warp_mma_k);

      ++this->warp_tile_iterator_A_;
      ++this->warp_tile_iterator_B_;
    }

    // If we bumped the read iterators to the end of the circular buffer, wrap them around to
    // align them with the write iterators
    if (smem_write_stage_idx == 0)
    {
      this->warp_tile_iterator_A_.add_tile_offset(
        {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset(
        {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,                            ///< number of iterations of the mainloop
    FragmentC &accum,                                 ///< destination accumulator tile
    IteratorA iterator_A,                             ///< iterator over A operand in global memory
    IteratorB iterator_B,                             ///< iterator over B operand in global memory
    FragmentC const &src_accum)                       ///< source accumulator tile
  {
    // Prologue
    prologue(iterator_A, iterator_B, gemm_k_iterations); // 这个函数的作用就是，将数据从gmem搬到reg，然后再搬到smem

    // Wait until we have at least one completed global fetch stage
    gmem_wait(); // 这里就是一个同步函数，因为前面是数据搬运，所以这里需要同步一下

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    // Perform the MAC-iterations
    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
