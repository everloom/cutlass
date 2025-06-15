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

  /**
共享内存中的双缓冲布局（假设kStages=2）：

矩阵A在共享内存中：
    K维度
  ┌─────┬─────┐
M │Stage│Stage│
  │  0  │  1  │
  └─────┴─────┘
   ↑write_stage_idx=0时写这里
         ↑write_stage_idx=1时写这里

矩阵B在共享内存中：
  ┌─────────┐ ← Stage 0 (write_stage_idx=0时写这里)
K │ Stage 0 │
  ├─────────┤
  │ Stage 1 │ N维度 (write_stage_idx=1时写这里)
  └─────────┘ ← Stage 1
   * * */

  /// Advance shared memory write-iterators to the next stage
  /// 
  /// 这个函数实现double buffer双缓冲机制中的写入阶段切换。在CUTLASS的双缓冲流水线中，
  /// 共享内存被分为两个stage（阶段），写入操作在两个stage之间轮换，
  /// 确保读写操作不会发生冲突。
  ///
  /// 双缓冲布局：
  /// 矩阵A: [Stage 0][Stage 1] (K维度分割)
  /// 矩阵B: [Stage 0]          (K维度分割)
  ///        [Stage 1]
  ///
  /// 工作原理：
  /// 1. 迭代器先前进到下一个写入位置
  /// 2. 如果到达缓冲区末尾，通过负偏移回绕到开始位置
  /// 3. 切换stage索引，为下一轮写入做准备
  CUTLASS_DEVICE
  void advance_smem_write_stage()
  {
      // 步骤1: 将共享内存写入迭代器前进到下一个tile位置
      // 这会让迭代器指向共享内存中的下一个写入位置
      ++this->smem_iterator_A_;    // A矩阵迭代器前进
      ++this->smem_iterator_B_;    // B矩阵迭代器前进

      // 步骤2: 检查是否需要回绕到循环缓冲区的开始位置
      // 当 smem_write_stage_idx == 1 时，表示刚才写入了Stage 1，
      // 下一次应该写入Stage 0，所以需要将迭代器拉回到缓冲区开始
      if (smem_write_stage_idx == 1) {
          // 为矩阵A添加负偏移，回绕到循环缓冲区的开始
          // {0, -Base::kStages}: M维度偏移=0，K维度向前回退kStages个tile
          // 由于kStages=2，这相当于从Stage 1回到Stage 0
          this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
          
          // 为矩阵B添加负偏移，回绕到循环缓冲区的开始  
          // {-Base::kStages, 0}: K维度向前回退kStages个tile，N维度偏移=0
          // 同样从Stage 1回到Stage 0
          this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
      }

      // 步骤3: 切换写入stage索引 (0 ↔ 1)
      // 使用XOR操作在0和1之间切换：0^1=1, 1^1=0
      // 这确保了下一次调用时能正确判断是否需要回绕
      smem_write_stage_idx ^= 1;
      
      // 执行流程示例：
      // 初始: smem_write_stage_idx=0, 迭代器指向Stage 0
      // 调用1: 迭代器前进到Stage 1, smem_write_stage_idx变为1
      // 调用2: 迭代器本来会超出边界，但通过负偏移回到Stage 0, 
      //        smem_write_stage_idx变为0
      // 调用3: 迭代器前进到Stage 1, smem_write_stage_idx变为1
      // ... 如此循环
  }



  /// Advance shared memory read- and write-iterators to the next stage
  /// 
  /// 这个函数是双缓冲流水线的核心同步机制，负责同时管理读写迭代器的推进和回绕。
  /// advance_smem_stages()这个函数不仅管理写入迭代器，还要处理读取迭代器的同步，在主循环中使用（相当于在advance_smem_write_stage基础上做了拓展）
  /// 而advance_smem_write_stage()只管理写入迭代器，在prologue阶段使用
  /// 
  /// 双缓冲工作原理：
  /// - Stage 0和Stage 1在共享内存中交替使用
  /// - 写入操作总是写入到"当前写入stage"
  /// - 读取操作总是从"另一个stage"读取（避免读写冲突）
  /// - 每次调用此函数，读写stage都会切换
  ///
  /// 时间轴示例：
  /// T0: 写→Stage0, 读→无     (初始化阶段)
  /// T1: 写→Stage1, 读→Stage0 (正常流水线)
  /// T2: 写→Stage0, 读→Stage1 (正常流水线)
  /// T3: 写→Stage1, 读→Stage0 (正常流水线)
  CUTLASS_DEVICE
  void advance_smem_stages()
  {
    // === 步骤1: 推进共享内存写入迭代器 ===
    // 将写入迭代器前进到下一个tile位置
    // 这些迭代器负责将从全局内存加载的数据写入共享内存
    ++this->smem_iterator_A_;    // A矩阵写入迭代器前进
    ++this->smem_iterator_B_;    // B矩阵写入迭代器前进

    // === 步骤2: 处理循环缓冲区的回绕逻辑 ===
    // 双缓冲需要在两个stage之间循环，当迭代器超出边界时需要回绕
    
    // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
    if (smem_write_stage_idx == 1) {
      // === 情况1: 写入stage回绕 (从Stage 1回到Stage 0) ===
      // 当前写入stage索引为1，意味着刚才写入了Stage 1
      // 下一次写入应该回到Stage 0，所以需要将写入迭代器拉回到循环缓冲区开始
      
      // wrap write stage
      // 为A矩阵写入迭代器添加负偏移，回绕到Stage 0
      // {0, -Base::kStages}: M维度偏移=0，K维度向前回退kStages个tile
      // 由于kStages=2，这相当于从Stage 1的位置回退到Stage 0
      this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      
      // 为B矩阵写入迭代器添加负偏移，回绕到Stage 0  
      // {-Base::kStages, 0}: K维度向前回退kStages个tile，N维度偏移=0
      this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
    }
    else
    {
      // === 情况2: 读取stage回绕 (从Stage 0回到Stage 1) ===
      // 当前写入stage索引为0，意味着读取迭代器需要从Stage 0回绕到Stage 1
      // 读取迭代器的回绕比写入迭代器复杂，因为它不仅要考虑stage切换，
      // 还要考虑warp内部的K维度迭代（kWarpGemmIterations）
      
      // wrap read stage
      // 计算读取迭代器的回绕偏移量：
      // -Base::kStages: 双缓冲的stage数量 (2)
      // * Policy::kPartitionsK: K维度分区数 (通常为1)  
      // * Base::kWarpGemmIterations: warp内部K维度迭代次数 (4)
      // 总偏移 = -2 * 1 * 4 = -8
      //
      // 为什么需要这么大的偏移？
      // 因为读取迭代器在一个外层K迭代中会前进kWarpGemmIterations次，
      // 当需要回绕时，必须回退所有这些前进的步数
      this->warp_tile_iterator_A_.add_tile_offset(
        {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset(
        {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
    }

    // === 步骤3: 切换写入stage索引 ===
    // 使用XOR操作在0和1之间切换：0^1=1, 1^1=0
    // 这确保了下一次调用时能正确判断应该处理哪种回绕情况
    smem_write_stage_idx ^= 1;
    
    // === 执行流程总结 ===
    // 调用1: smem_write_stage_idx=0→1, 写入迭代器前进, 无回绕
    // 调用2: smem_write_stage_idx=1→0, 写入迭代器前进+回绕, 读取迭代器回绕
    // 调用3: smem_write_stage_idx=0→1, 写入迭代器前进, 无回绕  
    // 调用4: smem_write_stage_idx=1→0, 写入迭代器前进+回绕, 读取迭代器回绕
    // ... 如此循环，实现完美的双缓冲同步
  }


  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
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
    ++iterator_A;

    // Load B fragment from global B
    FragmentB tb_frag_B;
    tb_frag_B.clear();
    iterator_B.load(tb_frag_B);
    ++iterator_B;

    // Store A and B fragments to shared
    this->smem_iterator_A_.store(transform_A_(tb_frag_A));// 这里就是将reg中的数据搬到smem，这里transform A主要做强制类型转换
    this->smem_iterator_B_.store(transform_B_(tb_frag_B));

    // Advance write stage
    /** cursor说在上面已经把数据载入到smem后还要调用advance_smem_write_stage()，是为了改变迭代器指向的位置
    这里smem使用双buffer，所以smem布局如下
    共享内存布局 (双缓冲):
    ┌─────────┬─────────┐
    │ Stage 0 │ Stage 1 │  ← K维度分为两个stage
    └─────────┴─────────┘
        ↑
    smem_iterator_A_ 当前指向这里
    
    然后下面需要调用advance_smem_write_stage，使smem_iterator_A_指向stage 1
    **/
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
/// 
/// 这是CUTLASS双缓冲流水线的核心主循环函数，实现了高效的矩阵乘法计算。
/// 该函数使用多层次的双缓冲机制来隐藏内存访问延迟：
/// 1. 共享内存层双缓冲：在Stage 0和Stage 1之间切换
/// 2. 寄存器层双缓冲：在warp_frag[0]和warp_frag[1]之间切换
/// 3. 全局内存预取：与计算重叠进行异步加载
CUTLASS_DEVICE
void gemm_iters(
  int gemm_k_iterations,        ///< number of threadblock mainloop iterations
  FragmentC &accum,             ///< [in|out] accumulator tile
  IteratorA &iterator_A,        ///< [in|out] iterator over A operand in global memory
  IteratorB &iterator_B)        ///< [in|out] iterator over B operand in global memory
{
  // 定义warp级别的fragment类型别名
  using WarpFragmentA = typename Operator::FragmentA;  // A矩阵的warp级fragment类型
  using WarpFragmentB = typename Operator::FragmentB;  // B矩阵的warp级fragment类型

  // Pair of fragments used to overlap shared memory loads and math instructions
  // 创建warp级别的寄存器双缓冲数组，用于重叠共享内存加载和数学计算指令
  // 这是第二层双缓冲：寄存器层面的ping-pong缓冲

  WarpFragmentA warp_frag_A[2]; // 这里也是类似于smem的double buffer，思想都是一样的
  WarpFragmentB warp_frag_B[2]; // 两个fragment在使用和加载之间切换

  // === 初始化阶段：预加载第一个warp fragment ===
  /**
    时间轴: 初始化 → 主循环迭代0 → 主循环迭代1 → ...

  初始化:
    warp_frag_A[0] ← load(kgroup=0)    // 这三行代码的作用
    warp_frag_A[1] ← 空闲

  迭代0:
    warp_frag_A[1] ← load(kgroup=1)    // 预加载
    计算使用 warp_frag_A[0]           // 使用初始化的数据

  迭代1:
    warp_frag_A[0] ← load(kgroup=2)    // 预加载  
    计算使用 warp_frag_A[1]           // 使用上次预加载的数据

  共享内存中的数据布局：
  ┌─────────┬─────────┬─────────┬─────────┐
  │ kgroup0 │ kgroup1 │ kgroup2 │ kgroup3 │
  └─────────┴─────────┴─────────┴─────────┘
      ↑
    初始位置(set_kgroup_index(0))

  加载到寄存器：
  warp_frag_A[0] ← kgroup0的数据
  warp_frag_A[1] ← (待加载)
  * */
  
  // Load A fragment from shared A
  // 设置K组索引为0，开始处理第一个K维度分组
  // K维度被分为多个组（kgroup），每组对应一次warp MMA操作
  // 设置为0表示从第一个K组开始
  // 
  // 假设Base::kWarpGemmIterations = 4，那么K维度分为4组：
  // [kgroup 0][kgroup 1][kgroup 2][kgroup 3]
  //    ↑
  //  从这里开始
  this->warp_tile_iterator_A_.set_kgroup_index(0);
  // 从共享内存加载第一个A fragment到warp_frag_A[0]
  // 这是寄存器双缓冲的初始化，为主循环做准备
  // 从共享内存加载数据到warp级别的fragment
  // warp_frag_A[0] 是双缓冲数组的第一个缓冲区
  // 
  // 数据流：Shared Memory → warp_frag_A[0] (寄存器)
  this->warp_tile_iterator_A_.load(warp_frag_A[0]); // 用于从共享内存加载warp级别的数据进行计算
  // 迭代器前进到下一个位置，为下次加载准备
  ++this->warp_tile_iterator_A_;

  // Load B fragment from shared B
  // 对B矩阵执行相同的初始化操作
  this->warp_tile_iterator_B_.set_kgroup_index(0);
  this->warp_tile_iterator_B_.load(warp_frag_B[0]);
  ++this->warp_tile_iterator_B_;

  // Pair of fragments used to overlap global memory loads and math instructions;
  // 创建threadblock级别的fragment，用于重叠全局内存加载和数学计算（threadblock就是block的意思）
  // 这些fragment存储从全局内存加载的原始数据，后续会转换并存储到共享内存
  FragmentA tb_frag_A;  // threadblock级别的A fragment (寄存器)
  FragmentB tb_frag_B;  // threadblock级别的B fragment (寄存器)

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

  // === 边界检查和掩码设置 ===
  
  // Avoid reading out of bounds
  // 如果只剩1次或更少的迭代，清除掩码以避免越界访问
  // 这是一种预防性措施，确保不会读取超出矩阵边界的数据
  iterator_A.clear_mask(gemm_k_iterations <= 1);
  iterator_B.clear_mask(gemm_k_iterations <= 1);

  //
  // Mainloop - 主循环：多层次流水线的核心
  //

  // Note: The main loop does not support Base::kWarpGemmIterations == 2.
  // 注意：主循环不支持Base::kWarpGemmIterations == 2的情况
  
  // 外层循环：遍历所有K维度的threadblock tile
  CUTLASS_GEMM_LOOP
  for (; gemm_k_iterations > 0; --gemm_k_iterations) {
    //
    // Loop over GEMM K dimension
    // 内层循环：遍历每个threadblock tile内的warp级MMA操作
    //

    // 展开内层循环以提高性能
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
        // 推进共享内存的读写阶段
        // 这里调用advance_smem_stages的目的时为了改变迭代器指向的smem的stage的位置
        // 这会切换共享内存双缓冲的stage (Stage 0 ↔ Stage 1)
        /**
        完整的时序图
                时间轴:    Prologue    主循环迭代1    主循环迭代2
                ┌─────────┐  ┌─────────┐   ┌─────────┐
        写入位置: │ Stage 0 │  │ Stage 1 │   │ Stage 0 │
                └─────────┘  └─────────┘   └─────────┘
        读取位置:     无       │ Stage 0 │   │ Stage 1 │
                            └─────────┘   └─────────┘

        调用函数: advance_    advance_     advance_
                smem_write_  smem_        smem_
                stage()      stages()     stages()
         */
        advance_smem_stages();
      }

      // === 寄存器级双缓冲：预加载下一个warp fragment ===
      
      // 设置下一个K组的索引，使用模运算实现循环
      // (warp_mma_k + 1) % Base::kWarpGemmIterations 确保索引在有效范围内循环
      this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);

      // 从共享内存预加载下一个iteration需要的数据到另一个fragment
      // (warp_mma_k + 1) % 2 实现在warp_frag[0]和warp_frag[1]之间的ping-pong切换
      // 当前iteration使用一个fragment进行计算时，同时加载数据到另一个fragment
      this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
      this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

      // 推进warp tile迭代器到下一个位置
      ++this->warp_tile_iterator_A_;
      ++this->warp_tile_iterator_B_;

      // === 全局内存预取：在第一个warp MMA迭代时进行 ===
      if (warp_mma_k == 0) {

        // Load fragment from global A
        // 从全局内存加载下一个threadblock tile的A数据
        // 这个操作与当前的warp MMA计算重叠，隐藏全局内存访问延迟
        tb_frag_A.clear();           // 清零fragment
        iterator_A.load(tb_frag_A);  // 异步加载数据到寄存器 // 用于从全局内存加载整个threadblock tile
        ++iterator_A;                // 迭代器前进到下一个tile

        // Load fragment from global B
        // 对B矩阵执行相同的全局内存预取操作
        tb_frag_B.clear();
        iterator_B.load(tb_frag_B);
        ++iterator_B;

        // Avoid reading out of bounds if this was the last loop iteration
        // 如果这是倒数第二次迭代，清除掩码以避免在下次迭代时越界
        // gemm_k_iterations <= 2 表示当前是倒数第二次或最后一次迭代
        iterator_A.clear_mask(gemm_k_iterations <= 2);
        iterator_B.clear_mask(gemm_k_iterations <= 2);
      }

      // === 核心计算：执行warp级矩阵乘累加 ===
      
      // 使用当前iteration的fragment进行矩阵乘累加计算
      // warp_mma_k % 2 确保使用当前iteration应该使用的fragment
      // 当加载下一个数据到warp_frag[(warp_mma_k+1)%2]时，
      // 计算使用warp_frag[warp_mma_k%2]中的数据
      warp_mma(
        accum,                          // [in|out] 累加器，存储计算结果
        warp_frag_A[warp_mma_k % 2],   // 当前A fragment (来自寄存器双缓冲)
        warp_frag_B[warp_mma_k % 2],   // 当前B fragment (来自寄存器双缓冲)  
        accum);                         // 累加到同一个累加器

      // === 流水线时序说明 ===
      // 在每个warp_mma_k迭代中，同时发生：
      // 1. 使用warp_frag[warp_mma_k % 2]进行当前计算
      // 2. 加载数据到warp_frag[(warp_mma_k + 1) % 2]为下次计算准备
      // 3. 在warp_mma_k==0时，从全局内存预取下一个threadblock tile
      // 4. 在warp_mma_k==Base::kWarpGemmIterations-1时，管理共享内存双缓冲
      //
      // 这种设计实现了三层流水线的完美重叠：
      // - 全局内存访问 ↔ 共享内存操作
      // - 共享内存访问 ↔ 寄存器操作  
      // - 寄存器加载 ↔ 计算执行
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
    gmem_wait();// 这里就是一个同步函数，因为前面是数据搬运，所以这里需要同步一下

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
