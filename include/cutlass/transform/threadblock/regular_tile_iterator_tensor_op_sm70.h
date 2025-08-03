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
    \brief Templates implementing loading of tiles from pitch-linear rank=2 tensors.

    This iterator uses masks to guard out-of-bounds accesses and visits the last "residue" tile
    first, with the objective of minimizing predicate mask updates during steady-state operation.

    A precomputed "Params" object minimizes the amount of state that must be stored in registers,
    and integer addition is used to advance the pointer through memory.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"

#include "cutlass/transform/threadblock/regular_tile_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for congruous arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
  Shape_,
  Element_,
  layout::VoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
  AdvanceRank,
  ThreadMap_,
  Alignment> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::VoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {

    /// This iterator is specialized for an access size that is 128 bits in length.
    static int const kAccessSizeInBits = 128;

    static_assert(
      sizeof_bits<Element_>::value * ThreadMap::kElementsPerAccess == kAccessSizeInBits,
      "This iterator requires a policy whose access size is 128bs");

    ///< Number of pointers
    static int const kPointerCount = (ThreadMap::Iterations::kStrided > 1 ? 2 : 1);
  };


private:

  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, ThreadMap::Iterations::kCount * Layout::kElementsPerAccess>;

private:

  //
  // Data members
  //

  /// Stride value
  StrideIndex stride_;

  /// Internal pointer to first access of tile
  AccessType * pointer_[Detail::kPointerCount];

  /// Internal byte offset
  Index byte_offset_;

public:

  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): stride_(ref.stride(0) / Layout::kElementsPerAccess), byte_offset_(0) {

    layout::PitchLinearCoord thread_offset_base = ThreadMap::initial_offset(thread_id);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {

      // This is the offset of a thread within a threadblock tile for a specific pointer
      // (units of elements)
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
        thread_offset_base + layout::PitchLinearCoord{0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};

      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(ref.data() + ref.offset(thread_offset_in_threadblock_tile));
    }
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {

    byte_offset_ += pointer_offset * sizeof(Element);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {

    add_pointer_offset((kAdvanceRank ? Shape::kStrided * stride_ * Layout::kElementsPerAccess : Shape::kContiguous));

    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {

    RegularTileIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    add_pointer_offset(
      coord.contiguous() * Shape::kContiguous / ThreadMap::kElementsPerAccess +
      coord.strided() * Shape::kStrided * stride_ * Layout::kElementsPerAccess
    );
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    Index vec_pointer_offset = pointer_offset / ThreadMap::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType *access_ptr = pointer_[s & 1];
      int stride_idx = (s & ~1);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int access_offset = stride_idx * ThreadMap::Delta::kStrided * stride_ +
            c * ThreadMap::Delta::kContiguous / ThreadMap::kElementsPerAccess +
            vec_pointer_offset;

        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char const *access_byte_ptr = reinterpret_cast<char const *>(access_ptr + access_offset);

        frag_ptr[access_idx] = *reinterpret_cast<AccessType const *>(access_byte_ptr + byte_offset_);
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,
    Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    Index vec_pointer_offset = pointer_offset / ThreadMap::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType *access_ptr = pointer_[s & 1];
      int stride_idx = (s & ~1);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int access_offset = stride_idx * ThreadMap::Delta::kStrided * stride_ +
          c * ThreadMap::Delta::kContiguous / ThreadMap::kElementsPerAccess +
          vec_pointer_offset;

        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char *access_byte_ptr = reinterpret_cast<char *>(access_ptr + access_offset);

        *reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_) = frag_ptr[access_idx];
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Tile Iterator specialized for column-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
  Shape_,
  Element_,
  layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
  AdvanceRank,
  ThreadMap_,
  Alignment> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for column-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
    layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
    Element,
    layout::VoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
    (kAdvanceRank == 0 ? 0 : 1),
    ThreadMap_>;

public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {

    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {

    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,
    Index pointer_offset) {

    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for row-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
  Shape_,
  Element_,
  layout::RowMajorVoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
  AdvanceRank,
  ThreadMap_,
  Alignment> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for row-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorVoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::VoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
    (kAdvanceRank == 0 ? 1 : 0),
    ThreadMap_>;

public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {

    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  // 对这个函数的理解可以参考include/cutlass/transform/threadblock/predicated_tile_iterator.h中PredicatedTileIterator operator++(int)的注释
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {

    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,
    Index pointer_offset) {

    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};
/// Tile iterator specialized for congruous arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
  Shape_,
  Element_,
  layout::VoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>,
  AdvanceRank,
  ThreadMap_,
  Alignment> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::VoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {

    /// This iterator is specialized for an access size that is 128 bits in length.
    static int const kAccessSizeInBits = 128;

    static_assert(
      sizeof_bits<Element_>::value * ThreadMap::kElementsPerAccess == kAccessSizeInBits,
      "This iterator requires a policy whose access size is 128bs");

    ///< Number of pointers
    static int const kPointerCount = (ThreadMap::Iterations::kStrided > 1 ? 2 : 1);
  };


private:

  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, ThreadMap::Iterations::kCount * Layout::kElementsPerAccess>;

private:

  //
  // Data members
  //

  /// Stride value
  StrideIndex stride_;

  /// Internal pointer to first access of tile
  AccessType * pointer_[Detail::kPointerCount];

  /// Internal byte offset
  Index byte_offset_;

public:

  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): stride_(ref.stride(0) / Layout::kElementsPerAccess), byte_offset_(0) {

    layout::PitchLinearCoord thread_offset_base = ThreadMap::initial_offset(thread_id);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {

      // This is the offset of a thread within a threadblock tile for a specific pointer
      // (units of elements)
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
        thread_offset_base + layout::PitchLinearCoord{0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};

      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(ref.data() + ref.offset(thread_offset_in_threadblock_tile));
    }
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {

    byte_offset_ += pointer_offset * sizeof(Element);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {

    add_pointer_offset((kAdvanceRank ? Shape::kStrided * stride_ * Layout::kElementsPerAccess : Shape::kContiguous));

    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {

    RegularTileIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    add_pointer_offset(
      coord.contiguous() * Shape::kContiguous / ThreadMap::kElementsPerAccess +
      coord.strided() * Shape::kStrided * stride_ * Layout::kElementsPerAccess
    );
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    Index vec_pointer_offset = pointer_offset / ThreadMap::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType *access_ptr = pointer_[s & 1];
      int stride_idx = (s & ~1);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int access_offset = stride_idx * ThreadMap::Delta::kStrided * stride_ +
            c * ThreadMap::Delta::kContiguous / ThreadMap::kElementsPerAccess +
            vec_pointer_offset;

        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char const *access_byte_ptr = reinterpret_cast<char const *>(access_ptr + access_offset);

        frag_ptr[access_idx] = *reinterpret_cast<AccessType const *>(access_byte_ptr + byte_offset_);
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// 将fragment数据存储到内存中，支持额外的指针偏移
  /// 
  /// 这个函数是CUTLASS中tensor core操作的核心存储函数，负责将寄存器中的fragment数据
  /// 高效地存储到共享内存或全局内存中。它处理复杂的内存布局、向量化访问和双缓冲机制。
  ///
  /// @param frag Fragment对象，包含要存储的数据（通常来自寄存器）
  /// @param pointer_offset 额外的指针偏移量，用于支持动态内存位置调整
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,           ///< 输入：要存储的fragment数据
    Index pointer_offset) {         ///< 输入：指针偏移量（以元素为单位）

    // === 步骤1: Fragment数据类型转换 ===
    // 将fragment转换为AccessType指针，以便进行向量化访问
    // AccessType通常是经过优化的数据类型（如float4, int4等），
    // 允许一次访问多个元素，提高内存带宽利用率
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    // === 步骤2: 计算向量化指针偏移 ===
    // 将以元素为单位的偏移量转换为以AccessType为单位的偏移量
    // 例如：如果AccessType是float4（128bit），kElementsPerAccess=4，
    // 那么pointer_offset=8个float元素对应vec_pointer_offset=2个float4访问
    Index vec_pointer_offset = pointer_offset / ThreadMap::kElementsPerAccess;

    // === 步骤3: 外层循环 - 遍历Strided维度 ===
    // Strided维度通常对应矩阵的行方向或较大的跨步维度
    // 这个循环处理fragment在该维度上的所有迭代
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      // === 步骤3.1: 双缓冲指针选择 ===
      // 使用 s & 1 实现双缓冲机制：偶数迭代使用pointer_[0]，奇数迭代使用pointer_[1]
      // 这是tensor core操作中常见的优化技术，用于隐藏内存访问延迟
      // 当一个缓冲区在进行内存访问时，另一个缓冲区可以同时进行计算
      AccessType *access_ptr = pointer_[s & 1];
      
      // === 步骤3.2: 计算Strided索引 ===
      // s & ~1 将奇数索引转换为对应的偶数索引
      // 例如：s=0→stride_idx=0, s=1→stride_idx=0, s=2→stride_idx=2, s=3→stride_idx=2
      // 这种模式配合双缓冲，确保相邻的两次迭代使用不同的缓冲区但具有相似的stride模式
      int stride_idx = (s & ~1);

      // === 步骤4: 内层循环 - 遍历Contiguous维度 ===
      // Contiguous维度通常对应矩阵的列方向或连续内存方向
      // 这个循环处理每个strided位置上的所有连续元素
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        // === 步骤4.1: 计算详细的内存访问偏移量 ===
        // 这是一个复合偏移量计算，包含三个部分：
        // 1. stride_idx * ThreadMap::Delta::kStrided * stride_: 
        //    Strided维度的基础偏移，stride_是内存布局中行间的跨步
        // 2. c * ThreadMap::Delta::kContiguous / ThreadMap::kElementsPerAccess:
        //    Contiguous维度的偏移，除以kElementsPerAccess是因为使用向量化访问
        // 3. vec_pointer_offset: 
        //    外部传入的额外偏移量（已转换为向量化单位）
        int access_offset = stride_idx * ThreadMap::Delta::kStrided * stride_ +
          c * ThreadMap::Delta::kContiguous / ThreadMap::kElementsPerAccess +
          vec_pointer_offset;

        // === 步骤4.2: 计算Fragment中的数据索引 ===
        // 将二维的(s,c)坐标转换为一维的线性索引
        // 这个索引用于从fragment中提取对应位置的数据
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        // === 步骤4.3: 计算最终的字节级内存地址 ===
        // 首先计算基础地址：access_ptr + access_offset
        // 然后转换为字节指针，以便进行精确的字节级偏移调整
        char *access_byte_ptr = reinterpret_cast<char *>(access_ptr + access_offset);

        // === 步骤4.4: 执行实际的内存存储操作 ===
        // 1. access_byte_ptr + byte_offset_: 添加字节级偏移量，处理内存对齐等细节
        // 2. reinterpret_cast<AccessType *>(...): 转换回AccessType指针类型
        // 3. *(...) = frag_ptr[access_idx]: 将fragment中的数据写入计算出的内存位置
        //
        // 这个操作是整个函数的核心：将寄存器中的fragment数据高效地存储到内存中
        *reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_) = frag_ptr[access_idx];
      }
    }
    
    // === 函数执行流程总结 ===
    // 1. 数据预处理：将fragment转换为向量化访问格式
    // 2. 双重循环遍历：外层处理strided维度，内层处理contiguous维度  
    // 3. 双缓冲优化：使用两个指针交替访问，隐藏内存延迟
    // 4. 复杂寻址：综合考虑stride、向量化、偏移等因素计算精确内存地址
    // 5. 高效存储：使用向量化指令将数据从寄存器写入内存
    //
    // 这种设计在GPU上能够充分利用内存带宽和tensor core的高性能特性
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for column-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
  Shape_,
  Element_,
  layout::ColumnMajorVoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>,
  AdvanceRank,
  ThreadMap_,
  Alignment> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for column-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorVoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
    layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
    Element,
    layout::VoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>,
    (kAdvanceRank == 0 ? 0 : 1),
    ThreadMap_>;

public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {

    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {

    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,
    Index pointer_offset) {

    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for row-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
  Shape_,
  Element_,
  layout::RowMajorVoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>,
  AdvanceRank,
  ThreadMap_,
  Alignment> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for row-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorVoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::VoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>,
    (kAdvanceRank == 0 ? 1 : 0),
    ThreadMap_>;

public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {

    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {

    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,
    Index pointer_offset) {

    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};


/// Tile iterator specialized for crosswise arrangements for TensorOps.
///
/// Volta TN SMEM layout is a little diffrent:
/// Crosseised elements will be stored in a line, while contiguous elements
/// sre stored in line-by-line.
/// Padding is used to reduce SMEM bank conflicts.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
    Shape_, Element_,
    layout::VoltaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                               Shape_::kContiguous>,
    AdvanceRank, ThreadMap_, Alignment> {

 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::VoltaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                                 Shape::kContiguous>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {

    ///< Number of pointers
    static int const kPointerCount = (ThreadMap::Iterations::kStrided > 1 ? 2 : 1);

    /// Iterations for the kElementsPerAccess of ThreadMap
    static int const kIterarionsPerAccess =
        ThreadMap::kElementsPerAccess / Layout::kElementsPerAccess;

    /// Contiguous elements per line
    static int const kContiguousElementsPerLine = 4;
  };

 private:
  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  /// Fragment object to be loaded or stored
  using Fragment =
      Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

 private:
  //
  // Data members
  //

  /// The crosswised elements will be stored in a line.
  /// line_size is size of crosswised dimension plus padding.
  /// in units of AccessType
  Index line_size;

  /// Internal pointer to first access of tile
  AccessType *pointer_[Detail::kPointerCount];

  /// Internal byte offset
  Index byte_offset_;


 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
      : line_size(ref.stride(0) * Detail::kContiguousElementsPerLine / Layout::kElementsPerAccess),
        byte_offset_(0) {

    layout::PitchLinearCoord thread_offset_base =
        ThreadMap::initial_offset(thread_id);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      // This is the offset of a thread within a threadblock tile for a specific
      // pointer (units of elements)
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
          thread_offset_base +
          layout::PitchLinearCoord{
              0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};

      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(
          ref.data() + ref.offset(thread_offset_in_threadblock_tile));
    }
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_offset_ += pointer_offset * sizeof(Element);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    // (Shape::kContiguous/Layout::kElementsPerAccess)*
    //   line_size * Layout::kElementsPerAccess
    add_pointer_offset(Shape::kContiguous * line_size);
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    add_pointer_offset((coord.contiguous() * (Shape::kContiguous / Layout::kElementsPerAccess) *
                       line_size + coord.strided() * Shape::kStrided) *
                       Layout::kElementsPerAccess);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    Index vec_pointer_offset = pointer_offset / Layout::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      AccessType *access_ptr = pointer_[(s & 1) ^ (s / 2)];

      access_ptr += 16 * (s / 2);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < Detail::kIterarionsPerAccess; ++i) {

          int access_offset = 
            c * ThreadMap::Delta::kContiguous / Detail::kContiguousElementsPerLine * line_size +
            vec_pointer_offset + i * line_size;

          int access_idx = (c + s * ThreadMap::Iterations::kContiguous) *
            Detail::kIterarionsPerAccess + i;

          char const *access_byte_ptr = reinterpret_cast<char const*>(access_ptr + access_offset);

          frag_ptr[access_idx] = *reinterpret_cast<AccessType const *>(
              access_byte_ptr + byte_offset_);
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    Index vec_pointer_offset = pointer_offset / Layout::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType *access_ptr = pointer_[(s & 1) ^ ((s >> 1) & 1)];

      access_ptr += 16 * (s / 2) + vec_pointer_offset;

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < Detail::kIterarionsPerAccess; ++i) {

          int access_offset = 
            c * ThreadMap::Delta::kContiguous / Detail::kContiguousElementsPerLine * line_size + i * line_size;

          int access_idx = (c + s * ThreadMap::Iterations::kContiguous) *
            Detail::kIterarionsPerAccess + i;

          char *access_byte_ptr = reinterpret_cast<char *>(access_ptr + access_offset);

          *reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_) =
              frag_ptr[access_idx];
        }
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for column-major crosswise TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<Shape_, Element_,
                          layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<
                              sizeof_bits<Element_>::value, Shape_::kRow>,
                          AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for column-major iterator may along advance along the "
      "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, Shape::kRow>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::VoltaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            Shape::kRow>,
      (kAdvanceRank == 0 ? 0 : 1), ThreadMap_>;

 public:
  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

 private:
  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
      : iterator_({ref.data(), ref.stride()}, thread_id) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for row-major crosswise TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,  
  int Alignment
>
class RegularTileIterator<Shape_, Element_,
                          layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
                              sizeof_bits<Element_>::value, Shape_::kColumn>,
                          AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for row-major iterator may along advance along the "
      "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, Shape::kColumn>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::VoltaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                                 Shape::kColumn>,
      (kAdvanceRank == 0 ? 1 : 0), ThreadMap_>;

 public:
  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

 private:
  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
      : iterator_({ref.data(), ref.stride()}, thread_id) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace transform
} // namespace cutlass
