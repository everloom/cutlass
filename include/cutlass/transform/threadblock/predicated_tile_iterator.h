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
    \brief Templates implementing loading of tiles from pitch-linear rank=2 tensors. 

    This iterator uses masks to guard out-of-bounds accesses. The first tile this
    iterator visits maybe partial, then the remaining tiles are complete. So, we 
    only need to compute the predicates twice, once before the first tile and 
    once for the remaining full tiles which can share the same predicates.

    A precomputed "Params" object minimizes the amount of state that must be stored in registers,
    and integer addition is used to advance the pointer through memory.
*/

#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// PredicatedTileIterator
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
/// Regular tile iterator using a precomputed control structure to minimize register liveness
/// and integer arithmetic.
///
/// Layout is assumed to be invariant at the time the precomputed "Params" object is constructed.
///
/// Base pointer and tensor extents may be specified at the time the iterator is constructed.
/// Subsequently, they are assumed to be immutable.
///
/// Adding a logical coordinate offset may be performed at the time the iterator is constructed.
/// Subsequent additions to logical coordinate offset may be performed but are relatively expensive.
///
/// Visitation order is intended to first visit a "residual" tile that may be partially full in
/// both the advance dimension and the steady-state dimension. This is assumed to be the last
/// tile in the iteration sequence. Advancing an iterator that has just been constructed moves to
/// the first tile that is full in the advance dimension and recomputes predicates. Subsequent
/// accesses may be performed without updating internal predicates and are efficient in terms of
/// live register state and pointer arithmetic instructions.
///
/// To be efficient, this assumes the iterator will be dereferenced and advanced at least once
/// outside any looping structure to minimize integer arithmetic. 
///
/// Acceses out of bounds are safe so long as `clear_mask()` is called prior to dereferencing
/// the iterator.
///
///
/// Example:
///
/// An efficient pipeline structure may be constructed as follows:
///
// template <typename Iterator>
// __global__ void kernel(
//   typename Iterator::Params params, 
//   typename Iterator::Element *ptr,
//   TensorCoord extent) {
//
//   typename Iterator::Fragment fragment;
//
//   TensorCoord threadblock_offset(0, 0);
//
//   Iterator iter(params, ptr, extent, threadIdx.x, threadblock_offsets);
//
//
//   fragment = *iter;        // load "residue" tile first
//   ++iter;                  // advance to first "steady state" tile and update internal masks
//
//
//   #pragma unroll
//   for (int i = Remaining - 1; i >= 0; --i) {
//
//     f(fragment);
//
//     if (!i) {
//       iter.clear_mask();   // light-weight operation to clear masks - subsequent loads become NO-OPs.
//     }
//  
//     fragment = *iter;      // load tile during "steady state" phase
//     ++iter;                // advance to next tile - lightweight due to steady-state masks
//   }
// }
//
// void host(TensorView<Element, 2, layout::PitchLinear> view) {
//
//   using Iterator = transform::threadblock::PredicatedTileIterator;
//
//   typename Iterator::Params params(view.layout());
//
//   kernel<Iterator>(params, view.data());
// }
///
///
template <
  typename Shape,
  typename Element,
  typename Layout,
  int AdvanceRank,
  typename ThreadMap,
  int AccessSize = ThreadMap::kElementsPerAccess,
  bool Gather = false,
  typename PermuteLayout = layout::NoPermute
>
class PredicatedTileIterator;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///

// 这里是对PitchLinear的layout进行的特化
// 这里AdvanceRank表示迭代器在哪个维度上推进(0表示contiguous维度，1表示strided维度)
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize, bool Gather, typename PermuteLayout>
class PredicatedTileIterator<Shape_, Element_, layout::PitchLinear, AdvanceRank,
                             ThreadMap_, AccessSize, Gather, PermuteLayout> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  /// Type used for internal memory accesses
  using AccessType = AlignedArray<Element, AccessSize, (AccessSize * sizeof_bits<Element>::value / 8)>;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator =
      PredicatedTileAccessIterator<Shape, Element, Layout, kAdvanceRank,
                                   ThreadMap, AccessType, Gather, PermuteLayout>;

  static int const kAccessesPerVector = TileAccessIterator::kAccessesPerVector;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                               ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename TileAccessIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   public:
    using Base = typename TileAccessIterator::Params::Base;

    friend PredicatedTileIterator;

   private:
    /// Parameters object
    typename TileAccessIterator::Params params_;

   public:
    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout) : params_(layout) {}

    /// Default constructor
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(Base const &base)
        : params_(base) {}
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char *;

 private:
  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;

 public:

  /// Default constructor
  PredicatedTileIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset,
      /// Gather indices
      int const *indices = nullptr)
      : address_iterator_(params.params_, pointer, extent, thread_id,
                          threadblock_offset, indices) {}

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PredicatedTileIterator(params, pointer, extent, thread_id,
                               make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    if (kAdvanceRank)
      address_iterator_.add_tile_offset({0, 1});
    else
      address_iterator_.add_tile_offset({1, 0});

    return *this;
  }


  /********************************************************************************************************************
  ********************************************************************************************************************
  ********************************************************************************************************************
  详细解释一下下面这个函数
  函数传参的(int): 这个匿名的 int 参数是 C++ 用来区分 后缀自增 和 前缀自增 的语法标志。编译器看到这个 int，就知道这是为 it++ 这种形式准备的。而前缀自增 ++it 的函数签名则是 operator++()，没有参数。
  
  PredicatedTileIterator self(*this);这一行的this是一个指向当前对象实例的指针。*this: 解引用 this 指针，得到当前 PredicatedTileIterator 对象本身。
  所以PredicatedTileIterator self(...): 即创建一个新的 PredicatedTileIterator 对象，命名为 self。
  self(*this): 调用 拷贝构造函数 (Copy Constructor)，用当前对象 *this 的所有状态（指针、偏移量等）来初始化新对象 self。
  结果就是：self 成为了当前迭代器状态的一个完整、独立的快照。它保存了迭代器在被修改之前的样子。

  operator++();
  它没有参数，因此这是在显式地调用同一个类中定义的 前缀自增运算符 operator++()。调用的是第 341 行的函数 (PredicatedTileIterator &operator++())。
  这个调用 operator++() 没有任何参数。因此，它唯一能匹配上的就是那个没有参数的 operator++() 版本，也就是第 341 行定义的前缀自增运算符。
  所有将迭代器实际向前推进的复杂逻辑（比如更新指针、增加偏移量、重新计算掩码等）都封装在前缀 operator++() 函数中。
  通过调用它，当前对象 (*this) 的状态被更新，它现在已经指向了下一个位置。

  return self;
  这一行完成了后缀自增的第二个要求。它返回的是 self 对象，也就是我们在第一步中创建的、代表迭代器 原始状态 的那个快照。
  它返回的是一个值（a copy），而不是一个引用，这符合后缀自增的语义。

  所以整个函数的流程可以这样理解
  当你写下 auto old_iterator = iterator++; 时，编译器会执行以下操作：
  1、进入 operator++(int) 函数。
  2、备份：创建一个 iterator 的副本，我们称之为 self。
  3、前进：调用 operator++()，让 iterator 自身前进到下一个位置。
  4、返回备份：函数返回 self，这个返回值被赋给 old_iterator。
  执行完毕后，old_iterator 保存了 iterator 开始时的位置，而 iterator 本身已经移动到了下一个位置。这种设计既高效又优雅，是 C++ 标准库和高性能代码库中实现运算符的通用模式。
   ***********************************************************************************************************************
   ***********************************************************************************************************************
   ***********************************************************************************************************************
   */

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { address_iterator_.clear_mask(enable); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { address_iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { address_iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { address_iterator_.get_mask(mask); }

  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    load_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  CUTLASS_DEVICE
  // 这里是专门针对PitchLinear的layout而特化的load_with_byte_offset
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {

          int idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);
          
          address_iterator_.set_iteration_index(idx);
          char const *byte_ptr = reinterpret_cast<char const *>(address_iterator_.get()) + byte_offset;

          AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_ptr);

          cutlass::arch::global_load<AccessType,
                                     sizeof(AccessType)
                                    >(
              frag_ptr[idx], access_ptr, address_iterator_.valid());

          ++address_iterator_;
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_byte_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    store_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {

          int idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

          char *byte_ptr = reinterpret_cast<char *>(address_iterator_.get()) + byte_offset;
          AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_ptr);

          if (address_iterator_.valid()) {
            *access_ptr = frag_ptr[idx];
          }
          ++address_iterator_;
        }
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_byte_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for column-major data.
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///

// 下面这个类针对ColumnMajor特化
// 这里AdvanceRank表示迭代器在哪个维度上推进(0表示contiguous维度，1表示strided维度)
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int AccessSize,
  bool Gather,
  typename PermuteLayout
>
class PredicatedTileIterator<Shape_, Element_, layout::ColumnMajor, AdvanceRank, 
                             ThreadMap_, AccessSize, Gather, PermuteLayout> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileIterator<
    layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 0 : 1),
    ThreadMap,
    AccessSize,
    Gather,
    PermuteLayout
  >;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
  private:

    friend PredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

  public:

    /// Default constructor
    Params() = default;

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout): params_(layout::PitchLinear(layout.stride(0)))
    {}

    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const &base)
        : params_(base) {}
  };


private:

  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

public:

  /// Default constructor
  PredicatedTileIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset, and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object 
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id,                                ///< ID of each participating thread
    TensorCoord const &threadblock_offset,         ///< Initial offset of threadblock
    int const *indices = nullptr     ///< gather/scatter indices, note no support for gather/scatter at this specialization
  ):
    iterator_(
      params.params_,
      pointer,
      layout::PitchLinearCoord(extent.row(), extent.column()),
      thread_id,
      layout::PitchLinearCoord(threadblock_offset.row(), threadblock_offset.column()),
      indices)
    { }

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id                                 ///< ID of each participating thread
  ): PredicatedTileIterator(params, pointer, extent, thread_id, make_Coord(0, 0)) { }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset) {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    iterator_.store_with_byte_offset(frag, byte_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for row-major data.
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///

// 下面这个类针对RowMajor特化
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int AccessSize,
  bool Gather,
  typename PermuteLayout
>
class PredicatedTileIterator<Shape_, Element_, layout::RowMajor, AdvanceRank, 
                             ThreadMap_, AccessSize, Gather, PermuteLayout> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileIterator<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 1 : 0),
    ThreadMap,
    AccessSize,
    Gather,
    PermuteLayout
  >;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
  private:

    friend PredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

  public:

    /// Default constructor
    Params() = default;

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout): params_(layout::PitchLinear(layout.stride(0))) {}

    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const &base)
        : params_(base) {}

  };

private:

  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

public:

  /// Default constructor
  PredicatedTileIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset, and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object 
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id,                                ///< ID of each participating thread
    TensorCoord const &threadblock_offset,        ///< Initial offset of threadblock
    int const *indices = nullptr                        ///< Gather indices
  ):
    iterator_(
      params.params_,
      pointer,
      layout::PitchLinearCoord(extent.column(), extent.row()),
      thread_id,
      layout::PitchLinearCoord(threadblock_offset.column(), threadblock_offset.row()),
      indices
    ) { }

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id                                 ///< ID of each participating thread
  ): PredicatedTileIterator(params, pointer, extent, thread_id, make_Coord(0, 0)) { }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset) {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }
  
  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    iterator_.store_with_byte_offset(frag, byte_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for affine rank-2 data.
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize>
class PredicatedTileIterator<Shape_, Element_, layout::AffineRankN<2>, AdvanceRank,
                             ThreadMap_, AccessSize, false> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::AffineRankN<2>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  /// Type used for internal memory accesses
  using AccessType = AlignedArray<Element, AccessSize, (AccessSize * sizeof_bits<Element>::value / 8)>;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator =
      PredicatedTileAccessIterator<Shape, Element, Layout, kAdvanceRank,
                                   ThreadMap, AccessType>;

  static int const kAccessesPerVector = TileAccessIterator::kAccessesPerVector;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                               ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename TileAccessIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   public:

    friend PredicatedTileIterator;

   private:
    /// Parameters object
    typename TileAccessIterator::Params params_;

   public:
    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout) : params_(layout) {}

    /// Default constructor
    Params() = default;
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char *;

 private:
  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;

 public:

  /// Default constructor
  PredicatedTileIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset,
      int const *indices = nullptr     ///< gather/scatter indices, note no support for gather/scatter at this specialization
      )
      : address_iterator_(params.params_, pointer, extent, thread_id,
                          threadblock_offset) {}

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PredicatedTileIterator(params, pointer, extent, thread_id,
                               make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    if (kAdvanceRank)
      address_iterator_.add_tile_offset(make_Coord(0, 1));
    else
      address_iterator_.add_tile_offset(make_Coord(1, 0));

    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { address_iterator_.clear_mask(enable); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { address_iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { address_iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { address_iterator_.get_mask(mask); }

  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    load_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {

          int idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);
          
          address_iterator_.set_iteration_index(idx);
          char const *byte_ptr = reinterpret_cast<char const *>(address_iterator_.get()) + byte_offset;

          AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_ptr);

          cutlass::arch::global_load<AccessType,
                                     sizeof(AccessType)
                                    >(
              frag_ptr[idx], access_ptr, address_iterator_.valid());

          ++address_iterator_;
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_byte_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    store_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {

          int idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

          char *byte_ptr = reinterpret_cast<char *>(address_iterator_.get()) + byte_offset;
          AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_ptr);

          if (address_iterator_.valid()) {
            *access_ptr = frag_ptr[idx];
          }
          ++address_iterator_;
        }
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_byte_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for affine rank 2 column-major data.
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int AccessSize
>
class PredicatedTileIterator<Shape_, Element_, layout::AffineRank2ColumnMajor, AdvanceRank, ThreadMap_, AccessSize, false> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::AffineRank2ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  // Map to the underlying AffineRankN<2> layout
  using UnderlyingIterator = PredicatedTileIterator<
    layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
    Element,
    layout::AffineRankN<2>,
    (kAdvanceRank == 0 ? 0 : 1),
    ThreadMap,
    AccessSize
  >;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
  private:

    friend PredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

  public:

    /// Default constructor
    Params() = default;

    /// Construct the Params object given an AffineRankN<2> tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout): params_(layout::AffineRankN<2>(layout.stride(0), layout.stride(1)))
    {}
  };

private:

  //
  // Data members
  //

  /// Underlying AffineRankN<2> tile iterator
  UnderlyingIterator iterator_;

public:

  /// Default constructor
  PredicatedTileIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset, and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object 
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id,                                ///< ID of each participating thread
    TensorCoord const &threadblock_offset,         ///< Initial offset of threadblock
    int const *indices = nullptr     ///< gather/scatter indices, note no support for gather/scatter at this specialization
  ):
    iterator_(
      params.params_,
      pointer,
      layout::PitchLinearCoord(extent.row(), extent.column()),
      thread_id,
      layout::PitchLinearCoord(threadblock_offset.row(), threadblock_offset.column())
    ) { }

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id                                 ///< ID of each participating thread
  ): PredicatedTileIterator(params, pointer, extent, thread_id, make_Coord(0, 0)) { }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset) {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    iterator_.store_with_byte_offset(frag, byte_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for affine rank 2 row-major data.
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int AccessSize
>
class PredicatedTileIterator<Shape_, Element_, layout::AffineRank2RowMajor, AdvanceRank, ThreadMap_, AccessSize, false> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::AffineRank2RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  // Map to the underlying AffineRankN<2> layout
  using UnderlyingIterator = PredicatedTileIterator<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::AffineRankN<2>,
    (kAdvanceRank == 0 ? 1 : 0),
    ThreadMap,
    AccessSize
  >;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
  private:

    friend PredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

  public:

    /// Default constructor
    Params() = default;

    /// Construct the Params object given an AffineRankN<2> tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout): params_(layout::AffineRankN<2>(layout.stride(1), layout.stride(0))) {}
  };


private:

  //
  // Data members
  //

  /// Underlying AffineRankN<2> tile iterator
  UnderlyingIterator iterator_;

public:

  /// Default constructor
  PredicatedTileIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset, and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object 
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id,                                ///< ID of each participating thread
    TensorCoord const &threadblock_offset,         ///< Initial offset of threadblock
    int const *indices = nullptr     ///< gather/scatter indices, note no support for gather/scatter at this specialization
  ):
    iterator_(
      params.params_,
      pointer,
      layout::PitchLinearCoord(extent.column(), extent.row()),
      thread_id,
      layout::PitchLinearCoord(threadblock_offset.column(), threadblock_offset.row())
    ) { }

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id                                 ///< ID of each participating thread
  ): PredicatedTileIterator(params, pointer, extent, thread_id, make_Coord(0, 0)) { }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset) {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }
  
  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    iterator_.store_with_byte_offset(frag, byte_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for interleaved data.  It is mapped
/// to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///

template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize, int InterleavedK>
class PredicatedTileIterator<Shape_, Element_,
                             layout::ColumnMajorInterleaved<InterleavedK>,
                             AdvanceRank, ThreadMap_, AccessSize, false> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  static int const kInterleavedK = InterleavedK;
  using Layout = layout::ColumnMajorInterleaved<kInterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileIterator<
      layout::PitchLinearShape<Shape::kRow * kInterleavedK,
                               Shape::kColumn / kInterleavedK>,
      Element, layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap, AccessSize>;


  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                               ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:

    /// Default constructor
    Params() = default;

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}

    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const &base)
        : params_(base) {}

  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:

  /// Default constructor
  PredicatedTileIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset,
      int const *indices = nullptr     ///< gather/scatter indices, note no support for gather/scatter at this specialization
      )
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.row() * kInterleavedK,
                                           extent.column() / kInterleavedK),
                  thread_id,
                  layout::PitchLinearCoord(
                      threadblock_offset.row() * kInterleavedK,
                      threadblock_offset.column() / kInterleavedK)) {}

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PredicatedTileIterator(params, pointer, extent, thread_id,
                               make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

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

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for interleaved-32 data.  It is
/// mapped to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize, int InterleavedK>
class PredicatedTileIterator<Shape_, Element_,
                             layout::RowMajorInterleaved<InterleavedK>,
                             AdvanceRank, ThreadMap_, AccessSize, false> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  static int const kInterleavedK = InterleavedK;
  using Layout = layout::RowMajorInterleaved<kInterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileIterator<
      layout::PitchLinearShape<Shape::kColumn * kInterleavedK,
                               Shape::kRow / kInterleavedK>,
      Element, layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap, AccessSize>;


  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                               ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:

    /// Default constructor
    Params() = default;

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}

    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const &base)
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:

  /// Default constructor
  PredicatedTileIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset,
      int const *indices = nullptr     ///< gather/scatter indices, note no support for gather/scatter at this specialization
      )
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.column() * kInterleavedK,
                                           extent.row() / kInterleavedK),
                  thread_id,
                  layout::PitchLinearCoord(
                      threadblock_offset.column() * kInterleavedK,
                      threadblock_offset.row() / kInterleavedK)) {}

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PredicatedTileIterator(params, pointer, extent, thread_id,
                               make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

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

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace transform
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
