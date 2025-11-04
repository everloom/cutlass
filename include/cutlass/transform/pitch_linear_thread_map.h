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
    \brief Templates implementing how threads are mapped to a given tile.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {

////////////////////////////////////////////////////////////////////////////////

/// Strip-mines a pitch-linear tile among a given number of threads, first along
/// the contiguous dimension then along the strided dimension.
///
/// The tile must be divisible by the thread count such that all threads may
/// execute the same number of iterations with the same delta to exhaustively
/// cover the tile.
///
/// This class satisfies the "RegularThreadMapping" concept.
///
/// This ThreadMap is used by SIMT kernels and operand E of the sparse tensor
/// kernels.
// 将一个pitch-linear瓦片在给定数量的线程中进行条带挖掘，首先沿着连续维度，然后沿着跨步维度。
// 瓦片必须能被线程数整除，这样所有线程都可以执行相同数量的迭代，使用相同的增量来详尽地覆盖整个瓦片。
// 这个类满足"RegularThreadMapping"概念。
// 这个ThreadMap被SIMT内核和稀疏张量内核的操作数E使用。
template <
  typename Shape_,
  int Threads,
  int ElementsPerAccess = 1
>
struct PitchLinearStripminedThreadMap {
  
  /// Tensor coordinate
  using TensorCoord = layout::PitchLinearCoord;

  /// Tile shape
  // 应该是threadblock tile的shape
  using Shape = Shape_;

  /// Number of threads total
  // 这里是一个block有多少线程
  static int const kThreads = Threads;

  /// Extract vector length from Layout
  /*
  kElementsPerAccess定义了每个线程的每次访存的个数，也即定义了每个线程的向量化访存的数据个数
  这里以default_mma_core_sm80.h为例讲一下, 这个代码中会对sm80下的PitchLinearStripminedThreadMap做初始化
  然后文件中有很多DefaultMmaCore的特化，其中基本DeafultMmaCore中在使用PitchLinearStripminedThreadMap模版时，这里的
  ElementsPerAccess传入的是kAccessSizeInBits / sizeof_bits<ElementA>::value>，其中kAccessSizeInBits在default_mma_core_sm80.h只有
  64和128的情况，cursor说之所以只能是这两个值是因为刚好对应了 8 / 16个byte，对合并访存更友好
  所以这里每个线程向量化访存的大小不是64bit就是128bit
  */
  static int const kElementsPerAccess = ElementsPerAccess;

  /// Shape of access by each thread
  using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

  /// Internal implementation details
  struct Detail {

    static_assert(!(Shape::kContiguous % kElementsPerAccess), "");

    /// Shape of the tile in units of vectors
    // 因为每个thread向量化访存kElementsPerAccess个数(64bit或者128bit)
    // 所以这里需要对threadblock tile做一下调整，调整为向量化访存的tile shape
    // 例如原始瓦片：Shape<64, 4>（64×4个元素）
    // 假设kElementsPerAccess = 2（一个访问两个数，数的type是int还是float根据你自己定义）
    // 向量化后的tile shape：ShapeVec<32, 4>（32×4个向量，每个向量2个元素）
    using ShapeVec = layout::PitchLinearShape<
      Shape::kContiguous / kElementsPerAccess,
      Shape::kStrided
    >;
    /*
    这里的断言做了这样的事情：
    1、当block的线程数小于向量化后的tile shape的连续维度大小时，需要满足向量化的tile shape的连续维度大小能整除block的thread数
    2、当block的线程数大于向量化后的tile shape的连续维度大小时，需要满足block的thread数量能整除向量化的tile shape的连续维度大小
    */
    static_assert((Threads < ShapeVec::kContiguous && !(ShapeVec::kContiguous % kThreads)) ||
                      (!(kThreads % ShapeVec::kContiguous)),
                  "Shape must be divisible by number of iterations of each thread.");
  };

  /*
  下面这个表达式比较复杂，这里解释一下（来自cursor）

  整体结构：
      using Iterations = typename platform::conditional<
        条件,
        类型A,
        类型B
      >::type;
    这是一个编译期类型选择，相当于：
      if (条件) {
        Iterations = 类型A;
      } else {
          Iterations = 类型B;
      }

  具体分解：
    1、条件判断，判断线程数是否足够覆盖连续维度
      Threads >= Detail::ShapeVec::kContiguous
    2、类型A（线程数充足时）
      layout::PitchLinearShape<
        1,  // 连续维只需1次迭代
        复杂表达式  // 跨步维需要的迭代次数
      >
    3、复杂表达式详解：
        (Detail::ShapeVec::kStrided + (kThreads / Detail::ShapeVec::kContiguous - 1)) /
        (kThreads / Detail::ShapeVec::kContiguous)
      这是在计算向上取整除法：ceil(kStrided / (kThreads/kContiguous))
      kThreads / kContiguous：每个跨步维"行"分配多少线程
      kStrided / (kThreads/kContiguous)：跨步维需要多少轮迭代
      + (除数 - 1) 然后 / 除数：这是整数向上取整的经典技巧
      而(Threads >= Detail::ShapeVec::kContiguous ? 表达式 : 0)，注释解释了：某些编译器会同时实例化conditional的两个分支，如果直接写除法可能在"不该执行的分支"上遇到除零错误，所以用三元运算符再次保护。
    4、类型B（线程数不足时）
      layout::PitchLinearShape<
          Detail::ShapeVec::kContiguous / kThreads,  // 连续维需要多轮
          Detail::ShapeVec::kStrided                 // 跨步维保持原样
      >
  */
  /// Number of iterations by each thread
  // iterations中包含连续维的结果和跨步维的结果，假设连续维为1，跨步维为32
  // 则表示一个线程，在连续维迭代一次，在跨步维迭代32次
  // 如果看不懂就看cutlass.md，上面这块有讲
  using Iterations = typename platform::conditional<
      Threads >= Detail::ShapeVec::kContiguous,
      layout::PitchLinearShape<
          1,
          // Redo the comparison here to work around divide by zero compiler
          // error.  The compiler evaluates both path of platform::conditional.
          // 在这里重新进行比较以解决编译器除零错误的问题。
          // 编译器会评估 platform::conditional 的两个路径。
          (Threads >= Detail::ShapeVec::kContiguous
               ? (Detail::ShapeVec::kStrided + (kThreads / Detail::ShapeVec::kContiguous - 1)) /
                     (kThreads / Detail::ShapeVec::kContiguous)
               : 0)>,
      layout::PitchLinearShape<Detail::ShapeVec::kContiguous / kThreads,
                               Detail::ShapeVec::kStrided>>::type;
  

  /// Interval between accesses along each dimension of the tensor's logical coordinate space
  /// (in units of Elements)
  /// 沿着张量逻辑坐标空间每个维度的访问间隔（以元素为单位）
  using Delta = typename platform::conditional<
    Threads >= Detail::ShapeVec::kContiguous,
    layout::PitchLinearShape<
      1,
      kThreads / Detail::ShapeVec::kContiguous
    >,
    layout::PitchLinearShape<
      kThreads * kElementsPerAccess,
      1
    >
  >::type;

  /// Shape of the tile in units of vectors
  /// 以向量为单位的tile形状
  using StorageShape = typename platform::conditional<
      Threads >= Detail::ShapeVec::kContiguous,
      layout::PitchLinearShape<Shape::kContiguous,
                               Iterations::kStrided*(kThreads / Detail::ShapeVec::kContiguous)>,
      layout::PitchLinearShape<Shape::kContiguous, Shape::kStrided>>::type;

  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  /// (in units of Elements)
  /// 将线程ID映射到张量逻辑坐标空间中的坐标偏移（以元素为单位）
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {
    return TensorCoord(
      (thread_id % Detail::ShapeVec::kContiguous) * kElementsPerAccess, 
      thread_id / Detail::ShapeVec::kContiguous);
  }
};

/// This ThreadMap is used by GEMV
template <
  typename Shape,
  int Threads,
  int ElementsPerAccess = 1
>
struct PitchLinearTilePolicyStripminedThreadContiguous
{
 static_assert((Shape::kContiguous % (Threads * ElementsPerAccess)) == 0,
              "Contiguous shape must divide number of threads");

  using TensorCoord = layout::PitchLinearCoord;

  static int const kThreads = Threads;
  static int const kElementsPerAccess = ElementsPerAccess;

  using Iterations = layout::PitchLinearShape<
                      Shape::kContiguous / (kThreads * kElementsPerAccess),
                      Shape::kStrided>;

  using Delta = layout::PitchLinearShape<1, 1>;

  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id)
  {
    return TensorCoord(thread_id * Iterations::kContiguous * kElementsPerAccess, 0);
  }
};

template <
  typename Shape,
  int Threads,
  int ElementsPerAccess = 1
>
struct PitchLinearTilePolicyStripminedThreadStrided
{
  static_assert((Shape::kStrided % Threads == 0),
                "Strided shape must divide number of threads");

  using TensorCoord = layout::PitchLinearCoord;

  static int const kThreads = Threads;
  static int const kElementsPerAccess = ElementsPerAccess;

  using Iterations = layout::PitchLinearShape<
                      Shape::kContiguous / kElementsPerAccess,
                      Shape::kStrided / kThreads>;

  using Delta = layout::PitchLinearShape<1, 1>;

  using ShapeVec = Shape;

  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id)
  {

    return TensorCoord(0, thread_id * Iterations::kStrided);
  }
};


////////////////////////////////////////////////////////////////////////////////

/// Policy defining a warp-raked arrangement in which a shape is partitioned into contiguous
/// elements.
///
/// This ThreadMap is used by tensor core kernels.
template <
  typename Shape_,
  int Threads,
  typename WarpThreadArrangement_,
  int ElementsPerAccess = 1
>
struct PitchLinearWarpRakedThreadMap {

  /// Tensor coordinate
  using TensorCoord = layout::PitchLinearCoord;

  /// Tile shape
  using Shape = Shape_;

  /// Number of threads total
  static int const kThreads = Threads;

  /// Extract vector length from Layout
  static int const kElementsPerAccess = ElementsPerAccess;

  /// Shape of access by each thread
  using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

  /// Internal details made public to facilitate introspection
  struct Detail {

    /// Fixed arrangement of threads within a warp (units of threads).
    using WarpThreadArrangement = WarpThreadArrangement_;

    /// Number of threads per warp
    static int const kWarpSize = WarpThreadArrangement::kCount;

    /// Number of participating warps
    static int const kWarpCount = kThreads / kWarpSize;

    static_assert(
      !(Shape::kContiguous % kElementsPerAccess),
      "Shape must be divisible by vector length.");

    /// Compute the 'shape' of the overall tile in units of vectors
    using ShapeInAccesses = layout::PitchLinearShape<
      Shape::kContiguous / kElementsPerAccess,
      Shape::kStrided
    >;

    static_assert(
      !(ShapeInAccesses::kContiguous % WarpThreadArrangement::kContiguous),
      "ShapeInAccesses must be divisible by WarpThreadArrangement.");

    static_assert(
      !(ShapeInAccesses::kStrided % WarpThreadArrangement::kStrided),
      "ShapeInAccesses must be divisible by WarpThreadArrangement.");

    // compute number of warp-level accesses total
    using WarpAccessIterations = layout::PitchLinearShape<
      ShapeInAccesses::kContiguous / WarpThreadArrangement::kContiguous,
      ShapeInAccesses::kStrided / WarpThreadArrangement::kStrided
    >;

    // Divide it into the number of warps, first partitioning the strided dimension then the
    // contiguous.
    static int const kWarpsStrided =
        (WarpAccessIterations::kStrided >= kWarpCount
             ? kWarpCount
             : WarpAccessIterations::kStrided);

    static int const kWarpsContiguous =
        (kWarpCount > WarpAccessIterations::kStrided
             ? kWarpCount / kWarpsStrided
             : 1);

    /// Arrangement of warps within a threadblock-scoped tile
    using WarpArrangement = layout::PitchLinearShape<
      kWarpsContiguous, kWarpsStrided
    >;
  };

  ///< Iterations along each dimension (concept: PitchLinearShape)
  using Iterations = layout::PitchLinearShape<
    Detail::WarpAccessIterations::kContiguous / Detail::kWarpsContiguous,
    Detail::WarpAccessIterations::kStrided / Detail::kWarpsStrided
  >;

  static_assert(Iterations::kCount,
    "Number of iterations must be non-zero");

  ///< Delta betweeen accesses (units of elements, concept: PitchLinearShape)
  using Delta = layout::PitchLinearShape<
    Detail::WarpThreadArrangement::kContiguous * kElementsPerAccess,
    Detail::WarpThreadArrangement::kStrided
  >;

  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {

    int warp_id = (thread_id / Detail::kWarpSize);
    int lane_id = (thread_id % Detail::kWarpSize);

    //
    // compute warp-level offset
    //

    // This is the shape of the entire area covered by a warp's memory access (in units of vectors)
    layout::PitchLinearCoord warp_footprint{
      Detail::WarpThreadArrangement::kContiguous * Iterations::kContiguous,
      Detail::WarpThreadArrangement::kStrided * Iterations::kStrided
    };

    // This is the offset of a specific warp (in units of vectors)
    layout::PitchLinearCoord warp_offset{
      (warp_id % Detail::kWarpsContiguous),
      (warp_id / Detail::kWarpsContiguous)
    };

    // This is the offset of a specific thread within a warp (units of vectors)
    layout::PitchLinearCoord thread_offset_in_warp{
      lane_id % Detail::WarpThreadArrangement::kContiguous,
      lane_id / Detail::WarpThreadArrangement::kContiguous
    };

    // This is the offset of a thread within a threadblock tile (units of vectors)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
      warp_footprint * warp_offset + thread_offset_in_warp;

    // This is the offset of a thread within a threadblock tile (units of elements)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
      thread_offset_in_threadblock_tile_vec.contiguous() * kElementsPerAccess,
      thread_offset_in_threadblock_tile_vec.strided()
    };

    return thread_offset_in_threadblock_tile_base;
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Policy defining a warp-raked arrangement in which a shape is partitioned into contiguous
/// elements. Warps are arranged based on a stride.
///
/// This ThreadMap is used by tensor core kernels for NCxHWx layout.
template <
  typename Shape_,
  int Threads,
  typename WarpThreadArrangement_,
  int ElementsPerAccess = 1
>
struct PitchLinearStridedWarpRakedThreadMap {

  /// Tensor coordinate
  using TensorCoord = layout::PitchLinearCoord;

  /// Tile shape
  using Shape = Shape_;

  /// Number of threads total
  static int const kThreads = Threads;

  using WarpThreadArrangement = WarpThreadArrangement_;

  /// Extract vector length from Layout
  static int const kElementsPerAccess = ElementsPerAccess;

  /// Base ThreadMap
  using BaseThreadMap = PitchLinearWarpRakedThreadMap<
    Shape,
    kThreads,
    WarpThreadArrangement,
    kElementsPerAccess
  >;

  /// Shape of access by each thread
  using ThreadAccessShape = typename BaseThreadMap::ThreadAccessShape;


  struct Detail {

    using WarpThreadArrangement = WarpThreadArrangement_;

    using WarpAccessIterations = typename BaseThreadMap::Detail::WarpAccessIterations;

    static int const kWarpSize = BaseThreadMap::Detail::kWarpSize;

    static int const kWarpCount = BaseThreadMap::Detail::kWarpCount;

    using ShapeInAccesses = typename BaseThreadMap::Detail::ShapeInAccesses;

    // Divide it into the number of warps, first partitioning the contiguous dimension then the
    // stride.
    static int const kWarpsContiguous =
        (WarpAccessIterations::kContiguous >= kWarpCount
             ? kWarpCount
             : WarpAccessIterations::kContiguous);

    static int const kWarpsStrided =
        (kWarpCount > WarpAccessIterations::kContiguous
             ? kWarpCount / kWarpsContiguous
             : 1);

    /// Arrangement of warps within a threadblock-scoped tile
    using WarpArrangement = layout::PitchLinearShape<
      kWarpsContiguous, kWarpsStrided
    >;

  };

  ///< Iterations along each dimension (concept: PitchLinearShape)
  using Iterations = layout::PitchLinearShape<
    Detail::WarpAccessIterations::kContiguous / Detail::kWarpsContiguous,
    Detail::WarpAccessIterations::kStrided / Detail::kWarpsStrided
  >;

  static_assert(Iterations::kCount,
    "Number of iterations must be non-zero");

  ///< Delta betweeen accesses (units of elements, concept: PitchLinearShape)
  using Delta = typename BaseThreadMap::Delta;

  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {

    int warp_id = (thread_id / Detail::kWarpSize);
    int lane_id = (thread_id % Detail::kWarpSize);

    //
    // compute warp-level offset
    //

    // This is the shape of the entire area covered by a warp's memory access (in units of vectors)
    layout::PitchLinearCoord warp_footprint{
      Detail::WarpThreadArrangement::kContiguous * Iterations::kContiguous,
      Detail::WarpThreadArrangement::kStrided * Iterations::kStrided
    };

    // This is the offset of a specific warp (in units of vectors)
    layout::PitchLinearCoord warp_offset{
      (warp_id % Detail::kWarpsContiguous),
      (warp_id / Detail::kWarpsContiguous)
    };

    // This is the offset of a specific thread within a warp (units of vectors)
    layout::PitchLinearCoord thread_offset_in_warp{
      lane_id % Detail::WarpThreadArrangement::kContiguous,
      lane_id / Detail::WarpThreadArrangement::kContiguous
    };

    // This is the offset of a thread within a threadblock tile (units of vectors)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
      warp_footprint * warp_offset + thread_offset_in_warp;

    // This is the offset of a thread within a threadblock tile (units of elements)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
      thread_offset_in_threadblock_tile_vec.contiguous() * kElementsPerAccess,
      thread_offset_in_threadblock_tile_vec.strided()
    };

    return thread_offset_in_threadblock_tile_base;
  }


};

////////////////////////////////////////////////////////////////////////////////

/// Transpose the existing ThreadMap.  For example, interleaved layout is like
/// congruous in the global memory and crosswise in the shared memory.  We need
/// to transpose the coordinates between two.

template <typename ThreadMap_, typename WarpThreadArrangement_>
struct TransposePitchLinearThreadMap {
  /// Underlying ThreadMap
  using ThreadMap = ThreadMap_;

  /// Tensor coordinate
  using TensorCoord = typename ThreadMap::TensorCoord;

  /// Tile shape
  using Shape = typename ThreadMap::Shape;

  /// Number of threads total
  static int const kThreads = ThreadMap::kThreads;

  /// Extract vector length from Layout
  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

  /// Shape of access by each thread
  using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// Fixed arrangement of threads within a warp (units of threads).
    using WarpThreadArrangement = WarpThreadArrangement_;

    /// Number of threads per warp
    static int const kWarpSize = WarpThreadArrangement::kCount;

    /// Number of participating warps
    static int const kWarpCount = kThreads / kWarpSize;

    static_assert(!(Shape::kContiguous % kElementsPerAccess),
                  "Shape must be divisible by vector length.");

    /// Arrangement of warps within a threadblock-scoped tile
    using WarpArrangement =
        layout::PitchLinearShape<ThreadMap::Detail::kWarpsStrided,
                                 ThreadMap::Detail::kWarpsContiguous>;
  };

  ///< Iterations along each dimension (concept: PitchLinearShape)
  using Iterations =
      layout::PitchLinearShape<ThreadMap::Iterations::kStrided,
                               ThreadMap::Iterations::kContiguous>;

  static_assert(Iterations::kContiguous == 1,
    "Contiguous iteration has to be one to reuse the same shared store function with those that don't need transpose");

  static_assert(Iterations::kCount, "Number of iterations must be non-zero");

  ///< Delta betweeen accesses (units of elements, concept: PitchLinearShape)
  using Delta =
      layout::PitchLinearShape<Detail::WarpThreadArrangement::kContiguous *
                                   kElementsPerAccess,
                               Detail::WarpThreadArrangement::kStrided>;

  /// Maps thread ID to a coordinate offset within the tensor's logical
  /// coordinate space Note this is slightly different from the one of
  /// PitchLinearWarpRakedThreadMap.
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {

    int warp_id = (thread_id / Detail::kWarpSize);
    int lane_id = (thread_id % Detail::kWarpSize);

    //
    // compute warp-level offset
    //

    // This is the shape of the entire area covered by a warp's memory access
    // (in units of vectors)
    layout::PitchLinearCoord warp_footprint{
        Detail::WarpThreadArrangement::kContiguous * Iterations::kContiguous,
        Detail::WarpThreadArrangement::kStrided * Iterations::kStrided};

    // This is the offset of a specific warp (in units of vectors)
    // Note the order of / and %. Also the 2nd operand is kStrided.
    layout::PitchLinearCoord warp_offset{
        (warp_id / Detail::WarpArrangement::kStrided),
        (warp_id % Detail::WarpArrangement::kStrided)};

    // This is the offset of a specific thread within a warp (units of vectors)
    layout::PitchLinearCoord thread_offset_in_warp{
        lane_id % Detail::WarpThreadArrangement::kContiguous,
        lane_id / Detail::WarpThreadArrangement::kContiguous};

    // This is the offset of a thread within a threadblock tile (units of
    // vectors)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
        warp_footprint * warp_offset + thread_offset_in_warp;

    // This is the offset of a thread within a threadblock tile (units of
    // elements)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
        thread_offset_in_threadblock_tile_vec.contiguous() * kElementsPerAccess,
        thread_offset_in_threadblock_tile_vec.strided()};

    return thread_offset_in_threadblock_tile_base;
  }
};

template <typename ThreadMap_>
struct TransposePitchLinearThreadMapSimt {
    /// Underlying ThreadMap
    using ThreadMap = ThreadMap_;

    /// Tensor coordinate
    using TensorCoord = typename ThreadMap::TensorCoord;

    /// Tile shape
    using Shape = typename ThreadMap::Shape;

    /// Number of threads total
    static int const kThreads = ThreadMap::kThreads;

    /// Extract vector length from Layout
    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

    static_assert(kElementsPerAccess == 1 , "Simt transpose requires elements per access to be 1");
    ///< Iterations along each dimension (concept: PitchLinearShape)
    using Iterations =
        layout::PitchLinearShape<ThreadMap::Iterations::kStrided,
        ThreadMap::Iterations::kContiguous>;

    static_assert(Iterations::kCount, "Number of iterations must be non-zero");

    static_assert(Iterations::kStrided == 1,
      "Strided iteration has to be one to reuse the same shared store function with those that don't need transpose");

    /// Shape of access by each thread
    using ThreadAccessShape = typename ThreadMap::ThreadAccessShape;

    ///< Delta betweeen accesses (units of elements, concept: PitchLinearShape)
    using Delta =
        layout::PitchLinearShape<ThreadMap::Delta::kStrided,
        ThreadMap::Delta::kContiguous>;


    /// Maps thread ID to a coordinate offset within the tensor's logical
    /// coordinate space Note this is slightly different from the one of
    /// PitchLinearWarpRakedThreadMap.
    CUTLASS_HOST_DEVICE
        static TensorCoord initial_offset(int thread_id) {

        TensorCoord coord = ThreadMap::initial_offset(thread_id);

        return TensorCoord(
            coord.strided(),
            coord.contiguous()
        );
    }
};

////////////////////////////////////////////////////////////////////////////////


/// Policy defining a warp-striped arrangement.  This partitions a tile into vectorized memory
/// accesses performed by each warp then distributes warps across them. Warps are striped in the
/// strided dimension and raked across the contiguous dimension.
template <
  typename Shape_,                          /// Overall shape to partition in units of elements
  int Threads,                              /// Number of partiticipation threads
  typename WarpThreadArrangement_,          /// Describes the shape of one memory access per warp
  int ElementsPerAccess = 1                 /// Number of elements accessed by each thread per memory operation (i.e. vector size)
>
struct PitchLinearWarpStripedThreadMap {

  /// Tensor coordinate
  using TensorCoord = layout::PitchLinearCoord;

  /// Tile shape
  using Shape = Shape_;

  /// Number of threads total
  static int const kThreads = Threads;

  /// Extract vector length from Layout
  static int const kElementsPerAccess = ElementsPerAccess;

  /// Shape of access by each thread
  using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

  /// Internal details made public to facilitate introspection
  struct Detail {

    /// Fixed arrangement of threads within a warp (units of threads).
    using WarpThreadArrangement = WarpThreadArrangement_;

    /// Number of threads per warp
    static int const kWarpSize = WarpThreadArrangement::kCount;

    /// Number of participating warps
    static int const kWarpCount = kThreads / kWarpSize;

    static_assert(
      !(Shape::kContiguous % kElementsPerAccess),
      "Shape must be divisible by vector length.");

    /// Compute the 'shape' of the overall tile in units of vectors
    using ShapeInAccesses = layout::PitchLinearShape<
      Shape::kContiguous / kElementsPerAccess,
      Shape::kStrided
    >;

    // compute number of warp-level accesses total
    using WarpAccessIterations = layout::PitchLinearShape<
      ShapeInAccesses::kContiguous / WarpThreadArrangement::kContiguous,
      ShapeInAccesses::kStrided / WarpThreadArrangement::kStrided
    >;

    // Divide it into the number of warps, first partitioning the strided dimension then the
    // contiguous.
    static int const kWarpsStrided =
      (WarpAccessIterations::kStrided >= kWarpCount
        ? kWarpCount : (kWarpCount / WarpAccessIterations::kStrided));

    static int const kWarpsContiguous =
      (kWarpCount > WarpAccessIterations::kStrided ?
        WarpAccessIterations::kContiguous / kWarpsStrided : 1);

    /// Arrangement of warps within a threadblock-scoped tile
    using WarpArrangement = layout::PitchLinearShape<
      kWarpsContiguous, kWarpsStrided
    >;
  };

  ///< Iterations along each dimension (concept: PitchLinearShape)
  using Iterations = layout::PitchLinearShape<
    Detail::WarpAccessIterations::kContiguous / Detail::kWarpsContiguous,
    Detail::WarpAccessIterations::kStrided / Detail::kWarpsStrided
  >;

  static_assert(Iterations::kCount,
    "Number of iterations must be non-zero");

  ///< Delta betweeen accesses (units of elements, concept: PitchLinearShape)
  using Delta = layout::PitchLinearShape<
    Detail::WarpThreadArrangement::kContiguous * kElementsPerAccess,
    Detail::WarpThreadArrangement::kStrided * Detail::WarpArrangement::kStrided
  >;

  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {

    int warp_id = (thread_id / Detail::kWarpSize);
    int lane_id = (thread_id % Detail::kWarpSize);

    //
    // compute warp-level offset
    //

    // This is the shape of the entire area covered by a warp's memory access (in units of vectors)
    layout::PitchLinearCoord warp_footprint{
      Detail::WarpThreadArrangement::kContiguous * Iterations::kContiguous,
      Detail::WarpThreadArrangement::kStrided
    };

    // This is the offset of a specific warp (in units of vectors)
    layout::PitchLinearCoord warp_offset{
      (warp_id % Detail::kWarpsContiguous),
      (warp_id / Detail::kWarpsContiguous)
    };

    // This is the offset of a specific thread within a warp (units of vectors)
    layout::PitchLinearCoord thread_offset_in_warp{
      lane_id % Detail::WarpThreadArrangement::kContiguous,
      lane_id / Detail::WarpThreadArrangement::kContiguous
    };

    // This is the offset of a thread within a threadblock tile (units of vectors)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
      warp_footprint * warp_offset + thread_offset_in_warp;

    // This is the offset of a thread within a threadblock tile (units of elements)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
      thread_offset_in_threadblock_tile_vec.contiguous() * kElementsPerAccess,
      thread_offset_in_threadblock_tile_vec.strided()
    };

    return thread_offset_in_threadblock_tile_base;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Strip-mines a pitch-linear tile among a given number of threads, first along the contiguous
/// dimension then along the strided dimension, while each thread access a 2D thread-tile.
///
/// The tile must be divisible by the thread count such that all threads may execute the same
/// number of iterations with the same delta to exhaustively cover the tile.
///
/// This class satisfies the "RegularThreadMapping" concept.
template <
  typename Shape_,
  int Threads,
        typename ThreadTileShape
>
struct PitchLinear2DThreadTileStripminedThreadMap;


template <
  typename Shape_,
  int Threads
>
struct PitchLinear2DThreadTileStripminedThreadMap <Shape_, Threads, cutlass::layout::PitchLinearShape<4, 4>>{

  /// Tensor coordinate
  using TensorCoord = layout::PitchLinearCoord;

  /// Tile shape
  using Shape = Shape_;

  /// Access Shape of each thread
  using ThreadAccessShape = cutlass::layout::PitchLinearShape<4, 4>;
  //using ThreadAccessShape = ThreadTileShape;

  /// Number of threads total
  static int const kThreads = Threads;

  /// Extract length of each access from Layout
  static int const kElementsPerAccess = ThreadAccessShape::kContiguous;

  static_assert(!(kElementsPerAccess % 4) , "kElementsPerAccess, needs to be multiple of 4 (32bits)");

  /// Internal implementation details
  struct Detail {

    static_assert(!(ThreadAccessShape::kContiguous % 4), "ThreadAccessShape, needs to be multiple of 4");

    static_assert(!(Shape::kContiguous % ThreadAccessShape::kContiguous), "");

    static_assert(!((Shape::kContiguous * Shape::kStrided) % (kThreads * ThreadAccessShape::kCount)),
      "Shape must be divisible thread count * accesses per thread.");

    /// Shape of the tile in units of vectors
    using ShapeVec = layout::PitchLinearShape<
      Shape::kContiguous / ThreadAccessShape::kContiguous,
      Shape::kStrided / ThreadAccessShape::kStrided
    >;

    static_assert(
      (Threads < ShapeVec::kContiguous && !(ShapeVec::kContiguous % kThreads)) ||
      (!(kThreads % ShapeVec::kContiguous) && !(ShapeVec::kStrided % (kThreads / ShapeVec::kContiguous))),
      "Shape must be divisible by number of iterations of each thread."
    );
  };

  /// Number of iterations by each thread
  using Iterations = typename platform::conditional<
      Threads >= Detail::ShapeVec::kContiguous,
      layout::PitchLinearShape<
          1,
          // Redo the comparison here to work around divide by zero compiler
          // error.  The compiler evaluates both path of platform::conditional.
          (Threads >= Detail::ShapeVec::kContiguous
               ? Detail::ShapeVec::kStrided /
                     (kThreads / Detail::ShapeVec::kContiguous)
               : 0)>,
      layout::PitchLinearShape<Detail::ShapeVec::kContiguous / kThreads,
                               Detail::ShapeVec::kStrided>>::type;

  /// Interval between accesses along each dimension of the tensor's logical coordinate space
  /// (in units of Elements)
  using Delta = typename platform::conditional<
    Threads >= Detail::ShapeVec::kContiguous,
    layout::PitchLinearShape<
      Shape::kContiguous,
      kThreads * ThreadAccessShape::kStrided / Detail::ShapeVec::kContiguous
    >,
    layout::PitchLinearShape<
      kThreads * ThreadAccessShape::kContiguous,
      1
    >
  >::type;

  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  /// (in units of Elements)
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {

    return TensorCoord(
      (thread_id % Detail::ShapeVec::kContiguous) * ThreadAccessShape::kContiguous,
      (thread_id / Detail::ShapeVec::kContiguous) * ThreadAccessShape::kStrided);
  }
};

/// Thread Mapping a 2D threadtiled mapping as a transposed Pitchlinear2DThreadTile mapping
template <typename ThreadMap_>
struct TransposePitchLinearThreadMap2DThreadTile {
    /// Underlying ThreadMap
    using ThreadMap = ThreadMap_;

    /// Tensor coordinate
    using TensorCoord = typename ThreadMap::TensorCoord;

    /// Tile shape
    using Shape = typename ThreadMap::Shape;

    /// Number of threads total
    static int const kThreads = ThreadMap::kThreads;

    /// Extract vector length from Layout
    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;


    static_assert(kElementsPerAccess > 1 , "Simt transpose requires elements per access to be 1");
    ///< Iterations along each dimension (concept: PitchLinearShape)
    using Iterations =
        layout::PitchLinearShape<ThreadMap::Iterations::kStrided,
        ThreadMap::Iterations::kContiguous>;

    static_assert(Iterations::kCount, "Number of iterations must be non-zero");

    /// Shape of access by each thread
    using ThreadAccessShape = typename ThreadMap::ThreadAccessShape;

    ///< Delta betweeen accesses (units of elements, concept: PitchLinearShape)
    using Delta =
        layout::PitchLinearShape<ThreadMap::Delta::kStrided,
        ThreadMap::Delta::kContiguous>;


    /// Maps thread ID to a coordinate offset within the tensor's logical
    /// coordinate space Note this is slightly different from the one of
    /// PitchLinearWarpRakedThreadMap.
    CUTLASS_HOST_DEVICE
        static TensorCoord initial_offset(int thread_id) {

        TensorCoord coord = ThreadMap::initial_offset(thread_id);
        return TensorCoord(
            coord.strided(),
            coord.contiguous()
        );
    }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace transform
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
