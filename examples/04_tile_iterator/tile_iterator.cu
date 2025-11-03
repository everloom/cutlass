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

/*
  This example demonstrates how to use the PredicatedTileIterator in CUTLASS to load data from
  addressable memory, and then store it back into addressable memory.

  TileIterator is a core concept in CUTLASS that enables efficient loading and storing of data to
  and from addressable memory. The PredicateTileIterator accepts a ThreadMap type, which defines
  the mapping of threads to a "tile" in memory. This separation of concerns enables user-defined
  thread mappings to be specified. 

  In this example, a PredicatedTileIterator is used to load elements from a tile in global memory,
  stored in column-major layout, into a fragment and then back into global memory in the same
  layout.

  This example uses CUTLASS utilities to ease the matrix operations.

  这个示例演示了如何在CUTLASS中使用PredicatedTileIterator从可寻址内存中加载数据，
  然后将其存储回可寻址内存。

  TileIterator（tile迭代器）是CUTLASS中的核心概念，它能够高效地从可寻址内存中
  加载和存储数据。PredicatedTileIterator接受一个ThreadMap类型，该类型定义了
  线程到内存中"tile"的映射关系。这种关注点分离使得用户可以指定自定义的
  线程映射方式。

  在这个示例中，PredicatedTileIterator被用来从全局内存中的一个tile加载元素
  （采用列主序布局），将其存储到fragment中，然后以相同的布局存储回全局内存。

  这个示例使用了CUTLASS工具来简化矩阵操作。

*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// CUTLASS includes
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

//
//  CUTLASS utility includes
//

// Defines operator<<() to write TensorView objects to std::ostream
#include "cutlass/util/tensor_view_io.h"

// Defines cutlass::HostTensor<>
#include "cutlass/util/host_tensor.h"

// Defines cutlass::reference::host::TensorFill() and
// cutlass::reference::host::TensorFillBlockSequential()
#include "cutlass/util/reference/host/tensor_fill.h"

#pragma warning( disable : 4503)
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define PredicatedTileIterators to load and store a M-by-K tile, in column major layout.


/*
这里说一下，64*4的threadblock tile是怎么和57*35的矩阵对应起来的
首先64*4中，64是连续的维度；57*35的矩阵中，57是连续的维度
这里相当于对57*35的矩阵分tile，每个tile是64*4的大小
M = 57 < 64：一个tile就能覆盖整个连续维；K = 35 > 4：需要 ⌈35/4⌉ = 9个tile来覆盖跨步维
考虑到，对于连续维，由于57 < 64，所以tile内有些位置超出边界，对于这种边界情况，PredicatedTileIterator 会自动处理这些越界访问
对于跨步维，35 > 4，需要 ⌈35/4⌉ = 9个tile来覆盖跨步维（对应下面的int iterations = (extent[1] + Iterator::Shape::kStrided - 1) / Iterator::Shape::kStrided;），对于最后一轮迭代，只会处理K=32-35的列（其余tile是处理4列）
然后tile和实际矩阵的可视化结果如下：

实际矩阵 (57×35):
┌─────────────────────────────────────┐
│ 每个tile 64×4                        │
│ ┌───┐ ┌───┐ ┌───┐ ... ┌───┐         │
│ │ 1 │ │ 2 │ │ 3 │ ... │ 9 │         │
│ └───┘ └───┘ └───┘     └───┘         │
│ (57×4)(57×4)(57×4)   (57×3)         │
└─────────────────────────────────────┘
*/

/*
上面是从threadblock tile整体的角度说明了tile和实际矩阵的对应关系，下面说一下具体每个线程处理矩阵中的哪的数据
由于代码中使用迭代器和ThreadMap，关于线程负责搬运哪些数据的逻辑全都被隐藏起来了。下面的回答基本来自于cursor
这里只启动了一个block，一个block中有32个线程，所以总共就一个warp在执行这个kernel
每个tile有64*4个元素，一个block有32个线程，所以每个线程负责64*4 / 32 = 8 个元素的处理（对应下面每次for循环中单个thread处理的元素数量）
根据PitchLinearStripminedThreadMap的工作方式：
  第一优先：沿连续维分配
    64个连续维位置分给32个线程
    每个线程负责：64÷32 = 2个连续位置
  第二优先：沿跨步维重复
    4个跨步维位置，每个线程都要处理
  所以每个线程总共处理：2×4 = 8个元素 （连续维度处理元素数量 x 跨步维度处理元素数量）

对于64×4的tile：
  线程ID  负责的连续维位置  负责的跨步维位置  总元素数
  0       [0,1]           [0,1,2,3]       8
  1       [2,3]           [0,1,2,3]       8  
  2       [4,5]           [0,1,2,3]       8
  ...
  31      [62,63]         [0,1,2,3]       8

每轮处理的范围：
  第1轮：处理列[0-3]的57×4子矩阵
  第2轮：处理列[4-7]的57×4子矩阵  
  第3轮：处理列[8-11]的57×4子矩阵
  ...
  第9轮：处理列[32-35]的57×4子矩阵（实际只有3列）
*/
template <typename Iterator>
__global__ void copy(
    typename Iterator::Params dst_params,
    typename Iterator::Element *dst_pointer,
    typename Iterator::Params src_params,
    typename Iterator::Element *src_pointer,
    cutlass::Coord<2> extent) {

    // 迭代器内部根据传入的threadIdx.x和ThreadMap来确定每个线程负责处理的数据位置
    Iterator dst_iterator(dst_params, dst_pointer, extent, threadIdx.x);
    Iterator src_iterator(src_params, src_pointer, extent, threadIdx.x);

    // PredicatedTileIterator uses PitchLinear layout and therefore takes in a PitchLinearShape.
    // The contiguous dimension can be accessed via Iterator::Shape::kContiguous and the strided
    // dimension can be accessed via Iterator::Shape::kStrided
    // PredicatedTileIterator使用PitchLinear布局，因此接受一个PitchLinearShape参数。
    // 连续维度可以通过Iterator::Shape::kContiguous访问，跨步维度可以通过
    // Iterator::Shape::kStrided访问
    //这里计算的是在strided维度上需要迭代多少次，即tile需要++多少次(tile需要移动多少次)
    int iterations = (extent[1] + Iterator::Shape::kStrided - 1) / Iterator::Shape::kStrided;

    // cursor说这里的fragment是每个线程私有的，在本例中，fragment的大小为8，因为下面for循环中每一次迭代就处理8个数据
    // cursor还说这里fragment的数据大小是根据ThreadMap算的
    typename Iterator::Fragment fragment;

    for(int i = 0; i < fragment.size(); ++i) {
      fragment[i] = 0;
    }

    /*
    cursor说，迭代器这里的load方法实现的功能可以用下面的伪代码来理解：
    // 伪代码展示隐藏的逻辑
    void load(Fragment& frag) {
        // 根据 threadIdx.x 和 ThreadMap 计算该线程负责的位置
        auto thread_positions = threadMap.getThreadPositions(threadIdx.x);
        
        for(int i = 0; i < frag.size(); ++i) {
            auto global_coord = current_tile_offset + thread_positions[i];
            frag[i] = src_pointer[layout(global_coord)];  // 从全局内存加载
        }
    }
    */
    src_iterator.load(fragment);
    /*
    cursor说，这里的store方法实现的功能可以用下面的伪代码来理解： 
    void store(const Fragment& frag) {
        auto thread_positions = threadMap.getThreadPositions(threadIdx.x);
        
        for(int i = 0; i < frag.size(); ++i) {
            auto global_coord = current_tile_offset + thread_positions[i];
            dst_pointer[layout(global_coord)] = frag[i];  // 存储到全局内存
        }
    }
    */
    dst_iterator.store(fragment);

    /*
    这里++方法的作用是移动到下一个tile，实现可以用下面伪代码来理解：
    Iterator& operator++() {
        current_tile_offset += tile_advance_delta;  // 移动到下一个瓦片
        return *this;
    }
    */
    ++src_iterator;
    ++dst_iterator;

    for(; iterations > 1; --iterations) {

      src_iterator.load(fragment);
      dst_iterator.store(fragment);

      ++src_iterator;
      ++dst_iterator;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Initializes the source tile with sequentially increasing values and performs the copy into
// the destination tile using two PredicatedTileIterators, one to load the data from addressable
// memory into a fragment (regiser-backed array of elements owned by each thread) and another to 
// store the data from the fragment back into the addressable memory of the destination tile.
// 使用顺序递增的值初始化源tile，并使用两个PredicatedTileIterator执行复制到
// 目标tile的操作，一个用于从可寻址内存将数据加载到fragment（每个线程拥有的
// 寄存器支持的元素数组）中，另一个用于将数据从fragment存储回目标tile的
// 可寻址内存中。
cudaError_t TestTileIterator(int M, int K) {

    // For this example, we chose a <64, 4> tile shape. The PredicateTileIterator expects
    // PitchLinearShape and PitchLinear layout.
    // 对于这个示例，我们选择了一个<64, 4>的tile形状。PredicateTileIterator期望
    // PitchLinearShape和PitchLinear布局。
    // 注意这里的64*4的tile是threadblock tile
    using Shape = cutlass::layout::PitchLinearShape<64, 4>; //64, 4
    using Layout = cutlass::layout::PitchLinear;
    using Element = int;
    int const kThreads = 32; // 32

    // ThreadMaps define how threads are mapped to a given tile. The PitchLinearStripminedThreadMap
    // stripmines a pitch-linear tile among a given number of threads, first along the contiguous
    // dimension then along the strided dimension.
    // ThreadMaps 定义了线程如何映射到给定的 Tile。PitchLinearStripminedThreadMap 会在给定数量的线程之
    // 间，首先沿着连续维度，然后沿着步幅维度，来对间距线性 Tile 进行条带化。
    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads>;

    // Define the PredicateTileIterator, using TileShape, Element, Layout, and ThreadMap types
    using Iterator = cutlass::transform::threadblock::PredicatedTileIterator<
        Shape, Element, Layout, 1, ThreadMap>;


    cutlass::Coord<2> copy_extent = cutlass::make_Coord(M, K);
    cutlass::Coord<2> alloc_extent = cutlass::make_Coord(M, K);

    // Allocate source and destination tensors
    // 问了下cursor，说是对于PitchLinear的layout，默认第[0]维是连续的，第[1]维是跨步的
    // 所以M是连续的，K是跨步的
    cutlass::HostTensor<Element, Layout> src_tensor(alloc_extent);
    cutlass::HostTensor<Element, Layout> dst_tensor(alloc_extent);

    // 定义类型别名
    using Iterator_threadmap = typename ThreadMap::Iterations;
    using Delta_threadmap = typename ThreadMap::Delta;
    using StorageShape_threadmap = typename ThreadMap::StorageShape;
    using ShapeVec_threadmap = typename ThreadMap::Detail::ShapeVec;
    // 调试：打印这些类型的静态成员值
    printf("=== ThreadMap Debug Info ===\n");
    printf("Iterations::kContiguous = %d\n", Iterator_threadmap::kContiguous);
    printf("Iterations::kStrided = %d\n", Iterator_threadmap::kStrided);
    
    printf("Delta::kContiguous = %d\n", Delta_threadmap::kContiguous);
    printf("Delta::kStrided = %d\n", Delta_threadmap::kStrided);
    
    printf("StorageShape::kContiguous = %d\n", StorageShape_threadmap::kContiguous);
    printf("StorageShape::kStrided = %d\n", StorageShape_threadmap::kStrided);

    printf("ShapeVec::kContiguous = %d\n", ShapeVec_threadmap::kContiguous);
    printf("ShapeVec::kStrided = %d\n", ShapeVec_threadmap::kStrided);
    printf("============================\n");


    Element oob_value = Element(-1);

    // Initialize destination tensor with all -1s
    cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
    // Initialize source tensor with sequentially increasing values
    cutlass::reference::host::BlockFillSequential(src_tensor.host_data(), src_tensor.capacity());

    dst_tensor.sync_device();
    src_tensor.sync_device();

    typename Iterator::Params dst_params(dst_tensor.layout());
    typename Iterator::Params src_params(src_tensor.layout());

    dim3 block(kThreads, 1);
    dim3 grid(1, 1);

    // Launch copy kernel to perform the copy
    copy<Iterator><<< grid, block >>>(
            dst_params,
            dst_tensor.device_data(),
            src_params,
            src_tensor.device_data(),
            copy_extent
    );

    cudaError_t result = cudaGetLastError();
    if(result != cudaSuccess) {
      std::cerr << "Error - kernel failed." << std::endl;
      return result;
    }

    dst_tensor.sync_host();

    // Verify results
    for(int s = 0; s < alloc_extent[1]; ++s) {
      for(int c = 0; c < alloc_extent[0]; ++c) {

          Element expected = Element(0);

          if(c < copy_extent[0] && s < copy_extent[1]) {
            expected = src_tensor.at({c, s});
          }
          else {
            expected = oob_value;
          }

          Element got = dst_tensor.at({c, s});
          bool equal = (expected == got);

          if(!equal) {
              std::cerr << "Error - source tile differs from destination tile." << std::endl;
            return cudaErrorUnknown;
          }
      }
    }

    return cudaSuccess;
}

int main(int argc, const char *arg[]) {

    cudaError_t result = TestTileIterator(57,35);

    if(result == cudaSuccess) {
      std::cout << "Passed." << std::endl;  
    }

    // Exit
    return result == cudaSuccess ? 0 : -1;
}

