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
    \brief Definitions for GEMM structures
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/arch/wmma.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

template <
  typename OperatorClass,
  typename ArchTag,
  typename ElementA, 
  typename ElementB, 
  typename ElementC,
  typename ElementAccumulator
>
struct DefaultGemmConfiguration;

////////////////////////////////////////////////////////////////////////////////

template <
  typename ArchTag,
  typename ElementA, 
  typename ElementB, 
  typename ElementC, 
  typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassSimt, 
  ArchTag,
  ElementA, 
  ElementB, 
  ElementC, 
  ElementAccumulator> {

  // 下面有很多模版的 kAlignment = 128 / sizeof_bits<Element_type>::value
  // kAlignment定义了访存的内存对齐要求，即矩阵的起始地址需要向多大的数对齐
  // 这里不以这个模版的kAlignmentA = 1解释kAlignment的含义，而是按照下面模版的kAlignment = 128 / sizeof_bits<Element_type>::value解释
  // 首先这里的128表示，内存访问地址需要对齐128bit(16个byte)
  // 然后这里需要计算，在128位对齐的情况下，包含了多少个当前element type的数，即kAlignment
  // 之所以要算kAlignment，是因为cutlass还是以元素为单位工作的，而不是以byte或者bit为单位工作
  // 矩阵的起始地址向kAlignment对齐了（对齐了128bit），那么从矩阵中第一次取数后，第二次、第三次取数也肯定对齐了128bit
  // 例如使用cp.async读取type为bf16的数据，cp.async最大能一次读取16byte（128bit），所以这里kAignment就为 128 / 16 = 8 
  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  // threadblock tile shape
  using ThreadblockShape = GemmShape<128, 128, 8>;
  // warp tile shape
  using WarpShape = GemmShape<32, 64, 8>;
  // 问了下cursor，说这个参数的含义是，定义了单条指令可以执行的矩阵乘法运算的大小
  // 对于simt，就是GemmShape<1,1,1>
  // 对于tensorcore，这里就是mma指令的计算大小了，例如GemmShape<16,8,16>
  // 下面有许多模版的InstructionShape就是tensorcore的
  using InstructionShape = GemmShape<1, 1, 1>;
  // 这里表示double buffer，即cutlass中最简单的gemm都是使用了double buffer的
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    1,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ArchTag,
  typename ElementC>
struct DefaultGemmConfiguration<arch::OpClassSimt, ArchTag, int8_t, int8_t, ElementC, int32_t> {
  
  static int const kAlignmentA = 4;
  static int const kAlignmentB = 4;
  using ThreadblockShape = GemmShape<128, 128, 32>;
  using WarpShape = GemmShape<32, 64, 32>;
  using InstructionShape = GemmShape<1, 1, 4>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
    ElementC,
    1,
    int32_t,
    float
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <
  typename ArchTag,
  typename ElementA, 
  typename ElementB, 
  typename ElementC, 
  typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassWmmaTensorOp, 
  ArchTag,
  ElementA, 
  ElementB, 
  ElementC, 
  ElementAccumulator> {
  
  static int const kAlignmentA = 128 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 128 / sizeof_bits<ElementB>::value;

  static int const kStages = 2;
  
  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    128 / sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA, 
  typename ElementB, 
  typename ElementC, 
  typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm70,
  ElementA, 
  ElementB, 
  ElementC, 
  ElementAccumulator> {
  
  static int const kAlignmentA = 128 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 128 / sizeof_bits<ElementB>::value;

  using ThreadblockShape = GemmShape<128, 256, 32>;
  using WarpShape = GemmShape<64, 64, 32>;
  using InstructionShape = GemmShape<8, 8, 4>;
  static int const kStages = 2;
  
  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    128 / sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA, 
  typename ElementB, 
  typename ElementC, 
  typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75,
  ElementA, 
  ElementB, 
  ElementC, 
  ElementAccumulator> {

  static int const kAlignmentA = 128 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 128 / sizeof_bits<ElementA>::value;
  using ThreadblockShape = GemmShape<128, 256, 32>;
  using WarpShape = GemmShape<64, 64, 32>;
  using InstructionShape = GemmShape<16, 8, 8>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    128 / sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = typename platform::conditional<
      (platform::is_same<ElementA, int8_t>::value ||
       platform::is_same<ElementA, int4b_t>::value ||
       platform::is_same<ElementA, uint8_t>::value ||
       platform::is_same<ElementA, uint4b_t>::value),
      arch::OpMultiplyAddSaturate, arch::OpMultiplyAdd>::type;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  int8_t, 
  int8_t, 
  ElementC, 
  int32_t> {
  
  static int const kAlignmentA = 128 / sizeof_bits<int8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int8_t>::value;

  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<8, 8, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  int8_t, 
  uint8_t, 
  ElementC, 
  int32_t> {
  
  static int const kAlignmentA = 128 / sizeof_bits<int8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint8_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<8, 8, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  uint8_t, 
  int8_t, 
  ElementC, 
  int32_t> {
  
  static int const kAlignmentA = 128 / sizeof_bits<uint8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int8_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<8, 8, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  uint8_t, 
  uint8_t, 
  ElementC, 
  int32_t> {
  
  static int const kAlignmentA = 128 / sizeof_bits<uint8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint8_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<8, 8, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  int4b_t, 
  int4b_t, 
  ElementC, 
  int32_t> {
   
  static int const kAlignmentA = 128 / sizeof_bits<int4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int4b_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<8, 8, 32>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  int4b_t, 
  uint4b_t, 
  ElementC, 
  int32_t> {
    
  static int const kAlignmentA = 128 / sizeof_bits<int4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint4b_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<8, 8, 32>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  uint4b_t, 
  int4b_t, 
  ElementC, 
  int32_t> {
  
  static int const kAlignmentA = 128 / sizeof_bits<uint4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int4b_t>::value;

  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<8, 8, 32>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  uint4b_t, 
  uint4b_t, 
  ElementC, 
  int32_t> {
   
  static int const kAlignmentA = 128 / sizeof_bits<uint4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint4b_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<8, 8, 32>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  uint1b_t, 
  uint1b_t, 
  ElementC, 
  int32_t> {
    
  static int const kAlignmentA = 128 / sizeof_bits<uint1b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint1b_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 512>;
  using WarpShape = GemmShape<64, 64, 512>;
  using InstructionShape = GemmShape<8, 8, 128>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpXorPopc;
};

////////////////////////////////////////////////////////////////////////////////

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm80, ElementA,
                                ElementB, ElementC, ElementAccumulator> {

  static int const kAlignmentA = 128 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 128 / sizeof_bits<ElementA>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 16>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      ElementC, 128 / sizeof_bits<ElementC>::value, ElementAccumulator,
      ElementAccumulator>;

  using Operator = typename platform::conditional<
      (platform::is_same<ElementA, int8_t>::value ||
       platform::is_same<ElementA, int4b_t>::value ||
       platform::is_same<ElementA, uint8_t>::value ||
       platform::is_same<ElementA, uint4b_t>::value),
      arch::OpMultiplyAddSaturate, arch::OpMultiplyAdd>::type;
};

////////////////////////////////////////////////////////////////////////////////
template <typename ElementC,
          typename ElementAccumulator>
struct DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm80, double,
                                double, ElementC, ElementAccumulator> {

  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  
  using ThreadblockShape = GemmShape<128, 128, 16>;
  using WarpShape = GemmShape<32, 64, 16>;
  using InstructionShape = GemmShape<8, 8, 4>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      ElementC, 1, ElementAccumulator,
      ElementAccumulator>;

  using Operator = arch::OpMultiplyAdd;
};


template <>
struct DefaultGemmConfiguration<
    arch::OpClassTensorOp, 
    arch::Sm80, 
    complex<double>,
    complex<double>, 
    complex<double>,
    complex<double>
  > {

  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  
  using ThreadblockShape = GemmShape<64, 64, 16>;
  using WarpShape = GemmShape<32, 32, 16>;
  using InstructionShape = GemmShape<8, 8, 4>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      complex<double>, 1, complex<double>,
      complex<double>>;

  using Operator = arch::OpMultiplyAddComplex;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  int8_t, 
  int8_t, 
  ElementC, 
  int32_t> {
     
  static int const kAlignmentA = 128 / sizeof_bits<int8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int8_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 32>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  int8_t, 
  uint8_t, 
  ElementC, 
  int32_t> {
      
  static int const kAlignmentA = 128 / sizeof_bits<int8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint8_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 32>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  uint8_t, 
  int8_t, 
  ElementC, 
  int32_t> {
      
  static int const kAlignmentA = 128 / sizeof_bits<uint8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int8_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 32>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  uint8_t, 
  uint8_t, 
  ElementC, 
  int32_t> {
      
  static int const kAlignmentA = 128 / sizeof_bits<uint8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint8_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 32>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  int4b_t, 
  int4b_t, 
  ElementC, 
  int32_t> {
      
  static int const kAlignmentA = 128 / sizeof_bits<int4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int4b_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<16, 8, 64>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  int4b_t, 
  uint4b_t, 
  ElementC, 
  int32_t> {
       
  static int const kAlignmentA = 128 / sizeof_bits<int4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint4b_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<16, 8, 64>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  uint4b_t, 
  int4b_t, 
  ElementC, 
  int32_t> {
       
  static int const kAlignmentA = 128 / sizeof_bits<uint4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int4b_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<16, 8, 64>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  uint4b_t, 
  uint4b_t, 
  ElementC, 
  int32_t> {
       
  static int const kAlignmentA = 128 / sizeof_bits<uint4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint4b_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<16, 8, 64>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  uint1b_t, 
  uint1b_t, 
  ElementC, 
  int32_t> {
       
  static int const kAlignmentA = 128 / sizeof_bits<uint1b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint1b_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 512>;
  using WarpShape = GemmShape<64, 64, 512>;
  using InstructionShape = GemmShape<16, 8, 256>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////
template <typename ElementC,
          typename ElementAccumulator>
struct DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm90, double,
                                double, ElementC, ElementAccumulator> {

  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 4>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      ElementC, 1, ElementAccumulator,
      ElementAccumulator>;

  using Operator = arch::OpMultiplyAdd;
};

template <>
struct DefaultGemmConfiguration<
    arch::OpClassTensorOp, 
    arch::Sm90, 
    complex<double>,
    complex<double>, 
    complex<double>,
    complex<double>
  > {

  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  
  using ThreadblockShape = GemmShape<64, 64, 16>;
  using WarpShape = GemmShape<32, 32, 16>;
  using InstructionShape = GemmShape<16, 8, 4>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      complex<double>, 1, complex<double>,
      complex<double>>;

  using Operator = arch::OpMultiplyAddComplex;
};

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
