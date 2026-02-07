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
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "device/b2b_gemm.h"
#include "b2b_gemm_run.h"
#include "test_run.h"

////////////////////////////////////////////////////////////////////////////////

cutlass::gemm::GemmCoord gemm_f16_sm80_problem_size_0(128*640, 64, 576);
cutlass::gemm::GemmCoord gemm_f16_sm80_problem_size_1(128*640, 256, 64);

bool run_nonfused_gemm_f16_sm80() {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  ElementCompute alpha0 = ElementCompute(1);
  ElementCompute beta0 = ElementCompute(1); //beta=1 for bias
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape0 = cutlass::gemm::GemmShape<32, 32, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 256, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using Gemm0 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    WarpShape0,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationMish<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3
  >;
  using Gemm1 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape1,
    WarpShape1,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationMish<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3
  >;

  B2bNonFusedGemmRun<Gemm0, Gemm1> nonFusedGemm;

  std::cout << "Running Non-fused back-to-back FP16 TN GEMMs...\n";
  bool pass = nonFusedGemm.run(gemm_f16_sm80_problem_size_0, gemm_f16_sm80_problem_size_1, alpha0, beta0, alpha1, beta1);
  if(pass)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return pass;
}

bool run_fused_gemm_f16_sm80_shmem() {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  ElementCompute alpha0 = ElementCompute(1);
  //Fused kernel has built-in bias, setting beta=0
  ElementCompute beta0 = ElementCompute(0); 
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape0 = cutlass::gemm::GemmShape<32, 32, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 256, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  /*
  下面两个epilogue模版类的传入模版参数的第二个参数表示一个thread处理的元素个数
  可以看到epilogue0和1的这个参数的大小不相同
  epilogue0设置为InstructionShape::kM * InstructionShape::kN / 32(这里结果为4)的原因：
  首先InstructionShape::kM * InstructionShape::kN表示一个warp的mma计算得到的结果的数据大小
  然后除以32就表示一个thread的寄存器中包含的数据个数
  所以对于epilogue0，每个thread处理当前thread中C矩阵寄存器的结果数量
  而对epilogue1，则是每个thread处理128 / cutlass::sizeof_bits<ElementOutput>::value(这里结果为8)，即
  每个线程向量化处理128bit的数据，这估计是为了使用最大的128bit向量化写回gmem

  然后有个问题就是，为什么epilogue0和epilogue1的每个thread处理的数据量不能一样？
  为什么epi1中每个thread能处理的数据量比epi0多一倍？
  这是因为，在b2bgemm的gemm1的mainloop算完之后，会调用include/cutlass/epilogue/threadblock/epilogue.h中的()重载
  先将计算结果写入smem，然后在smem中对数据的layout做处理，处理为行优先的排列，然后再对smem中的数据做epilogue1, 所以epi1可以一个thread处理8个数
  我理解是因为gemm1算完之后的，由于后面没有gemm了，所以不需要再保证结果数据的layout满足mma的要求，所以就可以放入smem做layout重排，然后epi1可以用更大的向量化处理方式
  而epi0因为还需要满足结果的layout满足mma对layout的要求（因为gemm1还需要用gemm0的结果数据），所以只能一个thread处理4个数
  */
  using EpilogueOutputOp0 = 
    cutlass::epilogue::thread::LinearCombinationMish<
      ElementOutput,
      InstructionShape::kM * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

  using EpilogueOutputOp1 = 
    cutlass::epilogue::thread::LinearCombinationMish<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;


  const bool SmemAccumulator = true;

  using B2bGemm = cutlass::gemm::device::B2bGemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3,
    SmemAccumulator
  >;

  B2bFusedGemmRun<B2bGemm> fusedGemm;

  std::cout << "Running Fused back-to-back FP16 TN GEMMs with shared memory staging...\n";
  bool passed = fusedGemm.run(gemm_f16_sm80_problem_size_0, gemm_f16_sm80_problem_size_1, alpha0, beta0, alpha1, beta1);
  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;

}


int main() {

  std::vector<bool (*)()>funcs = {
    &run_nonfused_gemm_f16_sm80,
    &run_fused_gemm_f16_sm80_shmem
  };

  return testRun(80, funcs, "gemm f16 shmem staging");


}



////////////////////////////////////////////////////////////////////////////////
