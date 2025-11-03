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

/**
This example shows how to use split-k version of matrix multiplication using functions and data
structures provided by CUTLASS; which we run on a NVIDIA Volta GPU.

What is split-k?
Consider a problem size of M = 128, N = 128, K = 4096. In this case, if my thread-block tile size (a
tile can be viewed as a 2d matrix) is 128x128x4096, then we launch a singled a thread-block taking
up a single SM of 84 SMs present on V100. Hence the efficiency of computation is really low. So, how
to solve it? This is where split-k comes in. It is a way of partitioning K-dimension of matrix
multiplication and distribute across multiple SMs and get better efficiency than single SM. In the
above example, we can partition K-dimension with split-k factor of 16 i.e., thread-block tile size
will be 128x128x256 and will be launching on 16 SMs. Once each thread-block computes their partial
inner product (1/16th of output), they accumulate to single output matrix.

Writing a single high performance matrix multiplication kernel is hard but do-able. Whereas writing
high performance kernels at scale which works for multiple problem sizes with good abstractions is
really hard. CUTLASS solves this problem by providing simplified abstractions to compose
multiple sections of gemm kernel. When used properly, the kernels can hit peak performance of GPU
easily.

CUTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
and thread-block level, they compute on their own tile-size with higher level of tile sizes being
composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
threadblock-tile (tile size computed by a threadblock).

In this example, we split variable initialization into
1. Setting up data properties : describes how matrices are laid out in the memory and how the kernel
can view them (logical to physical mapping)
2. Setting up computation properties : describes how the above set matrices will be used to compute
output of matrix multiplication.

First, we setup the data types of matrices A, B, C and D along with alpha, beta as the equation for
GEMM is D = alpha * A * B + beta * C. In CUTLASS, the kernels first compute A * B and leaves the
rest of the computation to end of the kernel as alpha * X + beta * C is a simple element-wise
operation on X (A * B) and C. We call this as epilogue of kernel. Hence, we setup data types for
alpha and beta to be equal to ElementComputeEpilogue = float. As we want to MMA instructions on
Volta and they support only half-precision floating point (fp16 or half), we use data type for
elements in input matrix A and B as cutlass::half_t. Volta also supports accumulation of partial dot
product to fp32, which can store wider range of numbers, we use it as data type of output matrix
elements and accumulation. We convey this to CUTLASS kernel by initializing template variables
ElementAccumulator (float), ElementComputeEpilogue (float), ElementInputA (cutlass::half_t),
ElementInputB (cutlass::half_t), ElementOutput (float). Communicating just the data type is not
enough. As the data is laid out linearly in memory, we have to convey the layout of matrices. We do
that by initializing template variable LayoutInputA to column major cutlass variable, LayoutInputB
to row major and LayoutOutput to row major. Next, we setup rules to compute alpha * X + beta * C
which is called epilogue of the kernel. We initialize template variable EpilogueOp, which takes the
data type of output ElementOutput (float), the number of elements per vector memory access (16),
data type of accumulator (float) and data type of computation of linear combination (alpha * X +
beta * C).

Now that we setup the properties of data, we have to setup properties of computation.

Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x128x32,
64x64x4, 8x8x4 (MxNxK) respectively. When passed to instantiate CUTLASS GEMM kernel, it internally
deduce the amount of threads needed per thread-block, amount of shared memory, storing data in
bank-conflict free manner, and ton of other variables required to compose, initialize and launch a
high performance GEMM kernel. This is the beauty of CUTLASS, it relieves developer from
understanding and coding complicated hardware optimizations which can easily go wrong.

There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an SM, CUDA SM architecture of GPU you want to run on.

These are all put together to create a template variable which describes CUTLASS GEMM kernel using
cutlass::gemm::device::GemmSplitKParallel template.

The next step is to initialize physical data, instantiate and initialize CUTLASS kernel and run it.
We use CUTLASS utilities to initialize, fill, compare matrices as they are simple and doesn't come
in the way of learning CUTLASS.

Once all the matrices are initialized and filled with data, create arguments tuple to launch CUTLASS
kernel which takes problem size (M = 5120, N = 4096 and K = 4096), matrices, alpha, beta and the
important one, split k-dimension factor. Along with that, we query CUTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to initialize CUTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference gemm kernel (from CUTLASS utilities) to compare if
the output from CUTLASS kernel is same as reference GEMM kernel.


这个示例展示了如何使用CUTLASS提供的函数和数据结构来使用split-k版本的矩阵乘法；
我们在NVIDIA Volta GPU上运行这个示例。

什么是split-k？
考虑一个问题规模为M = 128, N = 128, K = 4096。在这种情况下，如果我的线程块瓦片大小
（瓦片可以视为一个2D矩阵）是128x128x4096，那么我们启动单个线程块，只占用V100上84个SM
中的一个SM。因此计算效率非常低。那么如何解决？这就是split-k的用武之地。它是一种
对矩阵乘法的K维度进行分割并分布到多个SM上的方法，从而获得比单个SM更好的效率。在上面
的例子中，我们可以用split-k因子16来分割K维度，即线程块瓦片大小将是128x128x256，
并将在16个SM上启动。一旦每个线程块计算出它们的部分内积（输出的1/16），它们就累加到
单个输出矩阵中。

编写单个高性能矩阵乘法kernel是困难但可行的。而编写大规模的高性能kernel，能够以良好
的抽象为多种问题规模工作，这真的很困难。CUTLASS通过提供简化的抽象来组合GEMM kernel
的多个部分来解决这个问题。当正确使用时，这些kernel可以轻松达到GPU的峰值性能。

CUTLASS将kernel分为分层的可组合部分。这意味着，在每个线程、warp和线程块级别，它们
在自己的瓦片大小上计算，较高级别的瓦片大小由较低级别的瓦片组成。多个线程瓦片（每个
线程计算的瓦片大小）可以用来形成warp瓦片（每个warp计算的瓦片大小），多个warp瓦片
可以用来计算线程块瓦片（一个线程块计算的瓦片大小）。

在这个示例中，我们将变量初始化分为：
1. 设置数据属性：描述矩阵在内存中的布局方式以及kernel如何查看它们（逻辑到物理的映射）
2. 设置计算属性：描述如何使用上述设置的矩阵来计算矩阵乘法的输出。

首先，我们设置矩阵A、B、C和D的数据类型以及alpha、beta，因为GEMM的方程式是
D = alpha * A * B + beta * C。在CUTLASS中，kernel首先计算A * B，并将其余计算留到
kernel的末尾，因为alpha * X + beta * C是对X（A * B）和C的简单逐元素操作。我们称之为
kernel的后处理（epilogue）。因此，我们将alpha和beta的数据类型设置为等于
ElementComputeEpilogue = float。由于我们想在Volta上使用MMA指令，而它们只支持半精度
浮点（fp16或half），我们对输入矩阵A和B中的元素使用cutlass::half_t数据类型。Volta
还支持将部分点积累加到fp32，这可以存储更宽范围的数字，我们将其用作输出矩阵元素和
累加的数据类型。我们通过初始化模板变量ElementAccumulator（float）、
ElementComputeEpilogue（float）、ElementInputA（cutlass::half_t）、
ElementInputB（cutlass::half_t）、ElementOutput（float）来向CUTLASS kernel传达这一点。
仅仅传达数据类型是不够的。由于数据在内存中是线性布局的，我们必须传达矩阵的布局。
我们通过将模板变量LayoutInputA初始化为列主序cutlass变量、LayoutInputB为行主序、
LayoutOutput为行主序来做到这一点。接下来，我们设置计算alpha * X + beta * C的规则，
这被称为kernel的后处理。我们初始化模板变量EpilogueOp，它接受输出ElementOutput（float）
的数据类型、每个向量内存访问的元素数量（16）、累加器的数据类型（float）和线性组合
（alpha * X + beta * C）计算的数据类型。

现在我们设置了数据的属性，我们必须设置计算的属性。

其次，我们创建线程块、warp和mma-op的瓦片大小的模板变量，分别为128x128x32、64x64x4、
8x8x4（MxNxK）。当传递给实例化CUTLASS GEMM kernel时，它内部推导出每个线程块所需的
线程数量、共享内存量、以无bank冲突的方式存储数据，以及组成、初始化和启动高性能GEMM 
kernel所需的大量其他变量。这就是CUTLASS的美妙之处，它使开发者免于理解和编码复杂的
硬件优化，这些优化很容易出错。

还有一些其他初始化的模板变量，例如，输出矩阵的哪个线程块瓦片由在SM上启动的哪个线程块
完成，您想要运行的GPU的CUDA SM架构。

这些都被放在一起，使用cutlass::gemm::device::GemmSplitKParallel模板创建一个描述
CUTLASS GEMM kernel的模板变量。

下一步是初始化物理数据、实例化和初始化CUTLASS kernel并运行它。我们使用CUTLASS工具
来初始化、填充、比较矩阵，因为它们简单且不会妨碍学习CUTLASS。

一旦所有矩阵都被初始化并填充了数据，创建参数元组来启动CUTLASS kernel，它接受问题
规模（M = 5120, N = 4096和K = 4096）、矩阵、alpha、beta和重要的split k维度因子。
除此之外，我们查询CUTLASS是否需要我们实例化的kernel所需的任何临时空间内存。如果是，
我们创建它并与创建的其他参数一起传递以初始化CUTLASS kernel，然后启动kernel。

在这个示例中，我们稍后启动一个参考GEMM kernel（来自CUTLASS工具）来比较CUTLASS 
kernel的输出是否与参考GEMM kernel相同。


*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4

// This code section describes ?
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- This is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Put all the created template variables to create GemmSplitKParallel template variable
using Gemm = cutlass::gemm::device::GemmSplitKParallel<ElementInputA,
                                                       LayoutInputA,
                                                       ElementInputB,
                                                       LayoutInputB,
                                                       ElementOutput,
                                                       LayoutOutput,
                                                       ElementAccumulator,
                                                       MMAOp,
                                                       SmArch,
                                                       ShapeMMAThreadBlock,
                                                       ShapeMMAWarp,
                                                       ShapeMMAOp,
                                                       EpilogueOp>;

int run() {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 7) {
    std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
              << std::endl;

    // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }

  //
  // Define problem size
  //

  const int length_m = 5120;
  const int length_n = 4096;
  const int length_k = 4096;

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      0);  // <- Fill matrix A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(4),
      ElementInputB(-4),
      0);  // <- Fill matrix B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 16 partitions
  int split_k_slices = 16;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  CUTLASS_CHECK(status);

  // Create instantiation for device reference gemm kernel
  cutlass::reference::device::Gemm<ElementInputA,
                                   LayoutInputA,
                                   ElementInputB,
                                   LayoutInputB,
                                   ElementOutput,
                                   LayoutOutput,
                                   ElementComputeEpilogue,
                                   ElementComputeEpilogue>
      gemm_device;

  // Launch device reference gemm kernel
  gemm_device(problem_size,
              alpha,
              tensor_a.device_ref(),
              tensor_b.device_ref(),
              beta,
              tensor_c.device_ref(),
              tensor_ref_d.device_ref());

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());

  std::cout << (passed ? "Passed" : "Failed") << std::endl;

  return (passed ? 0  : -1);
}

int main() {

  //
  // Volta Tensor Core operations exposed with mma.sync are first available in CUDA 10.1.
  //
  // CUTLASS must be compiled with CUDA 10.1 Toolkit to run these examples.
  //
  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
    std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;

    // Returning zero, so this test passes when built with older CUDA Toolkits. Its action are no-op.
    return 0;
  }
  else {
    return run();
  }
}

