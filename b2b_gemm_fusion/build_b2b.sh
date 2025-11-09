#!/bin/bash

echo "==================================================================="
echo "编译 B2B GEMM FP16 Sm80 扁平化实现"
echo "==================================================================="

NVCC=nvcc
ARCH="-arch=sm_80"  # A100 或修改为 sm_86 (RTX 3090)
OPT="-O3"
STD="--std=c++14"
OUTPUT="b2b_gemm_f16_sm80"

echo ""
echo "编译命令:"
echo "$NVCC $ARCH $OPT $STD b2b_gemm_f16_sm80_complete.cu -o $OUTPUT"
echo ""

$NVCC $ARCH $OPT $STD b2b_gemm_f16_sm80_complete.cu -o $OUTPUT

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 编译成功!"
    echo ""
    echo "这个 kernel 使用的底层指令:"
    echo "  ✓ mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
    echo "  ✓ ldmatrix.sync.aligned.x4.m8n8.shared.b16"
    echo "  ✓ ldmatrix.sync.aligned.x2.m8n8.shared.b16"
    echo "  ✓ cp.async.ca.shared.global"
    echo "  ✓ cp.async.commit_group"
    echo "  ✓ cp.async.wait_group"
    echo ""
    echo "特性:"
    echo "  - 融合的两个 GEMM (避免中间结果写回 DRAM)"
    echo "  - FP16 Tensor Core 加速"
    echo "  - 3-stage pipeline"
    echo "  - Shared memory accumulator staging"
    echo "  - ReLU activation"
    echo ""
    echo "运行:"
    echo "  ./$OUTPUT"
    echo ""
else
    echo ""
    echo "✗ 编译失败"
    echo ""
    echo "注意: 需要 Sm80+ GPU (A100 或 RTX 3090)"
    exit 1
fi

