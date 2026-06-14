#pragma once
#include <cuda_runtime_api.h>

// Pure CUDA kernel launcher — no TRT headers, callable from both .cu and .cpp
void launchRopeKernel(
    const float* input,
    float*       output,
    int          batch,
    int          num_heads,
    int          seq_len,
    int          head_dim,
    cudaStream_t stream);
