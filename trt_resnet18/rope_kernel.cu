#include "rope_kernel.h"
#include <cuda_runtime.h>

// One threadblock per (batch, head, token position)
// One thread per dimension pair (2i, 2i+1)
__global__ void ropeKernel(
    const float* input,
    float*       output,
    int          num_heads,
    int          seq_len,
    int          head_dim)
{
    int batch    = blockIdx.z;
    int head     = blockIdx.y;
    int pos      = blockIdx.x;
    int pair_idx = threadIdx.x;

    if (pair_idx >= head_dim / 2) return;

    int offset = batch * (num_heads * seq_len * head_dim)
               + head  * (seq_len * head_dim)
               + pos   * head_dim
               + 2 * pair_idx;

    float x0 = input[offset];
    float x1 = input[offset + 1];

    float theta   = powf(10000.0f, -2.0f * pair_idx / head_dim);
    float cos_val = cosf(pos * theta);
    float sin_val = sinf(pos * theta);

    output[offset]     = x0 * cos_val - x1 * sin_val;
    output[offset + 1] = x0 * sin_val + x1 * cos_val;
}

void launchRopeKernel(
    const float* input,
    float*       output,
    int          batch,
    int          num_heads,
    int          seq_len,
    int          head_dim,
    cudaStream_t stream)
{
    dim3 grid(seq_len, num_heads, batch);
    dim3 block(head_dim / 2);

    // Use cudaLaunchKernel to bypass the broken __cudaLaunch stub in CUDA 12.8
    void* args[] = {
        (void*)&input,
        (void*)&output,
        (void*)&num_heads,
        (void*)&seq_len,
        (void*)&head_dim
    };
    cudaLaunchKernel((const void*)ropeKernel, grid, block, args, 0, stream);
}
