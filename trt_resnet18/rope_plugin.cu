#include "rope_plugin.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void ropeKernel(const float* input, float* output, int num_heads, int seq_len, int head_dim){
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int pos = blockIdx.x;
    int pair_idx = threadIdx.x;

    if(pair_idx < head_dim / 2){
        int idx1 = batch * (num_heads * seq_len * head_dim) + head * (seq_len * head_dim) + pos * head_dim + 2*pair_idx;
        int idx2 = idx1 + 1;
        float x0 = input[idx1];
        float x1 = input[idx2];
        float theta = powf(10000.0f, -2.0f * pair_idx / head_dim);
        float cos_val = cosf(pos * theta);
        float sin_val = sinf(pos * theta);

        output[idx1] = x0 * cos_val - x1 * sin_val;
        output[idx2] = x0 * sin_val + x1 * cos_val;
    }

}

int RoPEPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream
) noexcept {
    int batch = inputDesc[0].dims.d[0];
    int seq_len = inputDesc[0].dims.d[2];
    int head_dim = inputDesc[0].dims.d[3];

    const float* input_device = static_cast<const float*>(inputs[0]);
    float* output_device = static_cast<float*>(outputs[0]);

    dim3 grid(seq_len, num_heads_, batch);
    dim3 block(head_dim / 2);

    ropeKernel<<<grid, block, 0, stream>>>(input_device, output_device, num_heads_, seq_len, head_dim);
   
    return 0;
}