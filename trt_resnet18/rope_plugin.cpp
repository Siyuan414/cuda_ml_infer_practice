#include "rope_plugin.h"
#include "rope_kernel.h"  // plain C++ launcher — no CUDA kernel here

int RoPEPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    int batch   = inputDesc[0].dims.d[0];
    int seq_len = inputDesc[0].dims.d[2];

    const float* in  = static_cast<const float*>(inputs[0]);
    float*       out = static_cast<float*>(outputs[0]);

    launchRopeKernel(in, out, batch, num_heads_, seq_len, head_dim_, stream);
    return 0;
}
