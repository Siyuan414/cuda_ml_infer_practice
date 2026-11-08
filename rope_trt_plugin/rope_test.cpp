#include "rope_kernel.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cuda_runtime_api.h>

// ── Helpers ───────────────────────────────────────────────────────────────────
std::vector<float> loadBin(const std::string& path, size_t num_floats) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<float> data(num_floats);
    f.read(reinterpret_cast<char*>(data.data()), num_floats * sizeof(float));
    return data;
}

float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    return max_diff;
}

double meanAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += std::abs(a[i] - b[i]);
    return sum / a.size();
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    // 1. Dimensions — must match rope_ref.py exactly
    const int batch     = 1;
    const int num_heads = 2;
    const int seq_len   = 8;
    const int head_dim  = 64;
    const size_t num_floats = batch * num_heads * seq_len * head_dim;  // 1024

    // 2. Load input and reference from disk
    auto h_input = loadBin("rope_input.bin",      num_floats);
    auto h_ref   = loadBin("rope_output_ref.bin", num_floats);

    // 3. Allocate GPU buffers and copy input H2D
    float* d_input  = nullptr;
    float* d_output = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_input),  num_floats * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_output), num_floats * sizeof(float));
    cudaMemcpy(d_input, h_input.data(), num_floats * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Launch RoPE kernel on a stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    launchRopeKernel(d_input, d_output, batch, num_heads, seq_len, head_dim, stream);

    // 5. Sync and copy output D2H
    cudaStreamSynchronize(stream);
    std::vector<float> h_output(num_floats);
    cudaMemcpy(h_output.data(), d_output, num_floats * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Compare against PyTorch reference
    std::cout << "=== RoPE Kernel Validation ===\n";
    std::cout << "Max abs diff  : " << maxAbsDiff(h_output, h_ref)  << "\n";
    std::cout << "Mean abs diff : " << meanAbsDiff(h_output, h_ref) << "\n";

    // Spot-check pos=1 (pos=0 is always identity, not useful for verification)
    int offset = 1 * head_dim;  // batch=0, head=0, pos=1
    std::cout << "\nSpot-check pos=1, dims 0-3:\n";
    for (int d = 0; d < 4; ++d)
        std::cout << "  out[" << d << "] kernel=" << h_output[offset+d]
                  << "  ref=" << h_ref[offset+d] << "\n";

    // 7. Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    std::cout << "\nDone.\n";
    return 0;
}
