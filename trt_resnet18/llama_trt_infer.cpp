#include "trt_common.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime_api.h>

// ── TRTInferencer ─────────────────────────────────────────────────────────────
// Loads a serialized .trt engine and runs inference with dynamic seq_len.
// Buffers are allocated once at max size [4, 512, 256] and reused per call.
class TRTInferencer {
public:
    explicit TRTInferencer(Logger& logger) : logger_(logger) {}

    ~TRTInferencer() {
        if (d_input_)  cudaFree(d_input_);
        if (d_output_) cudaFree(d_output_);
        if (stream_)   cudaStreamDestroy(stream_);
    }

    // Load engine from .trt file and allocate max-size GPU buffers
    void load(const std::string& path) {
        // 1. Read serialized engine bytes
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open: " + path);
        std::vector<char> data((std::istreambuf_iterator<char>(f)), {});

        // 2. Deserialize — much faster than rebuilding from ONNX
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
        if (!engine_) throw std::runtime_error("Failed to deserialize engine");

        // 3. Create execution context
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("Failed to create context");

        // 4. Allocate GPU buffers at max shape [4, 512, 256]
        //    FP16 engine: 2 bytes per element
        constexpr size_t max_elems = 4ULL * 512ULL * 256ULL;
        cudaMalloc(&d_input_,  max_elems * 2);  // fp16
        cudaMalloc(&d_output_, max_elems * 2);  // fp16

        cudaStreamCreate(&stream_);
        std::cout << "Engine loaded: " << path << "\n";
    }

    // Run inference for shape [batch, seq_len, 256]
    // h_input / h_output are fp32 host buffers
    void infer(int batch, int seq_len,
               const std::vector<float>& h_input,
               std::vector<float>&       h_output)
    {
        // 5. Tell the context the actual shape for this call
        context_->setInputShape("input", nvinfer1::Dims3{batch, seq_len, 256});

        // 6. Bind device buffers (must be re-bound after setInputShape)
        context_->setTensorAddress("input",  d_input_);
        context_->setTensorAddress("output", d_output_);

        // 7. Copy fp32 input H2D — TRT casts to fp16 internally
        cudaMemcpyAsync(d_input_, h_input.data(),
                        h_input.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);

        // 8. Run inference
        context_->enqueueV3(stream_);

        // 9. Copy output D2H
        size_t out_elems = static_cast<size_t>(batch) * seq_len * 256;
        h_output.resize(out_elems);
        cudaMemcpyAsync(h_output.data(), d_output_,
                        out_elems * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }

private:
    Logger&                                   logger_;
    TRTUniquePtr<nvinfer1::IRuntime>          runtime_;
    TRTUniquePtr<nvinfer1::ICudaEngine>       engine_;
    TRTUniquePtr<nvinfer1::IExecutionContext> context_;
    void*        d_input_  = nullptr;
    void*        d_output_ = nullptr;
    cudaStream_t stream_   = nullptr;
};

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    Logger logger(nvinfer1::ILogger::Severity::kWARNING);
    TRTInferencer inferencer(logger);

    // ── Cold-start deserialization ─────────────────────────────────────────
    auto t0 = std::chrono::steady_clock::now();
    inferencer.load("llama_attn_fp16.trt");
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "Deserialize time : "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n\n";

    // ── Benchmark at seq_len=128 ───────────────────────────────────────────
    const int batch   = 1;
    const int seq_len = 128;
    const size_t n    = batch * seq_len * 256;

    std::vector<float> h_input(n, 1.0f);
    std::vector<float> h_output;

    // Warmup
    inferencer.infer(batch, seq_len, h_input, h_output);

    // 100-iteration benchmark
    const int N = 100;
    auto b0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i)
        inferencer.infer(batch, seq_len, h_input, h_output);
    auto b1 = std::chrono::steady_clock::now();

    double avg_ms = std::chrono::duration<double>(b1 - b0).count() * 1000.0 / N;
    std::cout << "seq_len=" << seq_len
              << "  avg latency : " << avg_ms << " ms/iter\n";

    // Sanity: output should be non-zero
    float sum = 0.f;
    for (float v : h_output) sum += v;
    std::cout << "Output sum (sanity): " << sum << "\n";

    return 0;
}
