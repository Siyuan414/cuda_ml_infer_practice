#include "trt_common.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>   // header-only C++ API — no link needed in CUDA 12

// ── TRTInferencer ─────────────────────────────────────────────────────────────
class TRTInferencer {
public:
    explicit TRTInferencer(Logger& logger) : logger_(logger) {}

    ~TRTInferencer() {
        if (d_input_)  cudaFree(d_input_);
        if (d_output_) cudaFree(d_output_);
        if (stream_)   cudaStreamDestroy(stream_);
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open: " + path);
        std::vector<char> data((std::istreambuf_iterator<char>(f)), {});

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
        if (!engine_) throw std::runtime_error("Failed to deserialize engine");

        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("Failed to create context");

        constexpr size_t max_elems = 4ULL * 512ULL * 256ULL;
        cudaMalloc(&d_input_,  max_elems * sizeof(float));
        cudaMalloc(&d_output_, max_elems * sizeof(float));
        cudaStreamCreate(&stream_);
        std::cout << "Engine loaded: " << path << "\n";
    }

    // Expose engine so double-buffer benchmark can create a second context
    nvinfer1::ICudaEngine* engine() { return engine_.get(); }

    void infer(int batch, int seq_len,
               const std::vector<float>& h_input,
               std::vector<float>&       h_output)
    {
        context_->setInputShape("input", nvinfer1::Dims3{batch, seq_len, 256});
        context_->setTensorAddress("input",  d_input_);
        context_->setTensorAddress("output", d_output_);

        { nvtx3::scoped_range r{"H2D"};
          cudaMemcpyAsync(d_input_, h_input.data(),
                          h_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice, stream_); }

        { nvtx3::scoped_range r{"enqueue"};
          context_->enqueueV3(stream_); }

        size_t out_elems = static_cast<size_t>(batch) * seq_len * 256;
        h_output.resize(out_elems);
        { nvtx3::scoped_range r{"D2H+sync"};
          cudaMemcpyAsync(h_output.data(), d_output_,
                          out_elems * sizeof(float),
                          cudaMemcpyDeviceToHost, stream_);
          cudaStreamSynchronize(stream_); }
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

// ── Double-buffer benchmark ───────────────────────────────────────────────────
// Two contexts, two streams, two buffer sets.
// While stream[prev] is doing D2H, stream[cur] is already running H2D+compute.
// This overlaps the CPU-idle bubble we saw in the single-stream profile.
double benchmarkDoubleBuffer(nvinfer1::ICudaEngine* engine,
                             int batch, int seq_len, int N)
{
    constexpr int S = 2;
    constexpr size_t max_elems = 4ULL * 512ULL * 256ULL;
    const size_t n = static_cast<size_t>(batch) * seq_len * 256;

    // Two independent contexts (each has its own internal state)
    TRTUniquePtr<nvinfer1::IExecutionContext> ctx[S] = {
        TRTUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext()),
        TRTUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext())
    };

    // Two streams
    cudaStream_t streams[S];
    for (int s = 0; s < S; ++s) cudaStreamCreate(&streams[s]);

    // Two sets of device buffers
    void* d_in[S], *d_out[S];
    for (int s = 0; s < S; ++s) {
        cudaMalloc(&d_in[s],  max_elems * sizeof(float));
        cudaMalloc(&d_out[s], max_elems * sizeof(float));
    }

    // Host side: one input, two output slots
    std::vector<float> h_in(n, 1.0f);
    std::vector<float> h_out[S];
    for (int s = 0; s < S; ++s) h_out[s].resize(n);

    // Submit one slot asynchronously
    auto submit = [&](int slot) {
        int s = slot % S;
        ctx[s]->setInputShape("input", nvinfer1::Dims3{batch, seq_len, 256});
        ctx[s]->setTensorAddress("input",  d_in[s]);
        ctx[s]->setTensorAddress("output", d_out[s]);
        cudaMemcpyAsync(d_in[s],  h_in.data(),      n * sizeof(float),
                        cudaMemcpyHostToDevice,  streams[s]);
        ctx[s]->enqueueV3(streams[s]);
        cudaMemcpyAsync(h_out[s].data(), d_out[s], n * sizeof(float),
                        cudaMemcpyDeviceToHost, streams[s]);
    };

    // Warmup both slots
    for (int s = 0; s < S; ++s) { submit(s); cudaStreamSynchronize(streams[s]); }

    // ── Ping-pong loop ────────────────────────────────────────────────────────
    // submit cur → sync prev → repeat
    // GPU timeline:  [H2D₀|compute₀|D2H₀]
    //                              [H2D₁|compute₁|D2H₁]
    //                                           [H2D₀|compute₀|D2H₀] ...
    auto t0 = std::chrono::steady_clock::now();

    submit(0);                          // kick off first slot
    for (int i = 1; i < N; ++i) {
        int cur  = i % S;
        int prev = 1 - cur;
        submit(cur);                    // submit next (async — GPU works on it)
        cudaStreamSynchronize(streams[prev]); // wait for previous output
    }
    cudaStreamSynchronize(streams[(N - 1) % S]); // drain last slot

    auto t1 = std::chrono::steady_clock::now();

    // Cleanup
    for (int s = 0; s < S; ++s) {
        cudaFree(d_in[s]); cudaFree(d_out[s]);
        cudaStreamDestroy(streams[s]);
    }

    return std::chrono::duration<double>(t1 - t0).count() * 1000.0 / N;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    Logger logger(nvinfer1::ILogger::Severity::kWARNING);
    TRTInferencer inferencer(logger);

    auto t0 = std::chrono::steady_clock::now();
    inferencer.load("llama_attn_fp16.trt");
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "Deserialize time : "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n\n";

    const int batch   = 1;
    const int seq_len = 128;
    const size_t n    = batch * seq_len * 256;
    const int N       = 100;

    std::vector<float> h_input(n, 1.0f);
    std::vector<float> h_output;

    // ── Single-stream baseline ─────────────────────────────────────────────
    inferencer.infer(batch, seq_len, h_input, h_output);  // warmup
    auto b0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i)
        inferencer.infer(batch, seq_len, h_input, h_output);
    auto b1 = std::chrono::steady_clock::now();
    double single_ms = std::chrono::duration<double>(b1 - b0).count() * 1000.0 / N;

    // ── Double-buffer benchmark ────────────────────────────────────────────
    double double_ms = benchmarkDoubleBuffer(
        inferencer.engine(), batch, seq_len, N);

    std::cout << "Single-stream  : " << single_ms << " ms/iter\n";
    std::cout << "Double-buffered: " << double_ms << " ms/iter\n";
    std::cout << "Speedup        : " << single_ms / double_ms << "x\n";

    float sum = 0.f;
    for (float v : h_output) sum += v;
    std::cout << "Output sum (sanity): " << sum << "\n";

    return 0;
}
