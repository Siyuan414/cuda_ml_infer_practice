#include "trt_common.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <stdexcept>
#include <cstring>

// ── INT8 Entropy Calibrator ───────────────────────────────────────────────────
// Implements IInt8EntropyCalibrator2 — TRT's recommended calibrator for CNNs
// and attention blocks.  Entropy calibration minimises KL-divergence between
// the original fp32 distribution and the quantised INT8 distribution.
//
// Strategy: generate NUM_BATCHES random fp32 batches that look like real
// attention inputs (standard-normal, shape [1, 128, 256]).  In production
// you would feed real representative data; random data is fine for a
// lab exercise because the calibrator only needs to see the activation
// *distribution*, not correct outputs.
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    static constexpr int   NUM_BATCHES = 100;
    static constexpr int   BATCH       = 1;
    static constexpr int   SEQ_LEN     = 128;
    static constexpr int   HIDDEN      = 256;
    static constexpr size_t ELEMS      = static_cast<size_t>(BATCH) * SEQ_LEN * HIDDEN;

    explicit Int8EntropyCalibrator(const std::string& cache_path)
        : cache_path_(cache_path), batch_idx_(0), d_input_(nullptr)
    {
        // Allocate one device batch buffer (reused every getBatch call)
        cudaMalloc(&d_input_, ELEMS * sizeof(float));

        // Pre-generate all calibration data on the host
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.f, 1.f);
        calib_data_.resize(NUM_BATCHES * ELEMS);
        for (auto& v : calib_data_) v = dist(rng);

        std::cout << "Calibrator: " << NUM_BATCHES
                  << " batches x [" << BATCH << "," << SEQ_LEN << "," << HIDDEN << "]\n";
    }

    ~Int8EntropyCalibrator() override {
        if (d_input_) cudaFree(d_input_);
    }

    // ── IInt8EntropyCalibrator2 interface ─────────────────────────────────────

    // TRT calls this to learn the calibration batch size
    int getBatchSize() const noexcept override { return BATCH; }

    // TRT calls this repeatedly until it returns false.
    // bindings[i] must be set to the device pointer for input i.
    bool getBatch(void* bindings[], const char* names[],
                  int nbBindings) noexcept override
    {
        if (batch_idx_ >= NUM_BATCHES) return false;

        const float* src = calib_data_.data() + batch_idx_ * ELEMS;
        cudaMemcpy(d_input_, src, ELEMS * sizeof(float), cudaMemcpyHostToDevice);
        bindings[0] = d_input_;   // only one input tensor: "input"
        ++batch_idx_;
        return true;
    }

    // Return cached calibration table (avoids re-running calibration each build)
    const void* readCalibrationCache(size_t& length) noexcept override {
        std::ifstream f(cache_path_, std::ios::binary);
        if (!f) { length = 0; return nullptr; }
        calib_cache_.assign(std::istreambuf_iterator<char>(f), {});
        length = calib_cache_.size();
        std::cout << "Loaded calibration cache: " << length << " bytes\n";
        return calib_cache_.data();
    }

    // TRT writes the calibration table here after running calibration
    void writeCalibrationCache(const void* ptr, size_t length) noexcept override {
        std::ofstream f(cache_path_, std::ios::binary);
        f.write(reinterpret_cast<const char*>(ptr), length);
        std::cout << "Wrote calibration cache: " << length
                  << " bytes → " << cache_path_ << "\n";
    }

private:
    std::string          cache_path_;
    int                  batch_idx_;
    void*                d_input_;
    std::vector<float>   calib_data_;
    std::vector<char>    calib_cache_;
};

// ── Build INT8 engine ─────────────────────────────────────────────────────────
std::vector<char> buildInt8Engine(
    const std::string& onnx_path,
    Logger& logger,
    Int8EntropyCalibrator& calibrator,
    size_t workspace_mb = 2048)
{
    TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("Failed to create builder");

    const uint32_t flags = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(flags));

    TRTUniquePtr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX");
    }
    std::cout << "Parsed ONNX: " << network->getNbLayers() << " layers\n";

    TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               workspace_mb * 1024ULL * 1024ULL);

    // INT8 requires both kINT8 flag AND a calibrator.
    // kFP16 is ALSO required — it provides the fallback precision for ops that
    // cannot run INT8 (softmax, dequantize boundary nodes, layernorm, etc.).
    // Without kFP16 those ops fall back to FP32, causing mixed-precision errors.
    if (!builder->platformHasFastInt8())
        throw std::runtime_error("GPU does not support fast INT8");

    config->setFlag(nvinfer1::BuilderFlag::kFP16);   // fallback for non-INT8 ops
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setInt8Calibrator(&calibrator);
    std::cout << "INT8 + FP16-fallback mode enabled\n";

    // Dynamic-shape profile — same as FP16 builder
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims3{1,   1, 256});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims3{1, 128, 256});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims3{4, 512, 256});
    config->addOptimizationProfile(profile);

    // Tell TRT which shape to use when collecting activation statistics.
    // Without this, dynamic-shape networks calibrate at the minimum dims,
    // giving the calibrator wrong scale estimates for the opt/max shapes.
    auto calib_profile = builder->createOptimizationProfile();
    calib_profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN,
                                 nvinfer1::Dims3{1, 128, 256});
    calib_profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT,
                                 nvinfer1::Dims3{1, 128, 256});
    calib_profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX,
                                 nvinfer1::Dims3{1, 128, 256});
    config->setCalibrationProfile(calib_profile);

    std::cout << "Running INT8 calibration (100 batches)...\n";
    TRTUniquePtr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) throw std::runtime_error("Failed to build INT8 engine");

    return std::vector<char>(
        static_cast<const char*>(serialized->data()),
        static_cast<const char*>(serialized->data()) + serialized->size());
}

// ── Quick accuracy check ──────────────────────────────────────────────────────
// Load both FP16 and INT8 engines, run the same input, compare output sums.
// A good INT8 build should produce output within ~1% of FP16.
void compareAccuracy(Logger& logger,
                     const std::string& fp16_path,
                     const std::string& int8_path)
{
    constexpr int B = 1, S = 128, H = 256;
    const size_t n = B * S * H;

    auto runEngine = [&](const std::string& path) -> float {
        std::ifstream f(path, std::ios::binary);
        std::vector<char> data((std::istreambuf_iterator<char>(f)), {});
        TRTUniquePtr<nvinfer1::IRuntime>          rt(nvinfer1::createInferRuntime(logger));
        TRTUniquePtr<nvinfer1::ICudaEngine>       eng(rt->deserializeCudaEngine(data.data(), data.size()));
        TRTUniquePtr<nvinfer1::IExecutionContext> ctx(eng->createExecutionContext());

        void *d_in, *d_out;
        cudaMalloc(&d_in,  n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaStream_t s; cudaStreamCreate(&s);

        std::vector<float> h_in(n, 1.0f), h_out(n);
        ctx->setInputShape("input", nvinfer1::Dims3{B, S, H});
        ctx->setTensorAddress("input",  d_in);
        ctx->setTensorAddress("output", d_out);
        cudaMemcpyAsync(d_in, h_in.data(), n*sizeof(float), cudaMemcpyHostToDevice, s);
        ctx->enqueueV3(s);
        cudaMemcpyAsync(h_out.data(), d_out, n*sizeof(float), cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);

        float sum = 0.f;
        for (float v : h_out) sum += v;
        cudaFree(d_in); cudaFree(d_out); cudaStreamDestroy(s);
        return sum;
    };

    float fp16_sum = runEngine(fp16_path);
    float int8_sum = runEngine(int8_path);
    float diff_pct = std::abs(int8_sum - fp16_sum) / (std::abs(fp16_sum) + 1e-6f) * 100.f;

    std::cout << "\n── Accuracy comparison ──────────────────────────\n";
    std::cout << "FP16 output sum : " << fp16_sum  << "\n";
    std::cout << "INT8 output sum : " << int8_sum  << "\n";
    std::cout << "Relative diff   : " << diff_pct  << "%\n";
    if (diff_pct < 1.0f)
        std::cout << "✓  INT8 matches FP16 within 1% — calibration OK\n";
    else
        std::cout << "⚠  >1% diff — consider more/better calibration data\n";
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    Logger logger(nvinfer1::ILogger::Severity::kWARNING);

    Int8EntropyCalibrator calibrator("llama_attn_calib.cache");

    auto t0 = std::chrono::steady_clock::now();
    auto engine_data = buildInt8Engine("llama_attn_dynamic.onnx", logger, calibrator);
    auto t1 = std::chrono::steady_clock::now();

    std::ofstream f("llama_attn_int8.trt", std::ios::binary);
    f.write(engine_data.data(), engine_data.size());
    f.close();

    std::cout << "\nEngine size : "
              << engine_data.size() / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Build time  : "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    std::cout << "Saved       : llama_attn_int8.trt\n";

    // Compare against the FP16 engine built on Day 25
    compareAccuracy(logger, "llama_attn_fp16.trt", "llama_attn_int8.trt");

    return 0;
}
