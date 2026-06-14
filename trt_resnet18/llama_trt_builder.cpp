#include "trt_common.h"
#include <NvOnnxParser.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <cuda_runtime_api.h>

// ── Build engine from ONNX ────────────────────────────────────────────────────
std::vector<char> buildEngine(
    const std::string& onnx_path,
    Logger& logger,
    bool enable_fp16  = true,
    size_t workspace_mb = 1024)
{
    // 1. Builder
    TRTUniquePtr<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("Failed to create builder");

    // 2. Network
    const uint32_t flags = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(flags));
    if (!network) throw std::runtime_error("Failed to create network");

    // 3. ONNX parser
    TRTUniquePtr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i)
            std::cerr << parser->getError(i)->desc() << "\n";
        throw std::runtime_error("Failed to parse ONNX");
    }
    std::cout << "Parsed ONNX: " << network->getNbLayers() << " layers\n";

    // 4. Build config
    TRTUniquePtr<nvinfer1::IBuilderConfig> config(
        builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               workspace_mb * 1024ULL * 1024ULL);
    if (enable_fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 mode enabled\n";
    }

    // 5. Optimization profile — [batch, seq_len, hidden=256]
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims3{1,   1, 256});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims3{1, 128, 256});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims3{4, 512, 256});
    config->addOptimizationProfile(profile);

    // 6. Build and serialize
    TRTUniquePtr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) throw std::runtime_error("Failed to build engine");

    return std::vector<char>(
        static_cast<const char*>(serialized->data()),
        static_cast<const char*>(serialized->data()) + serialized->size());
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    Logger logger(nvinfer1::ILogger::Severity::kINFO);

    auto t0 = std::chrono::steady_clock::now();
    auto engine_data = buildEngine("llama_attn_dynamic.onnx", logger);
    auto t1 = std::chrono::steady_clock::now();

    std::ofstream f("llama_attn_fp16.trt", std::ios::binary);
    f.write(engine_data.data(), engine_data.size());
    f.close();

    std::cout << "Engine size : "
              << engine_data.size() / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Build time  : "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    std::cout << "Saved       : llama_attn_fp16.trt\n";
    return 0;
}
