#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <chrono>

#include <NvInfer.h>
#include <NvOnnxParser.h>      // was: OnnxParser.h  (wrong name)
#include <cuda_runtime_api.h>

// ── RAII wrapper ───────────────────────────────────────────────────────────────
struct TRTDestroy {
    template<typename T>
    void operator()(T* obj) const {
        delete obj;  // destroy() was removed in TRT 9 — use delete
    }
};

template<typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

// ── Logger ─────────────────────────────────────────────────────────────────────
// Bug fixes: class name was lowercase 'logger', base class was 'Ilogger' (wrong case)
// Missing semicolon after closing brace
class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) noexcept override {
        if (severity > reportableSeverity) return;
        switch (severity) {
            case Severity::kINTERNAL_ERROR: std::cerr << "[TRT FATAL] "; break;
            case Severity::kERROR:          std::cerr << "[TRT ERROR] "; break;
            case Severity::kWARNING:        std::cerr << "[TRT WARN ] "; break;
            case Severity::kINFO:           std::cerr << "[TRT INFO ] "; break;
            default:                        std::cerr << "[TRT VERB ] "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};  // <-- was missing semicolon

// ── dtype size helper ──────────────────────────────────────────────────────────
// Was missing entirely — called in allocateBuffers()
size_t dtypeSize(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        default: return 4;
    }
}

// ── Build engine from ONNX ─────────────────────────────────────────────────────
std::vector<char> buildEngine(
    const std::string& onnx_path,
    Logger& logger,
    bool enable_fp16 = true,
    size_t workspace_mb = 1024)
{
    // 1. Builder
    // Bug: was 'Ibuilder' (wrong case)
    TRTUniquePtr<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(logger));
    if (!builder)
        throw std::runtime_error("Failed to create TensorRT builder");

    // 2. Network — kEXPLICIT_BATCH required for dynamic shapes
    const uint32_t flags = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(flags));

    // 3. ONNX parser
    TRTUniquePtr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, logger));

    // Bug: second arg was 'logger' object — must be int severity level
    if (!parser->parseFromFile(onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i)
            std::cerr << parser->getError(i)->desc() << std::endl;
        throw std::runtime_error("Failed to parse ONNX file");
    }
    std::cout << "Parsed ONNX: " << network->getNbLayers() << " layers\n";

    // 4. Builder config
    // Bug: setMemeoryPoolLimit() — misspelled, and broken syntax mixing setFlag into it
    TRTUniquePtr<nvinfer1::IBuilderConfig> config(
        builder->createBuilderConfig());
    config->setMemoryPoolLimit(
        nvinfer1::MemoryPoolType::kWORKSPACE,
        workspace_mb * 1024ULL * 1024ULL);

    if (enable_fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 mode enabled\n";
    }

    // 5. Dynamic shape optimization profile
    // Bug: 'profile' was used but never declared
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input",
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1,  3, 224, 224});
    profile->setDimensions("input",
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{8,  3, 224, 224});
    profile->setDimensions("input",
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{16, 3, 224, 224});
    config->addOptimizationProfile(profile);

    // 6. Build and serialize
    // Bug: was 'TRTUniqueptr' (lowercase p)
    TRTUniquePtr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized)
        throw std::runtime_error("Failed to build engine");

    // Bug: 'reutrn' typo, and missing closing paren on cast
    return std::vector<char>(
        static_cast<const char*>(serialized->data()),
        static_cast<const char*>(serialized->data()) + serialized->size());
}

// ── Save engine to disk ────────────────────────────────────────────────────────
void saveEngine(const std::vector<char>& engine_data, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open engine file for writing");
    f.write(engine_data.data(), engine_data.size());
    std::cout << "Engine saved to " << path << "\n";
}

// ── Inference engine wrapper ───────────────────────────────────────────────────
class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& engine_path, Logger& logger) {
        // Bug: 'data' vector was used but never declared
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Cannot open engine file: " + engine_path);
        size_t size = file.tellg();
        file.seekg(0);
        std::vector<char> data(size);          // <-- was missing
        file.read(data.data(), size);

        TRTUniquePtr<nvinfer1::IRuntime> runtime(
            nvinfer1::createInferRuntime(logger));

        // Bug: old API had a 3rd nullptr arg — current API takes just (data, size)
        engine_.reset(runtime->deserializeCudaEngine(data.data(), size));
        if (!engine_) throw std::runtime_error("Failed to deserialize engine");
        
        for(int i = 0; i < 2; ++i)
            context_[i].reset(engine_->createExecutionContext());
        if (!context_[0]) throw std::runtime_error("Failed to create execution context0");
        if (!context_[1]) throw std::runtime_error("Failed to create execution context1");

        // Inspect bindings
        int num_io = engine_->getNbIOTensors();
        std::cout << "Engine has " << num_io << " I/O tensors:\n";
        for (int i = 0; i < num_io; ++i) {
            const char* name = engine_->getIOTensorName(i);
            auto dims   = engine_->getTensorShape(name);
            auto dtype  = engine_->getTensorDataType(name);
            bool is_in  = engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;

            std::cout << (is_in ? "  INPUT " : "  OUTPUT")
                      << " [" << name << "] dtype=" << static_cast<int>(dtype) << " dims=(";
            for (int d = 0; d < dims.nbDims; ++d)
                std::cout << dims.d[d] << (d + 1 < dims.nbDims ? "," : "");
            std::cout << ")\n";
        }

        for (int i = 0; i < 2; ++i)
            cudaStreamCreate(&stream_[i]);

        allocateBuffers(0);
        allocateBuffers(1);
        
    }

    ~InferenceEngine() {
        for (auto& [name, ptr] : buffers_[0])
            cudaFree(ptr);
        for (auto& [name, ptr] : buffers_[1])
            cudaFree(ptr);
    }
    void sync(int slot) {
        cudaStreamSynchronize(stream_[slot]);
    }

    void infer(const float* h_input, float* h_output, int batch_size,int slot) {
        // This function would enqueue the inference work on the GPU using the context
        // and the allocated buffers. For simplicity, it's left as a placeholder.
        // Set input shape for dynamic batch size
        context_[slot]->setInputShape("input", nvinfer1::Dims4{batch_size, 3, 224, 224});
        //re register buffer address after setting shape
        for(auto& [name, ptr] : buffers_[slot])
            context_[slot]->setTensorAddress(name.c_str(), ptr);
        
        //compute byte size for this batch 
        size_t input_bytes = batch_size * 3 * 224 * 224 * sizeof(float);
        size_t output_bytes = batch_size * 1000 * sizeof(float); // assuming output

        //H2D-async 
        cudaMemcpyAsync(buffers_[slot]["input"], h_input, input_bytes, cudaMemcpyHostToDevice, stream_[slot]);

        //Enqueue inference
        bool ok = context_[slot]->enqueueV3(stream_[slot]);
        if (!ok) {
            std::cerr << "Failed to enqueue inference\n";
            return;
        }
        //D2H-async — do NOT sync here, caller controls when to wait
        cudaMemcpyAsync(h_output, buffers_[slot]["output"], output_bytes, cudaMemcpyDeviceToHost, stream_[slot]);
        
    }

private:
    void allocateBuffers(int slot) {
        
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            auto dims  = engine_->getTensorShape(name);
            auto dtype = engine_->getTensorDataType(name);

            size_t nelems = 1;
            for (int d = 0; d < dims.nbDims; ++d)
                nelems *= (dims.d[d] < 0 ? static_cast<int64_t>(max_batch_) : dims.d[d]);

            size_t bytes = nelems * dtypeSize(dtype);

            void* ptr = nullptr;
            cudaMalloc(&ptr, bytes);
            buffers_[slot][name] = ptr;

            std::cout << "Allocated " << bytes / 1024 << " KB for '" << name << "'\n";
        }
    }

    TRTUniquePtr<nvinfer1::ICudaEngine>      engine_;
    TRTUniquePtr<nvinfer1::IExecutionContext> context_[2];
    std::unordered_map<std::string, void*>   buffers_[2];
    int max_batch_ = 16;
    cudaStream_t stream_[2];
};  // <-- was missing semicolon


std::vector<float> loadBin(const std::string& path, size_t num_floats) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<float> data(num_floats);
    f.read(reinterpret_cast<char*>(data.data()), num_floats * sizeof(float));
    return data;
}

float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Size mismatch for maxAbsDiff\n";
        return 0.0f;
    }
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Max absolute difference: " << max_diff << "\n";
    return max_diff;
}

double meanAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Size mismatch for meanAbsDiff\n";
        return 0.0f;
    }
    double sum_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum_diff += std::abs(a[i] - b[i]);
    }
    double mean_diff = sum_diff / a.size();
    std::cout << "Mean absolute difference: " << mean_diff << "\n";
    return static_cast<double>(mean_diff);
}

int argmax(const std::vector<float>& data) {
    int max_idx = 0;
    float max_val = data[0];
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = static_cast<int>(i);
        }
    }
    return max_idx;
}
// ── main ───────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Logger logger(nvinfer1::ILogger::Severity::kINFO);

    const std::string onnx_path   = (argc > 1) ? argv[1] : "resnet18.onnx";
    const std::string engine_path = (argc > 2) ? argv[2] : "resnet18.engine";

    std::cout << "=== Building engine ===\n";
    auto engine_data = buildEngine(onnx_path, logger);
    saveEngine(engine_data, engine_path);

    std::cout << "\n=== Loading and inspecting engine ===\n";
    InferenceEngine engine(engine_path, logger);

    // Real host buffers — pinned memory for faster H2D transfers
    const int batch_size  = 8;
    const int num_batches = 10;
    const size_t input_elems  = batch_size * 3 * 224 * 224;
    const size_t output_elems = batch_size * 1000;

    std::vector<float> h_input(input_elems, 0.5f);   // dummy input
    std::vector<float> h_output(output_elems, 0.0f);
    // single-stream baseline
    auto t_single_0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_batches; ++i) {
        engine.sync(0);
        engine.infer(h_input.data(), h_output.data(), batch_size, 0);
    }
    engine.sync(0);
    auto t_single_1 = std::chrono::high_resolution_clock::now();
    double ms_single = std::chrono::duration<double, std::milli>(t_single_1 - t_single_0).count();
    std::cout << "Single-stream: " << ms_single / num_batches << " ms/batch\n";
    
    std::cout << "\n=== Running double-buffered inference ===\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_batches; ++i) {
        int slot = i % 2;
        engine.sync(slot);   // wait for this slot's previous work before reusing it
        engine.infer(h_input.data(), h_output.data(), batch_size, slot);
    }
    // drain both streams
    engine.sync(0);
    engine.sync(1);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << num_batches << " batches in " << ms << " ms"
              << "  (" << ms / num_batches << " ms/batch)\n";
   

    std::cout << "\n=== Validating against PyTorch reference ===\n";

    // 1. Load input and reference output from disk
    auto h_val_input = loadBin("resnet18_input.bin", 1 * 3 * 224 * 224);  // batch_size=1, C=3, H=224, W=224
    auto h_ref        = loadBin("resnet18_output.bin", 1 * 1000);     // batch_size=1, num_classes=1000

    // 2. Allocate output buffer for TRT result
    std::vector<float> h_trt_out(1 * 1000);  // batch_size=1, num_classes=1000

    // 3. Run inference on slot 0 with batch_size = 1
    engine.infer(h_val_input.data(), h_trt_out.data(), 1, 0);
    engine.sync(0);

    // 4. Compare
    maxAbsDiff(h_trt_out, h_ref);
    meanAbsDiff(h_trt_out, h_ref);
    std::cout << "PyTorch argmax : " << argmax(h_ref)     << "\n";
    std::cout << "TRT     argmax : " << argmax(h_trt_out) << "\n";


    std::cout << "\nAll Day 21 tasks done.\n";
    return 0;
}
