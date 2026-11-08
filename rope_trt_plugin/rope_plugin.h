#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

// ── RoPE Plugin ───────────────────────────────────────────────────────────────
class RoPEPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    RoPEPlugin(int num_heads, int head_dim)
        : num_heads_(num_heads), head_dim_(head_dim) {}

    // Output count
    int getNbOutputs() const noexcept override { return 1; }

    // Output shape == input shape (RoPE is in-place rotation)
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override
    {
        return inputs[0];
    }

    // Only support FP32 linear layout
    bool supportsFormatCombination(
        int pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs, int nbOutputs) noexcept override
    {
        return inOut[pos].type   == nvinfer1::DataType::kFLOAT &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }

    // Called once after format negotiation — nothing to pre-compute here
    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in,  int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {}

    // Runtime: launch RoPE CUDA kernel (implemented in rope_plugin.cu)
    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;

    // Bookkeeping
    const char* getPluginType()    const noexcept override { return "RoPEPlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }

    size_t getSerializationSize() const noexcept override { return sizeof(int) * 2; }

    void serialize(void* buffer) const noexcept override {
        int* buf = static_cast<int*>(buffer);
        buf[0] = num_heads_;
        buf[1] = head_dim_;
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        return new RoPEPlugin(num_heads_, head_dim_);
    }

    void destroy() noexcept override { delete this; }

    // No GPU workspace needed beyond the I/O buffers
    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,  int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override
    { return 0; }

    // Output dtype matches input — always FP32
    nvinfer1::DataType getOutputDataType(
        int index,
        const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept override
    { return nvinfer1::DataType::kFLOAT; }

    // Nothing to allocate/free at engine load/unload time
    int32_t initialize() noexcept override { return 0; }
    void    terminate()  noexcept override {}

    void setPluginNamespace(const char* ns) noexcept override { namespace_ = ns; }
    const char* getPluginNamespace() const noexcept override  { return namespace_.c_str(); }

private:
    int num_heads_;
    int head_dim_;
    std::string namespace_;
};

// ── RoPE Plugin Creator ───────────────────────────────────────────────────────
class RoPEPluginCreator : public nvinfer1::IPluginCreator {
public:
    RoPEPluginCreator() {
        plugin_attributes_.emplace_back(
            nvinfer1::PluginField("num_heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        plugin_attributes_.emplace_back(
            nvinfer1::PluginField("head_dim",  nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        field_collection_.nbFields = static_cast<int>(plugin_attributes_.size());
        field_collection_.fields   = plugin_attributes_.data();
    }

    const char* getPluginName()    const noexcept override { return "RoPEPlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        return &field_collection_;
    }

    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc) noexcept override
    {
        int num_heads = 0, head_dim = 0;
        for (int i = 0; i < fc->nbFields; ++i) {
            std::string field_name(fc->fields[i].name);
            if (field_name == "num_heads")
                num_heads = *static_cast<const int*>(fc->fields[i].data);
            else if (field_name == "head_dim")
                head_dim  = *static_cast<const int*>(fc->fields[i].data);
        }
        return new RoPEPlugin(num_heads, head_dim);
    }

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData, size_t serialLength) noexcept override
    {
        const int* buf = static_cast<const int*>(serialData);
        return new RoPEPlugin(buf[0], buf[1]);
    }

    void setPluginNamespace(const char* ns) noexcept override { namespace_ = ns; }
    const char* getPluginNamespace() const noexcept override  { return namespace_.c_str(); }

private:
    std::vector<nvinfer1::PluginField> plugin_attributes_;
    nvinfer1::PluginFieldCollection    field_collection_{};
    std::string                        namespace_;
};

// ── Registration ──────────────────────────────────────────────────────────────
REGISTER_TENSORRT_PLUGIN(RoPEPluginCreator);
