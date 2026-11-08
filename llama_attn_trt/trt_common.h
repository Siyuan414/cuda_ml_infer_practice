#pragma once
#include <NvInfer.h>
#include <memory>
#include <iostream>

// ── RAII deleter ─────────────────────────────────────────────────────────────
// TRT 9+ removed destroy(); use delete instead.
struct TRTDestroy {
    template<typename T>
    void operator()(T* obj) const { delete obj; }
};

template<typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

// ── Logger ───────────────────────────────────────────────────────────────────
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
        std::cerr << msg << "\n";
    }

    Severity reportableSeverity;
};

// ── dtype byte-size helper ────────────────────────────────────────────────────
inline size_t dtypeSize(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        default:                         return 4;
    }
}
