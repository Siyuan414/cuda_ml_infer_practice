cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_COMPILER g++)
project(llama_attn_trt LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/TrtCuda.cmake)

# FP16 dynamic-shape engine: llama_attn_dynamic.onnx -> llama_attn_fp16.trt
add_executable(llama_trt_builder llama_trt_builder.cpp)
target_link_libraries(llama_trt_builder nvinfer nvonnxparser cudart)

# Deserialize + run with NVTX ranges and two-stream ping-pong
# (NVTX3 in CUDA 12+ is header-only, no link needed)
add_executable(llama_trt_infer llama_trt_infer.cpp)
target_link_libraries(llama_trt_infer nvinfer cudart)

# INT8 engine with IInt8EntropyCalibrator2, compared against the FP16 engine
add_executable(llama_int8_builder llama_int8_builder.cpp)
target_link_libraries(llama_int8_builder nvinfer nvonnxparser cudart)
