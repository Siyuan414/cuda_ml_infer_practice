cmake_minimum_required(VERSION 3.18)
project(trt_resnet18 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force g++ for C++
set(CMAKE_CXX_COMPILER g++)

# Locate nvcc for custom kernel compilation
find_program(NVCC_EXECUTABLE nvcc HINTS /usr/local/cuda/bin)
if(NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found. Add /usr/local/cuda/bin to PATH.")
endif()
message(STATUS "nvcc: ${NVCC_EXECUTABLE}")

# ── Locate CUDA ────────────────────────────────────────────────────────────────
# Checks the most common install locations; set CUDA_ROOT in your env to override
if(DEFINED ENV{CUDA_ROOT})
    set(CUDA_ROOT $ENV{CUDA_ROOT})
else()
    foreach(candidate /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-11 /usr/cuda)
        if(EXISTS ${candidate})
            set(CUDA_ROOT ${candidate})
            break()
        endif()
    endforeach()
endif()

if(NOT CUDA_ROOT)
    message(FATAL_ERROR
        "CUDA not found. Set CUDA_ROOT in your environment:\n"
        "  export CUDA_ROOT=/usr/local/cuda")
endif()
message(STATUS "CUDA root: ${CUDA_ROOT}")

# ── Locate TensorRT ────────────────────────────────────────────────────────────
# Checks common TRT install locations; set TRT_ROOT in your env to override
if(DEFINED ENV{TRT_ROOT})
    set(TRT_ROOT $ENV{TRT_ROOT})
else()
    foreach(candidate
            /usr/local/tensorrt
            /opt/tensorrt
            /usr/local/TensorRT
            /opt/TensorRT
            /usr)
        if(EXISTS ${candidate}/include/NvInfer.h)
            set(TRT_ROOT ${candidate})
            break()
        endif()
    endforeach()
endif()

if(NOT TRT_ROOT)
    message(FATAL_ERROR
        "TensorRT not found. Set TRT_ROOT in your environment:\n"
        "  export TRT_ROOT=/usr/local/tensorrt")
endif()
message(STATUS "TensorRT root: ${TRT_ROOT}")

# ── Include paths ──────────────────────────────────────────────────────────────
include_directories(
    ${CUDA_ROOT}/include
    ${TRT_ROOT}/include
)

# ── Library paths ──────────────────────────────────────────────────────────────
link_directories(
    ${CUDA_ROOT}/lib64
    ${TRT_ROOT}/lib
)

# ── Compiler flags ─────────────────────────────────────────────────────────────
# -Wno-deprecated: TRT headers use some deprecated constructs internally
add_compile_options(-O2 -Wall -Wno-deprecated)

# ── Targets ────────────────────────────────────────────────────────────────────
add_executable(resnet_trt resnet_trt.cpp)
target_link_libraries(resnet_trt
    nvinfer          # core TRT
    nvonnxparser     # ONNX frontend
    cudart           # CUDA runtime (needed for cudaMalloc etc.)
)

# ── RoPE plugin library ───────────────────────────────────────────────────────
# Compile rope_kernel.cu directly with nvcc (bypasses CMake CUDA machinery
# which triggers a __cudaLaunch stub bug in CUDA 12.8)
set(ROPE_KERNEL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/rope_kernel.o)
add_custom_command(
    OUTPUT  ${ROPE_KERNEL_OBJ}
    COMMAND ${NVCC_EXECUTABLE}
            -arch=sm_89
            -I${CUDA_ROOT}/include
            -I${TRT_ROOT}/include
            -std=c++17
            -c ${CMAKE_CURRENT_SOURCE_DIR}/rope_kernel.cu
            -o ${ROPE_KERNEL_OBJ}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/rope_kernel.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/rope_kernel.h
    COMMENT "Compiling rope_kernel.cu with nvcc (direct)"
)
add_custom_target(rope_kernel_obj DEPENDS ${ROPE_KERNEL_OBJ})

# rope_plugin.cpp is plain C++, compiled by g++
add_library(rope_plugin STATIC rope_plugin.cpp ${ROPE_KERNEL_OBJ})
add_dependencies(rope_plugin rope_kernel_obj)
target_include_directories(rope_plugin PUBLIC ${CUDA_ROOT}/include ${TRT_ROOT}/include)
target_link_libraries(rope_plugin nvinfer cudart)

# ── RoPE kernel test binary ───────────────────────────────────────────────────
add_executable(rope_test rope_test.cpp)
add_dependencies(rope_test rope_kernel_obj)
target_include_directories(rope_test PRIVATE ${CUDA_ROOT}/include ${TRT_ROOT}/include)
target_link_libraries(rope_test rope_plugin cudart)

# ── ONNX protobuf parser ──────────────────────────────────────────────────────
add_executable(onnx_parser onnx_parser.cpp onnx_proto/onnx.pb.cc)
target_link_libraries(onnx_parser protobuf)
target_include_directories(onnx_parser PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ── Graph optimizer (constant folding + dead node elimination) ────────────────
add_executable(graph_optimizer graph_optimizer.cpp onnx_proto/onnx.pb.cc)
target_link_libraries(graph_optimizer protobuf)
target_include_directories(graph_optimizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ── LLaMA attention TRT builder ───────────────────────────────────────────────
add_executable(llama_trt_builder llama_trt_builder.cpp)
target_link_libraries(llama_trt_builder nvinfer nvonnxparser cudart)

# ── LLaMA attention TRT inferencer ───────────────────────────────────────────
add_executable(llama_trt_infer llama_trt_infer.cpp)
target_link_libraries(llama_trt_infer nvinfer cudart)

# ── Build info ─────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "Build configuration:")
message(STATUS "  Compiler : ${CMAKE_CXX_COMPILER}")
message(STATUS "  Standard : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA     : ${CUDA_ROOT}")
message(STATUS "  TensorRT : ${TRT_ROOT}")
message(STATUS "")
message(STATUS "To build:")
message(STATUS "  mkdir -p build && cd build")
message(STATUS "  cmake .. && make -j\$(nproc)")
