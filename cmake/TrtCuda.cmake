# Shared CUDA + TensorRT setup for every project in Tensorrt/.
# Usage (from a project's CMakeLists.txt):
#   include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/TrtCuda.cmake)
#
# Override the search with env vars:  export CUDA_ROOT=...  TRT_ROOT=...

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── CUDA ──────────────────────────────────────────────────────────────────────
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
    message(FATAL_ERROR "CUDA not found. Set CUDA_ROOT, e.g. export CUDA_ROOT=/usr/local/cuda")
endif()

# ── TensorRT ──────────────────────────────────────────────────────────────────
if(DEFINED ENV{TRT_ROOT})
    set(TRT_INCLUDE_DIR $ENV{TRT_ROOT}/include)
    set(TRT_LIB_DIR     $ENV{TRT_ROOT}/lib)
elseif(EXISTS /usr/include/x86_64-linux-gnu/NvInfer.h)
    # apt package layout (Ubuntu / WSL)
    set(TRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
    set(TRT_LIB_DIR     /usr/lib/x86_64-linux-gnu)
else()
    foreach(candidate /usr/local/tensorrt /opt/tensorrt /usr/local/TensorRT /opt/TensorRT /usr)
        if(EXISTS ${candidate}/include/NvInfer.h)
            set(TRT_INCLUDE_DIR ${candidate}/include)
            set(TRT_LIB_DIR     ${candidate}/lib)
            break()
        endif()
    endforeach()
endif()
if(NOT TRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT not found. Set TRT_ROOT, e.g. export TRT_ROOT=/usr/local/tensorrt")
endif()

message(STATUS "CUDA root      : ${CUDA_ROOT}")
message(STATUS "TensorRT headers: ${TRT_INCLUDE_DIR}")

include_directories(${CUDA_ROOT}/include ${TRT_INCLUDE_DIR})
link_directories(${CUDA_ROOT}/lib64 ${TRT_LIB_DIR})

# -Wno-deprecated: TRT headers use some deprecated constructs internally
add_compile_options(-O2 -Wall -Wno-deprecated)
