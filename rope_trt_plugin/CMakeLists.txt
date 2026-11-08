cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_COMPILER g++)
project(rope_trt_plugin LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/TrtCuda.cmake)

find_program(NVCC_EXECUTABLE nvcc HINTS ${CUDA_ROOT}/bin /usr/local/cuda/bin)
if(NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found. Add /usr/local/cuda/bin to PATH.")
endif()

# Compile rope_kernel.cu directly with nvcc (bypasses CMake's CUDA language
# support, which hit a __cudaLaunch stub bug in CUDA 12.8)
set(ROPE_KERNEL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/rope_kernel.o)
add_custom_command(
    OUTPUT  ${ROPE_KERNEL_OBJ}
    COMMAND ${NVCC_EXECUTABLE} -arch=sm_89 -std=c++17
            -I${CUDA_ROOT}/include -I${TRT_INCLUDE_DIR}
            -c ${CMAKE_CURRENT_SOURCE_DIR}/rope_kernel.cu
            -o ${ROPE_KERNEL_OBJ}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/rope_kernel.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/rope_kernel.h
    COMMENT "Compiling rope_kernel.cu with nvcc"
)
add_custom_target(rope_kernel_obj DEPENDS ${ROPE_KERNEL_OBJ})

# IPluginV2DynamicExt plugin (registered via REGISTER_TENSORRT_PLUGIN)
add_library(rope_plugin STATIC rope_plugin.cpp ${ROPE_KERNEL_OBJ})
add_dependencies(rope_plugin rope_kernel_obj)
target_link_libraries(rope_plugin nvinfer cudart)

# Kernel correctness test against the PyTorch reference (rope_reference.py)
add_executable(rope_test rope_test.cpp)
add_dependencies(rope_test rope_kernel_obj)
target_link_libraries(rope_test rope_plugin cudart)
