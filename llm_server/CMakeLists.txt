cmake_minimum_required(VERSION 3.18)
project(llm_server LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Platform detection: Jetson (aarch64) vs desktop (x86_64) ─────────────────
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(CMAKE_CUDA_ARCHITECTURES 87)          # Jetson Orin
    set(TRT_LIB_DIR /usr/lib/aarch64-linux-gnu)
    set(TRT_INC_DIR /usr/include/aarch64-linux-gnu)
else()
    set(CMAKE_CUDA_ARCHITECTURES 120)         # RTX 5070 Ti (Blackwell)
    set(TRT_LIB_DIR /usr/lib/x86_64-linux-gnu)
    set(TRT_INC_DIR /usr/include/x86_64-linux-gnu)
endif()

find_package(CUDAToolkit REQUIRED)

# ── runtime: single-request engine (Stage 1) ─────────────────────────────────
add_executable(runtime
    src/runtime.cpp
)
set_source_files_properties(src/runtime.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(runtime PRIVATE ${TRT_INC_DIR}
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/kernels)
target_link_directories(runtime PRIVATE ${TRT_LIB_DIR})
target_link_libraries(runtime
    nvinfer
    CUDA::cudart
    CUDA::cublas
)

# ── batch_runtime: continuous batching (Stage 2A) ────────────────────────────
add_executable(batch_runtime
    src/batch_runtime.cpp
)
set_source_files_properties(src/batch_runtime.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(batch_runtime PRIVATE ${TRT_INC_DIR}
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/kernels)
target_link_directories(batch_runtime PRIVATE ${TRT_LIB_DIR})
target_link_libraries(batch_runtime
    nvinfer
    CUDA::cudart
    CUDA::cublas
)

# Future targets (uncomment as stages progress):
# add_executable(engine_builder tools/engine_builder.cpp)   # S1.1
# add_executable(bench tools/bench.cpp)                     # S1.6
# add_executable(server src/server.cpp ...)                 # Stage 2
