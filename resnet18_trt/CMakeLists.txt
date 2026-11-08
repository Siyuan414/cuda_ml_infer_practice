cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_COMPILER g++)
project(resnet18_trt LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/TrtCuda.cmake)

# ONNX -> TRT engine build, serialize, and accuracy check vs PyTorch reference
add_executable(resnet_trt resnet_trt.cpp)
target_link_libraries(resnet_trt nvinfer nvonnxparser cudart)
