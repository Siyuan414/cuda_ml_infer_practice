cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_COMPILER g++)
project(onnx_graph_optimizer LANGUAGES CXX)

# Pure C++ + protobuf; no CUDA or TensorRT needed.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O2 -Wall)

# Prints the nodes, inputs and outputs of an ONNX model
add_executable(onnx_parser onnx_parser.cpp onnx_proto/onnx.pb.cc)
target_link_libraries(onnx_parser protobuf)
target_include_directories(onnx_parser PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Constant folding + dead-node elimination (reverse DFS from graph outputs)
add_executable(graph_optimizer graph_optimizer.cpp onnx_proto/onnx.pb.cc)
target_link_libraries(graph_optimizer protobuf)
target_include_directories(graph_optimizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
