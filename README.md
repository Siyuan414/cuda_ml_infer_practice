# TensorRT practice projects

Four standalone projects. Each has its own `CMakeLists.txt` and shares the CUDA/TensorRT lookup in `cmake/TrtCuda.cmake`.

| Folder | What it does |
|---|---|
| `resnet18_trt/` | Warm-up project. ResNet-18 ONNX → TensorRT engine, engine serialization, and an output check against PyTorch. |
| `rope_trt_plugin/` | RoPE as a custom TensorRT plugin: CUDA kernel, `IPluginV2DynamicExt` class registered with `REGISTER_TENSORRT_PLUGIN`, a kernel test against a PyTorch reference, and an onnx-graphsurgeon script that swaps the RoPE subgraph for the plugin node. |
| `onnx_graph_optimizer/` | C++ ONNX tools built on protobuf. `onnx_parser` prints a graph. `graph_optimizer` finds constant-foldable nodes and removes dead nodes with a reverse DFS from the graph outputs. |
| `llama_attn_trt/` | LLaMA attention block: FP16 dynamic-shape engine builder, inference runner with NVTX ranges and two-stream ping-pong, and an INT8 calibrated build. Nsight Systems profiles are in `profiles/`. |

## Build (Linux / WSL)

```bash
cd <project>
mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

Run the binaries from the project folder (e.g. `./build/llama_trt_infer`), since they load model files by relative path.

Set `CUDA_ROOT` / `TRT_ROOT` if CUDA or TensorRT is not in a standard location.

## Generated files

`.onnx`, `.trt`, `.engine`, `.bin`, and profile files are git-ignored. Regenerate them with each project's Python scripts. The shared Python env is `venv/` (`source venv/bin/activate`).

- `resnet18_trt/`: `export_onnx.py`, `export_reference.py`
- `rope_trt_plugin/`: `rope_reference.py`, `rope_surgeon.py`
- `llama_attn_trt/`: `llama_attn.py` writes `llama_attn_dynamic.onnx`. Its commented-out block writes `llama_attn_dead.onnx`, the test input for `onnx_graph_optimizer/`.
