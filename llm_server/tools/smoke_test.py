"""
smoke_test.py — S1.1: Verify both engine profiles execute and produce finite outputs.

Tests:
  1. PREFILL (profile 0): seq=128, past=0  → hidden [1,128,2048] finite
  2. DECODE  (profile 1): seq=1,  past=512 → hidden [1,1,2048]  finite

Run from llm_server/:  python tools/smoke_test.py
"""

from pathlib import Path

import numpy as np
import tensorrt as trt
try:
    from cuda.bindings import runtime as cudart  # cuda-python >= 12.6/13.x
except ImportError:
    from cuda import cudart  # older cuda-python

NUM_LAYERS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 64
HIDDEN = 2048

ENGINE = Path(__file__).parent.parent / "engine" / "llama1b_fp16.trt"


def check(err):
    e = err[0] if isinstance(err, tuple) else err
    if e != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {e}")
    return err[1:] if isinstance(err, tuple) and len(err) > 1 else None


def alloc(nbytes):
    (ptr,) = check(cudart.cudaMalloc(nbytes))
    return ptr


def run_phase(engine, profile, seq, past):
    print(f"\n── profile {profile}: seq={seq} past={past} ──")
    ctx = engine.create_execution_context()
    (stream,) = check(cudart.cudaStreamCreate())
    ctx.set_optimization_profile_async(profile, stream)
    check(cudart.cudaStreamSynchronize(stream))

    total = past + seq
    buffers = {}

    def upload(name, arr):
        arr = np.ascontiguousarray(arr)
        # cudaMalloc(0) returns nullptr and TRT rejects null addresses —
        # always allocate at least 2 bytes (dummy for past_len=0 KV inputs)
        ptr = alloc(max(arr.nbytes, 2))
        if arr.nbytes:
            check(cudart.cudaMemcpy(ptr, arr.ctypes.data, arr.nbytes,
                                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        buffers[name] = ptr
        ctx.set_input_shape(name, arr.shape)
        ctx.set_tensor_address(name, ptr)

    upload("input_ids", np.random.randint(0, 128000, (1, seq), dtype=np.int64))
    upload("position_ids", np.arange(past, past + seq, dtype=np.int64)[None, :])
    upload("attention_mask", np.ones((1, total), dtype=np.int64))
    kv = np.zeros((1, NUM_KV_HEADS, past, HEAD_DIM), dtype=np.float16)
    for i in range(NUM_LAYERS):
        upload(f"past_key_values.{i}.key", kv)
        upload(f"past_key_values.{i}.value", kv)

    # Outputs
    out_ptrs = {}
    hidden_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        shape = tuple(ctx.get_tensor_shape(name))
        nbytes = int(np.prod(shape)) * 2  # fp16
        ptr = alloc(max(nbytes, 2))
        out_ptrs[name] = (ptr, shape)
        ctx.set_tensor_address(name, ptr)
        if "present" not in name:
            hidden_name = name

    assert ctx.execute_async_v3(stream), "enqueue failed"
    check(cudart.cudaStreamSynchronize(stream))

    ptr, shape = out_ptrs[hidden_name]
    host = np.empty(shape, dtype=np.float16)
    check(cudart.cudaMemcpy(host.ctypes.data, ptr, host.nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    ok = np.isfinite(host.astype(np.float32)).all()
    print(f"  hidden '{hidden_name}' shape={shape}  finite={ok}  "
          f"mean|x|={np.abs(host.astype(np.float32)).mean():.4f}")

    for ptr in list(buffers.values()) + [p for p, _ in out_ptrs.values()]:
        cudart.cudaFree(ptr)
    cudart.cudaStreamDestroy(stream)
    if not ok:
        raise SystemExit("  NaN/Inf in output — FAIL")


def main():
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(ENGINE.read_bytes())
    print(f"Engine: {ENGINE.name}  profiles={engine.num_optimization_profiles}")

    run_phase(engine, profile=0, seq=128, past=0)   # prefill
    run_phase(engine, profile=1, seq=1, past=512)   # decode

    print("\nSMOKE TEST PASSED — both profiles execute.")


if __name__ == "__main__":
    main()
