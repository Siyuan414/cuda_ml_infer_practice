"""Quick sanity check that PyTorch sees the GPU and gptqmodel imports cleanly.

Run inside the activated env:
    python verify_gpu.py
"""
import platform


def main() -> None:
    import torch

    print(f"python              : {platform.python_version()}")
    print(f"torch               : {torch.__version__}")
    print(f"torch built w/ cuda : {torch.version.cuda}")
    print(f"cuda available      : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA not available - check NVIDIA driver in Windows and `nvidia-smi` in WSL.")
        return

    n = torch.cuda.device_count()
    print(f"device count        : {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        free, total = torch.cuda.mem_get_info(i)
        print(
            f"  [{i}] {name}  sm_{major}{minor}  "
            f"{free/1e9:.1f}GB free / {total/1e9:.1f}GB"
        )

    print(f"torch arch list     : {torch.cuda.get_arch_list()}")

    # Tiny matmul to make sure kernels actually launch on this arch.
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    print(f"sample matmul ok    : out shape={tuple(c.shape)} dtype={c.dtype}")

    try:
        import gptqmodel

        print(f"gptqmodel           : {gptqmodel.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"gptqmodel import FAILED: {e}")


if __name__ == "__main__":
    main()
