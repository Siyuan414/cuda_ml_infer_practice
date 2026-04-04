#swish(x) = x* sig(x)
import triton 
import triton.language as tl 
import torch
import torch.utils.benchmark as benchmark

@triton.jit
def swish_kernel(x_ptr,out_ptr,n_elements,BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr+offsets,mask=mask)
    sigmoid = 1 / (1+tl.exp(-x))
    out = x * sigmoid
    tl.store(out_ptr + offsets, out,mask=mask)

def swish_torch(x):
    return x * torch.sigmoid(x)


def swish_triton(x):
    assert x.is_cuda

    n_elements = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out

def bandwidth(bytes_moved, time_s):
    return bytes_moved / time_s / 1e9  # GB/s

#test benchmark
x = torch.randn(1_000_000, device='cuda')

y_ref = swish_torch(x)
y_triton = swish_triton(x)

print((y_ref - y_triton).abs().max())
assert torch.allclose(y_ref, y_triton, atol=1e-5)

t0 = benchmark.Timer(
    stmt='swish_torch(x)',
    globals={'x': x, 'swish_torch': swish_torch}
)

t1 = benchmark.Timer(
    stmt='swish_triton(x)',
    globals={'x': x, 'swish_triton': swish_triton}
)

print(t0.timeit(100))
print(t1.timeit(100))

N = x.numel()
bytes_per_elem = 4

# unfused
bytes_unfused = 5 * N * bytes_per_elem

# fused
bytes_fused = 2 * N * bytes_per_elem
res0 = t0.timeit(100)
res1 = t1.timeit(100)

bw_unfused = bandwidth(bytes_unfused, res0.mean)
bw_fused = bandwidth(bytes_fused, res1.mean)

print("Unfused BW:", bw_unfused)
print("Fused BW:", bw_fused)