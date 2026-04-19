import torch, torch.nn.functional as F
q = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
    F.scaled_dot_product_attention(q, q, q)
print("flash backend works on this build")