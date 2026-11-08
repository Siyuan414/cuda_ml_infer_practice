import torch
import numpy as np


#define tensor dimensions
batch_size = 1
num_heads = 2
seq_len = 8 
head_dim = 64

torch.manual_seed(42)  # For reproducibility
x = torch.randn(batch_size, num_heads, seq_len, head_dim)
print(f"Input shape: {x.shape}")

def ref_rope(x):
    batch_size, num_heads, seq_len, head_dim = x.shape
    out = torch.zeros_like(x)
    for pos in range(seq_len):
        for i in range(head_dim):
            theta = 10000 ** (-2 * (i // 2) / head_dim)
            if i % 2 == 0:
                out[:, :, pos, i] = x[:, :, pos, i] * torch.cos(torch.tensor(pos * theta)) - x[:, :, pos, i + 1] * torch.sin(torch.tensor(pos * theta))
            else:
                out[:, :, pos, i] = x[:, :, pos, i] * torch.cos(torch.tensor(pos * theta)) + x[:, :, pos, i - 1] * torch.sin(torch.tensor(pos * theta))

    return out

output = ref_rope(x)
print(f"Output shape: {output.shape}")

# Save input and output as raw float32 binary
x.numpy().astype(np.float32).tofile("rope_input.bin")
output.numpy().astype(np.float32).tofile("rope_output_ref.bin")

# Print a few values for manual verification in C++
print(f"input[0,0,1,:4]  = {x[0,0,1,:4].tolist()}")
print(f"output[0,0,1,:4] = {output[0,0,1,:4].tolist()}")
print(f"Total floats — input: {x.numel()}, output: {output.numel()}")
x.numpy().astype(np.float32).tofile("rope_input.bin")
output.numpy().astype(np.float32).tofile("rope_output_ref.bin")
print("Saved rope_input.bin and rope_output_ref.bin")


