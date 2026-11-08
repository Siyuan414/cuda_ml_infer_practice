import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LlamaAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Dead branch: a "router" linear that feeds nothing used in output
        self.unused_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)   # [B,H,T,D]
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Dead branch — computed but never used in output
        _ = self.unused_proj(x)

        attn = (q @ k.transpose(-2, -1)) * self.scale          # [B,H,T,T]
        attn = F.softmax(attn, dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

model = LlamaAttention(hidden_dim=256, num_heads=4).eval()
x = torch.randn(1, 16, 256)  # batch=1, seq=16, hidden=256

torch.onnx.export(
    model, x,
   "llama_attn_dynamic.onnx",
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch", 1: "seq_len"},
        "output": {0: "batch", 1: "seq_len"},
    },
    do_constant_folding=False,   # keep all nodes so our optimizer sees them
)
print("Exported llama_attn_dynamic.onnx")

# import onnx
# from onnx import numpy_helper, TensorProto

# model = onnx.load("llama_attn.onnx")
# graph = model.graph

# # Inject a dead Relu node: reads from a real tensor, writes to nowhere
# dead_node = onnx.helper.make_node(
#     "Relu",
#     inputs=["input"],        # reads the real graph input
#     outputs=["dead_relu_out"],  # output never consumed by anyone
#     name="dead_relu"
# )
# graph.node.append(dead_node)

# # Inject a second dead node that reads from the first dead node
# dead_node2 = onnx.helper.make_node(
#     "Relu",
#     inputs=["dead_relu_out"],
#     outputs=["dead_relu_out2"],
#     name="dead_relu2"
# )
# graph.node.append(dead_node2)

# onnx.save(model, "llama_attn_dead.onnx")
# print("Saved llama_attn_dead.onnx (injected 2 dead nodes)")