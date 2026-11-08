import torch
import torch.nn as nn
import numpy as np
import onnx
import onnx_graphsurgeon as gs

class RoPEModule(nn.Module):
    def __init__(self, seq_len, head_dim):
        super(RoPEModule, self).__init__()
        position = torch.arange(seq_len).unsqueeze(1).float()
        dim_idx  = torch.arange(0, head_dim, 2).float()
        theta    = position / (10000 ** (dim_idx / head_dim))
        self.register_buffer("cos_table", torch.cos(theta))  # [seq_len, head_dim//2]
        self.register_buffer("sin_table", torch.sin(theta))  # [seq_len, head_dim//2]

    def forward(self, x):
        x0 = x[..., 0::2]  # even dims
        x1 = x[..., 1::2]  # odd dims
        cos = self.cos_table
        sin = self.sin_table
        out_even = x0 * cos - x1 * sin
        out_odd  = x0 * sin + x1 * cos
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)
    

sqp_len = 128
head_dim = 64
model = RoPEModule(sqp_len, head_dim)
x = torch.randn(1, sqp_len, head_dim)
torch.onnx.export(
    model, x, 
    "rope_standard.onnx", opset_version=13, 
    input_names=["input"], output_names=["output"])
print("ONNX model exported successfully.")

model = onnx.load("rope_standard.onnx")
for node in model.graph.node:
    print(f"{node.op_type:20s}  inputs={list(node.input)}  outputs={list(node.output)}")

# Build plugin graph from scratch — fresh tensors, no old graph imported
input_tensor  = gs.Variable("input",  dtype=np.float32, shape=[1, 2, 8, 64])
output_tensor = gs.Variable("output", dtype=np.float32, shape=[1, 2, 8, 64])

plugin_node = gs.Node(
    op="RoPEPlugin",
    name="rope_0",
    attrs={"num_heads": 2, "head_dim": 64},
    inputs=[input_tensor],
    outputs=[output_tensor]
)

# Create brand new graph — no connection to old imported graph
new_graph = gs.Graph(
    nodes=[plugin_node],
    inputs=[input_tensor],
    outputs=[output_tensor]
)

onnx.save(gs.export_onnx(new_graph), "rope_plugin.onnx")
print("Saved rope_plugin.onnx")

# Verify — should show exactly one node
model2 = onnx.load("rope_plugin.onnx")
for node in model2.graph.node:
    print(f"Node: {node.op_type}  attrs={[a.name for a in node.attribute]}")