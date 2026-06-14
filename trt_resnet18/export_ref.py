import torch
from torchvision import models
import numpy as np

#load pretrained resnet18
model = models.resnet18(pretrained=True)
model.eval()    
#generate random input
input_data = torch.randn(1, 3, 224, 224)
#forward pass
with torch.no_grad():
    output = model(input_data)
#Save input and output as raw float32 binary
input_data.numpy().astype(np.float32).tofile('resnet18_input.bin')
output.numpy().astype(np.float32).tofile('resnet18_output.bin')

print(f"Input  shape : {input_data.shape}  -> test_input.bin")
print(f"Output shape : {output.shape} -> test_output_ref.bin")
print(f"Predicted class (PyTorch): {output.argmax().item()}")
