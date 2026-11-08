import torch 
import torchvision.models as models

model = models.resnet18(pretrained=True).eval().cuda()
dummy_input = torch.randn(1, 3, 224, 224).cuda()

torch.onnx.export(
    model, dummy_input, 
    "resnet18.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0:"batch"}},
    opset_version=17)