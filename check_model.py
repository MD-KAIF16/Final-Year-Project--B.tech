import torch

model = torch.load("resnet101_lung_model_320.pth", map_location=torch.device('cpu'))
print(model)
