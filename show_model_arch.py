import torch
import torch.nn as nn
from torchvision import models

# Same as your model setup
class_names = ['COVID', 'Normal', 'Pneumonia', 'Pneumothorax', 'Tuberculosis']
model = models.resnet101(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Linear(512, len(class_names))
)

# Load weights
model.load_state_dict(torch.load("resnet101_lung_model_320.pth", map_location="cpu"), strict=False)

# Print architecture
print(model)
