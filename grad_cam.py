# =============================================================
#               GRAD-CAM IMPLEMENTATION (ResNet101)
# =============================================================
# This code:
# 1. Loads a trained ResNet101 model
# 2. Takes an X-ray image as input
# 3. Extracts feature maps from target layer (layer4)
# 4. Gets gradients w.r.t predicted class
# 5. Applies Grad-CAM algorithm to generate heatmap
# 6. Overlays heatmap on original image to show infected area
# =============================================================

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------------------------------------
#                     CONFIGURATION
# -------------------------------------------------------------
IMG_PATH = "142.jpeg"           # Input chest X-ray
MODEL_PATH = "your_model.pth"   # Final trained model path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target layer from which Grad-CAM feature maps will be extracted
# layer4 → best for showing localized features in ResNet
TARGET_LAYER = "layer4"


# -------------------------------------------------------------
#                     LOAD MODEL
# -------------------------------------------------------------
from torchvision.models import resnet101

# Create base ResNet101
model = resnet101(pretrained=False)

# Replace last FC layer for custom classes (5 diseases)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5)

# Load trained weights of your project
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()    # evaluation mode → no dropout, fixed behavior


# -------------------------------------------------------------
#                DEFINE TRANSFORMATIONS
# -------------------------------------------------------------
# Preprocessing same as used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize for ResNet input
    transforms.ToTensor(),              # Convert to tensor format
    transforms.Normalize(
        [0.485, 0.456, 0.406],          # Imagenet mean
        [0.229, 0.224, 0.225]           # Imagenet std
    )
])


# -------------------------------------------------------------
#                 LOAD INPUT IMAGE
# -------------------------------------------------------------
# Grad-CAM needs raw image + transformed tensor
image = Image.open(IMG_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # add batch dimension


# -------------------------------------------------------------
#        FORWARD HOOK: TO CAPTURE ACTIVATION MAPS
# -------------------------------------------------------------
# Grad-CAM principle:
#   1. Forward pass → get feature maps (activation)
#   2. Backward pass → get gradients
#   3. Weighted sum → heatmap

features = []   # Will store activation maps

def forward_hook(module, input, output):
    """
    This hook runs automatically when the forward pass reaches layer4.
    It extracts output feature maps of the selected layer.
    """
    features.append(output)

# Attach hook to the model's target layer dynamically
getattr(model, TARGET_LAYER).register_forward_hook(forward_hook)


# -------------------------------------------------------------
#                   FORWARD PASS
# -------------------------------------------------------------
# Run the model → calculate output prediction scores
output = model(input_tensor)

# Get the predicted class index (0–4)
pred_class = output.argmax(dim=1).item()


# -------------------------------------------------------------
#                BACKWARD PASS (Gradient)
# -------------------------------------------------------------
# Set gradients to zero
model.zero_grad()

# Compute gradient of predicted class w.r.t feature maps
output[0, pred_class].backward()

# Extract gradient from layer4
# NOTE: this depends on exact architecture blocks
grads = model.layer4[2].conv3.weight.grad

# If gradient exists inside features
gradients = features[0].grad if features[0].grad is not None else features[0]


# -------------------------------------------------------------
#    GLOBAL AVERAGE POOLING ON GRADIENTS (Grad-CAM Formula)
# -------------------------------------------------------------
# Each channel's importance value (α_k)
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# Extract feature maps of the target layer
activations = features[0][0]

# Multiply each channel of activations by corresponding gradient
for i in range(len(pooled_gradients)):
    activations[i, :, :] *= pooled_gradients[i]


# -------------------------------------------------------------
#     GENERATE HEATMAP (Grad-CAM Final Step)
# -------------------------------------------------------------
heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()

# Relu → Remove negative values
heatmap = np.maximum(heatmap, 0)

# Normalize heatmap between 0–1
heatmap /= heatmap.max()


# -------------------------------------------------------------
#           OVERLAY HEATMAP ON ORIGINAL IMAGE
# -------------------------------------------------------------
img = cv2.imread(IMG_PATH)

# Resize heatmap to match image size
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# Convert to color map (JET)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Overlay (0.4 = transparency factor)
superimposed_img = heatmap * 0.4 + img


# -------------------------------------------------------------
#                SAVE OUTPUT GRAD-CAM IMAGE
# -------------------------------------------------------------
output_path = "gradcam_result.jpg"
cv2.imwrite(output_path, superimposed_img)

print(f"[✔] Grad-CAM saved to {output_path}")
