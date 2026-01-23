import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.health_model import HealthCNN

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, class_idx):
        self.model.zero_grad()
        output = self.model(input_image)
        output[0, class_idx].backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam


# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HealthCNN().to(device)
model.load_state_dict(torch.load("models/health_model.pth", map_location=device))
model.eval()

# Target last conv layer
target_layer = model.model.layer4[-1].conv2
gradcam = GradCAM(model, target_layer)

# Load image
img_path = "data/health/healthy_leaf/" + os.listdir("data/health/healthy_leaf")[0]
original = cv2.imread(img_path)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
resized = cv2.resize(original, (224, 224)) / 255.0

input_tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1)
input_tensor = input_tensor.unsqueeze(0).to(device)

# Predict
output = model(input_tensor)
pred_class = torch.argmax(output).item()

# Generate CAM
cam = gradcam.generate(input_tensor, pred_class)

# Overlay heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# Show results
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(original)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Grad-CAM")
plt.imshow(cam, cmap="jet")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.show()
