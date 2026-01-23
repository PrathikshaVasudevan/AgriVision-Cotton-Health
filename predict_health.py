import os
import torch
import cv2
import numpy as np
from models.health_model import HealthCNN

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HealthCNN().to(device)
model.load_state_dict(torch.load("models/health_model.pth", map_location=device))
model.eval()

def predict_health(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]

    healthy_prob = probs[0].item()
    damaged_prob = probs[1].item()

    health_score = int(healthy_prob * 100)

    result = {
        "health_status": "Healthy" if healthy_prob > damaged_prob else "Damaged",
        "health_score": health_score
    }

    return result

# TEST (change image path)
test_image = "data/health/healthy_leaf/" + os.listdir("data/health/healthy_leaf")[0]
print(predict_health(test_image))
