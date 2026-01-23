from fastapi import FastAPI, UploadFile, File
import torch
import cv2
import numpy as np
import os
from models.health_model import HealthCNN

app = FastAPI(title="Agri-Vision Cotton Health API")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HealthCNN().to(device)
model.load_state_dict(torch.load("models/health_model.pth", map_location=device))
model.eval()

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Read image
    contents = await image.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    healthy_prob = probs[0].item()
    damaged_prob = probs[1].item()

    health_score = int(healthy_prob * 100)

    return {
        "health_status": "Healthy" if healthy_prob > damaged_prob else "Damaged",
        "health_score": health_score
    }
