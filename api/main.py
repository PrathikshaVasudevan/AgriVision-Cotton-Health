from fastapi import FastAPI, UploadFile, File
import torch
import cv2
import numpy as np
import os

from models.health_model import HealthCNN
from models.stage_model import StageCNN

app = FastAPI(title="Agri-Vision Cotton API")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load HEALTH model
health_model = HealthCNN().to(device)
health_model.load_state_dict(torch.load("models/health_model.pth", map_location=device))
health_model.eval()

# Load STAGE model
stage_model = StageCNN().to(device)
stage_model.load_state_dict(torch.load("models/stage_model.pth", map_location=device))
stage_model.eval()

# Stage names
stage_names = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Read image
    contents = await image.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    # To tensor
    tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        # ---- STAGE PREDICTION ----
        stage_out = stage_model(tensor)
        stage_idx = torch.argmax(stage_out).item()
        stage = stage_names[stage_idx]

        # ---- HEALTH PREDICTION ----
        health_out = health_model(tensor)
        probs = torch.softmax(health_out, dim=1)[0]

        healthy_prob = probs[0].item()
        damaged_prob = probs[1].item()

        health_score = int(healthy_prob * 100)
        health_status = "Healthy" if healthy_prob > damaged_prob else "Damaged"

    # is_ripped logic
    is_ripped = True if stage in ["Phase 3", "Phase 4"] else False

    return {
        "stage": stage,
        "is_ripped": is_ripped,
        "health_status": health_status,
        "health_score": health_score
    }
