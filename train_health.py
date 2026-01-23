import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import CottonHealthDataset
from models.health_model import HealthCNN

# Paths
healthy_dir = "data/health/healthy_leaf"
damaged_dir = "data/health/damaged"

# Dataset & Loader
dataset = CottonHealthDataset(healthy_dir, damaged_dir)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HealthCNN().to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training
epochs = 3   # keep small for now

for epoch in range(epochs):
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "models/health_model.pth")
print("Model saved!")
