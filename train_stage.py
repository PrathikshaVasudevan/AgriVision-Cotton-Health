import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.stage_dataset import CottonStageDataset
from models.stage_model import StageCNN

dataset = CottonStageDataset("data/stage", augment=True)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = StageCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 5

for epoch in range(epochs):
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/stage_model.pth")
print("Stage model saved!")
