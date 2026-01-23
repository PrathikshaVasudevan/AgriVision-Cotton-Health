import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

class HealthCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace last layer (2 classes: healthy / damaged)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)
