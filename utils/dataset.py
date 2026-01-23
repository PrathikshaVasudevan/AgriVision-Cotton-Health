import os
import cv2
import torch
from torch.utils.data import Dataset

class CottonHealthDataset(Dataset):
    def __init__(self, healthy_dir, damaged_dir, img_size=224):
        self.images = []
        self.labels = []
        self.img_size = img_size

        # Healthy images → label 0
        for img_name in os.listdir(healthy_dir):
            self.images.append(os.path.join(healthy_dir, img_name))
            self.labels.append(0)

        # Damaged images → label 1
        for img_name in os.listdir(damaged_dir):
            self.images.append(os.path.join(damaged_dir, img_name))
            self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        img = img / 255.0  # normalize
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)  # HWC → CHW

        label = torch.tensor(label, dtype=torch.long)

        return img, label
