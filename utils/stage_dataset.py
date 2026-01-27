import os
import cv2
import torch
from torch.utils.data import Dataset
import random

class CottonStageDataset(Dataset):
    def __init__(self, root_dir, img_size=224, augment=True):
        self.images = []
        self.labels = []
        self.img_size = img_size
        self.augment = augment

        self.class_names = sorted(os.listdir(root_dir))

        for idx, folder in enumerate(self.class_names):
            folder_path = os.path.join(root_dir, folder)
            for img_name in os.listdir(folder_path):
                self.images.append(os.path.join(folder_path, img_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def augment_image(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        if random.random() < 0.5:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if random.random() < 0.5:
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        if random.random() < 0.3:
            noise = random.randint(0,20)
            img = cv2.add(img, noise)
        return img

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.augment:
            img = self.augment_image(img)

        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
        label = torch.tensor(label, dtype=torch.long)

        return img, label
