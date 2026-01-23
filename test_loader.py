import os
import cv2
import matplotlib.pyplot as plt

healthy_path = "data/health/healthy_leaf"
damaged_path = "data/health/damaged"

def show_images(folder, label, n=3):
    images = os.listdir(folder)
    print(f"{label} images found:", len(images))

    for i in range(min(n, len(images))):
        img_path = os.path.join(folder, images[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
        plt.show()

show_images(healthy_path, "Healthy")
show_images(damaged_path, "Damaged")
