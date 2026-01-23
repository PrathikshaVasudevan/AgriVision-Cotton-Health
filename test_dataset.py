from utils.dataset import CottonHealthDataset

dataset = CottonHealthDataset(
    healthy_dir="data/health/healthy_leaf",
    damaged_dir="data/health/damaged"
)

print("Total images:", len(dataset))

img, label = dataset[0]
print("Image shape:", img.shape)
print("Label:", label)
