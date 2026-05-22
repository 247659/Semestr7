import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

base_transform = transforms.Compose([
    transforms.Resize((160, 160))
])

augmentation_1 = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4)
])

augmentation_2 = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomResizedCrop(128, scale=(0.6, 1.0)),
    transforms.GaussianBlur(kernel_size=3)
])

dataset = datasets.Imagenette(root="./data", split="train", size="160px", download=True, transform=None)

fig, axes = plt.subplots(2, 3, figsize=(10, 6))
fig.suptitle("Przykłady augmentacji Imagenette", fontsize=14)

indices = random.sample(range(len(dataset)), 2)

for i, idx in enumerate(indices):
    img, label = dataset[idx]

    axes[i, 0].imshow(base_transform(img))
    axes[i, 0].set_title("Oryginał") if i == 0 else None
    axes[i, 0].axis('off')

    axes[i, 1].imshow(augmentation_1(img))
    axes[i, 1].set_title("Aug 1 (Odbicie + Kolor)") if i == 0 else None
    axes[i, 1].axis('off')

    axes[i, 2].imshow(augmentation_2(img))
    axes[i, 2].set_title("Aug 2 (Crop + Blur)") if i == 0 else None
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()