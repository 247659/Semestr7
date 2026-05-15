import random

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

augmentation_1 = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

augmentation_2 = transforms.Compose([
    transforms.RandomAffine(degrees=15, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=None)

def show(img):
    return img if isinstance(img, torch.Tensor) else transforms.ToTensor()(img)

fig, axes = plt.subplots(2, 3, figsize=(10, 6))
fig.suptitle("Przykłady augmentacji MNIST", fontsize=14)

indices = random.sample(range(len(dataset)), 2)

for i, idx in enumerate(indices):
    img, label = dataset[idx]

    # ORYGINAŁ
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].axis('off')

    # AUG 1
    aug1 = augmentation_1(img)
    axes[i, 1].imshow(aug1.squeeze(), cmap='gray')
    axes[i, 1].axis('off')

    # AUG 2
    aug2 = augmentation_2(img)
    axes[i, 2].imshow(aug2.squeeze(), cmap='gray')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()