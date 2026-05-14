import torchvision.transforms as transforms

from torchvision.datasets import MNIST, Imagenette
from torch.utils.data import DataLoader


def get_mnist_loaders(batch_size=64):

    # Sieci neuronowe najlepiej działają gdy dane są wycentrowane wokół zera i mają podobną skalę
    # mean - średnia
    # std - odchylenie standardowe
    # poniżej zastosowano wartości najlepsze dla zbioru MNIST 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def get_imagenette_loaders(
        batch_size=32):

    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    train_dataset = Imagenette(
        root="./data",
        split="train",
        download=True,
        transform=train_transform
    )

    test_dataset = Imagenette(
        root="./data",
        split="val",
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, train_dataset.classes