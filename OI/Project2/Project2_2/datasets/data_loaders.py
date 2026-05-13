import torchvision.transforms as transforms

from torchvision.datasets import MNIST, ImageFolder
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
        train_path,
        test_path,
        batch_size=32):

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    train_dataset = ImageFolder(
        train_path,
        transform=train_transform
    )

    test_dataset = ImageFolder(
        test_path,
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