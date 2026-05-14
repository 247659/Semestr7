import torch

from models_mnist.cnn_classifier_first import CNNClassifier

from datasets.data_loaders import (
    get_imagenette_loaders
)

from utils.train import train_model

from utils.plots import plot_accuracy

from utils.metrics import (
    show_confusion_matrix
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================
# ŚCIEŻKI
# ============================================

TRAIN_PATH = "./imagenette2/train"
TEST_PATH = "./imagenette2/val"

# ============================================
# LOADERS
# ============================================

train_loader, test_loader, classes = (
    get_imagenette_loaders(
        TRAIN_PATH,
        TEST_PATH
    )
)

# ============================================
# MODEL
# ============================================

model = CNNClassifier(
    num_classes=10,
    in_channels=3
).to(device)

train_acc, test_acc = train_model(
    model,
    train_loader,
    test_loader,
    epochs=10
)

plot_accuracy(
    train_acc,
    test_acc,
    "Imagenette - CNN"
)

show_confusion_matrix(
    model,
    test_loader,
    classes=classes,
    device=device
)

torch.save(
    model.state_dict(),
    "imagenette_cnn.pth"
)