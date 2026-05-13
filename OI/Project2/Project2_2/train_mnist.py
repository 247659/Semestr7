import torch

from models.cnn_classifier_first import CNNClassifier
from models.cnn_2features_first import CNN2Features

from datasets.data_loaders import get_mnist_loaders

from utils.train import train_model
from utils.plots import (
    plot_accuracy,
    plot_decision_boundaries
)

from utils.metrics import (
    show_confusion_matrix
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================
# LOADERS
# ============================================

train_loader, test_loader = get_mnist_loaders()

# ============================================
# MODEL 1 - STANDARD CNN
# ============================================

# model = CNNClassifier(
#     num_classes=10
# ).to(device)
#
# train_acc, test_acc = train_model(
#     model,
#     train_loader,
#     test_loader,
#     "CNNFirst"
# )
#
# plot_accuracy(
#     train_acc,
#     test_acc,
#     "MNIST - CNN"
# )
#
# show_confusion_matrix(
#     model,
#     test_loader,
#     classes=[str(i) for i in range(10)],
#     device=device
# )

# ============================================
# MODEL 2 - 2 FEATURES
# ============================================

model_2f = CNN2Features().to(device)

train_acc_2f, test_acc_2f = train_model(
    model_2f,
    train_loader,
    test_loader,
    "CNN2FeaturesFirst",
    feature_model=True
)

plot_accuracy(
    train_acc_2f,
    test_acc_2f,
    "MNIST - CNN 2 Features"
)

plot_decision_boundaries(
    model_2f,
    train_loader,
    test_loader,
    device,
    "Granice decyzyjne"
)

