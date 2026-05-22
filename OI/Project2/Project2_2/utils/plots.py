import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import torch


def plot_accuracy(train_acc, test_acc, title):
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(test_acc, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.show()


class PyTorchSklearnWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.classes_ = np.arange(10)
        self._estimator_type = "classifier"

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model.classifier(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()


def extract_features(model, loader, device):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _, features = model(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(features_list), np.concatenate(labels_list)


def plot_decision_boundaries(model, train_loader, test_loader, device, title, max_points=500, class_names=None):
    X_train, y_train = extract_features(model, train_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)

    if len(X_train) > max_points:
        idx = np.random.choice(
            len(X_train),
            max_points,
            replace=False
        )

        X_train = X_train[idx]
        y_train = y_train[idx]

    if len(X_test) > max_points:
        idx = np.random.choice(
            len(X_test),
            max_points,
            replace=False
        )

        X_test = X_test[idx]
        y_test = y_test[idx]

    cmap = ListedColormap(
        plt.cm.tab10.colors
    )
    X_all = np.vstack((X_train, X_test))

    x_min = X_all[:, 0].min() - 5
    x_max = X_all[:, 0].max() + 5

    y_min = X_all[:, 1].min() - 5
    y_max = X_all[:, 1].max() + 5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    model.eval()

    with torch.no_grad():
        grid_tensor = torch.tensor(
            grid,
            dtype=torch.float32
        ).to(device)

        outputs = model.classifier(grid_tensor)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

    Z = predictions.cpu().numpy()
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    ax.contourf(
        xx,
        yy,
        Z,
        alpha=0.25,
        levels=np.arange(11) - 0.5,
        cmap=cmap
    )

    ax.contour(
        xx,
        yy,
        Z,
        colors='k',
        linewidths=0.3,
        alpha=0.4
    )

    scatter_train = ax.scatter(

        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap=cmap,
        edgecolor='k',
        s=30,
        marker='o',
        label='Train',
        vmin=0,
        vmax=9
    )

    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=cmap,
        s=40,
        marker='^',
        label='Test',
        vmin=0,
        vmax=9
    )

    colors = cmap.colors

    if class_names is None:
        class_names = [f'Cyfra {i}' for i in range(10)]

    class_elements = [
        Line2D(
            [0], [0],
            marker='s', color='w',
            label=class_names[i],
            markerfacecolor=colors[i],
            markersize=10
        )
        for i in range(10)
    ]

    shape_elements = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Zbiór treningowy',
            markerfacecolor='gray',
            markeredgecolor='k',
            markersize=10
        ),

        Line2D(
            [0],
            [0],
            marker='^',
            color='w',
            label='Zbiór testowy',
            markerfacecolor='gray',
            markersize=10
        )
    ]

    ax.legend(
        handles=
        class_elements
        + [Line2D([0], [0], color='w', label='')]
        + shape_elements,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    ax.set_title(title)
    ax.set_xlabel("Cecha 1")
    ax.set_ylabel("Cecha 2")
    plt.tight_layout()
    plt.show()


