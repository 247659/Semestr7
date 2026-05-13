import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from sklearn.inspection import DecisionBoundaryDisplay


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
    """Klasa udająca model scikit-learn, adaptująca klasyfikator z PyTorch."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.classes_ = np.arange(10)  # sklearn wymaga zdefiniowania klas

        # DODANO: Flaga informująca sklearn, że ten estymator to klasyfikator
        self._estimator_type = "classifier"

        # DODANO: Pusta metoda fit, której scikit-learn bezwzględnie wymaga do walidacji

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            # Konwersja wejścia numpy (N, 2) na tensor
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            # Przepuszczenie wektora TYLKO przez końcowy klasyfikator Linear(2, 10)
            outputs = self.model.classifier(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

def extract_features(model, loader, device):
    """Pomocnicza funkcja zbierająca wszystkie cechy (X) i etykiety (y) z loadera."""
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

def plot_decision_boundaries(model, train_loader, test_loader, device, title):
    """Zaktualizowana funkcja tworząca granice decyzyjne i plotująca punkty."""

    print("Ekstrakcja cech dla wykresu (to może chwilę potrwać)...")
    X_train, y_train = extract_features(model, train_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)

    wrapped_model = PyTorchSklearnWrapper(model, device)

    fig, ax = plt.subplots(figsize=(10, 8))

    # =========================
    # Granice decyzyjne (wykorzystujące adapter)
    # =========================
    DecisionBoundaryDisplay.from_estimator(
        wrapped_model,
        X_train,  # Używamy punktów treningowych do określenia rozmiaru tła
        response_method="predict",
        alpha=0.35,
        ax=ax,
        eps=0.5,
        cmap="tab10"  # Dodałem cmap, by tło pasowało kolorystycznie do kropek!
    )

    # =========================
    # Punkty treningowe
    # =========================
    scatter_train = ax.scatter(
        X_train[:, 0], X_train[:, 1],
        c=y_train, cmap="tab10", edgecolor='k', s=20
    )

    # =========================
    # Punkty testowe
    # =========================
    ax.scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_test, cmap="tab10", marker='^', s=25
    )

    # =========================
    # LEGENDA CYFR
    # =========================
    colors = scatter_train.cmap(np.linspace(0, 1, 10))

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Cyfra {i}',
               markerfacecolor=colors[i], markersize=10)
        for i in range(10)
    ]

    ax.legend(handles=legend_elements, title="Klasy", bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title(title)

    # UWAGA: W sieciach CNN cechy są automatyczne i abstrakcyjne.
    # Etykiety "Suma pikseli" nie mają tu sensu fizycznego.
    ax.set_xlabel("Wyuczona Cecha 1")
    ax.set_ylabel("Wyuczona Cecha 2")

    plt.tight_layout()
    plt.show()


