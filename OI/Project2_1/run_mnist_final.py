import random

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Importy z Twoich plików
from dataset import NumpyDataset
from model import MLP
from train import train_model
from visualize import plot_decision


def set_seed(seed=42):
    """
    Zamraża wszystkie źródła losowości, aby wyniki były w 100% odtwarzalne.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================================================================
# WRAPPER DO RYSOWANIA GRANIC DECYZYJNYCH
# =====================================================================
class PyTorchSklearnWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, model):
        self.model = model
        self.classes_ = np.arange(10)

    def fit(self, X, y):
        # sklearn oczekuje istnienia fit()
        return self

    def predict(self, X):

        self.model.eval()

        with torch.no_grad():

            X_tensor = torch.tensor(X, dtype=torch.float32)

            outputs = self.model(X_tensor)

            predictions = torch.argmax(outputs, dim=1)

        return predictions.numpy()


# =====================================================================
# FUNKCJE POMOCNICZE
# =====================================================================
def evaluate_and_plot_cm(model, data_loader, dataset_name, subset_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in data_loader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(y.numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"[{dataset_name}] Skuteczność na zbiorze {subset_name}: {acc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Macierz pomyłek - {dataset_name} ({subset_name})")
    plt.ylabel("Prawdziwa klasa")
    plt.xlabel("Przewidziana klasa")
    plt.tight_layout()
    plt.show()


def prepare_dataloaders(X_train, y_train, X_test, y_test, batch_size=256):  # Zwiększony batch_size dla MNIST
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    train_loader = DataLoader(NumpyDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NumpyDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(NumpyDataset(X_test_scaled, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_tr.shape[1], X_tr, y_tr, X_test_scaled


def run_optimal_model(name, X_train, y_train, X_test, y_test, num_classes, best_config, draw_boundaries=False):
    print(f"\n{'=' * 60}")
    print(f"TRENOWANIE OPTYMALNEGO MODELU: {name}")
    print(f"Użyte hiperparametry: {best_config}")
    print(f"{'=' * 60}")
    set_seed()
    train_loader, val_loader, test_loader, input_size, X_tr, y_tr, X_te_scaled = prepare_dataloaders(X_train, y_train,
                                                                                                     X_test, y_test)

    model = MLP(input_size=input_size,
                hidden_size=best_config['hidden_size'],
                num_classes=num_classes,
                use_relu=best_config['use_relu'])

    trained_model = train_model(model, train_loader, val_loader, epochs=300, lr=best_config['lr'], patience=5)

    print("\n--- GENEROWANIE WYNIKÓW ---")
    evaluate_and_plot_cm(trained_model, train_loader, name, "Treningowy")
    evaluate_and_plot_cm(trained_model, test_loader, name, "Testowy")

    # Rysowanie granic decyzyjnych, jeśli wymiarowość = 2
    if draw_boundaries and input_size == 2:
        print(f"\nGenerowanie granic decyzyjnych dla: {name}...")
        wrapped_model = PyTorchSklearnWrapper(trained_model)

        # Pobieramy próbkę danych, żeby wykres nie rysował się w nieskończoność
        num_samples = 300
        indices = np.random.choice(len(X_tr), num_samples, replace=False)
        plot_decision(
            wrapped_model,
            X_tr[indices],
            X_te_scaled[:500],
            y_tr[indices],
            y_test[:500],
            f"Granice decyzyjne - {name}"
        )


# =====================================================================
# GŁÓWNA FUNKCJA
# =====================================================================
def main():
    # Wczytanie etykiet
    try:
        y_train = np.load("features/y_train.npy")
        y_test = np.load("features/y_test.npy")
    except FileNotFoundError:
        print("[BŁĄD] Brak plików. Uruchom najpierw skrypt features.py!")
        return

    # 1. Metoda 1_1: 2 Cechy (Suma, Balans)
    X1_tr = np.load("features/X1_train.npy")
    X1_te = np.load("features/X1_test.npy")
    cfg_1 = {'hidden_size': 8, 'lr': 0.01, 'use_relu': True}
    run_optimal_model("MNIST - 2 Cechy (Suma, Balans)", X1_tr, y_train, X1_te, y_test, 10, cfg_1, draw_boundaries=True)

    # 2. Metoda 2_1: 2 Cechy (Pętle, Wariancja)
    X2_tr = np.load("features/X2_train.npy")
    X2_te = np.load("features/X2_test.npy")
    cfg_2 = {'hidden_size': 16, 'lr': 0.01, 'use_relu': True}
    run_optimal_model("MNIST - 2 Cechy (Pętle, Wariancja)", X2_tr, y_train, X2_te, y_test, 10, cfg_2,
                      draw_boundaries=True)

    # 3. Metoda 2_2: 6 Cech Intuicyjnych
    X3_tr = np.load("features/X3_train.npy")
    X3_te = np.load("features/X3_test.npy")
    cfg_3 = {'hidden_size': 64, 'lr': 0.01, 'use_relu': True}
    run_optimal_model("MNIST - 6 Cech Intuicyjnych", X3_tr, y_train, X3_te, y_test, 10, cfg_3, draw_boundaries=False)

    # 4. Metoda 1_2: 20 Cech (PCA)
    X_pca_tr = np.load("features/X_pca_train.npy")
    X_pca_te = np.load("features/X_pca_test.npy")
    cfg_pca = {'hidden_size': 100, 'lr': 0.01, 'use_relu': True}
    run_optimal_model("MNIST - 20 Cech (PCA)", X_pca_tr, y_train, X_pca_te, y_test, 10, cfg_pca, draw_boundaries=False)


if __name__ == "__main__":
    main()