import random

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Importy z Twoich plików
from dataset import NumpyDataset
from model import MLP
from train import train_model


# =====================================================================
# 1. FUNKCJE POMOCNICZE
# =====================================================================

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

def evaluate_and_plot_cm(model, data_loader, dataset_name, subset_name):
    """
    Oblicza skuteczność i rysuje macierz pomyłek.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in data_loader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(y.numpy())

    # Obliczenie Accuracy
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"[{dataset_name}] Skuteczność na zbiorze {subset_name}: {acc:.4f}")

    # Rysowanie macierzy pomyłek
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Macierz pomyłek - {dataset_name} ({subset_name})")
    plt.ylabel("Prawdziwa klasa")
    plt.xlabel("Przewidziana klasa")
    plt.tight_layout()
    plt.show()


def prepare_dataloaders(X_train, y_train, X_test, y_test, batch_size=16):
    """
    Standaryzuje dane, wydziela zbiór walidacyjny i tworzy DataLoadery.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    train_loader = DataLoader(NumpyDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NumpyDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(NumpyDataset(X_test_scaled, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_tr.shape[1]


# =====================================================================
# 2. GŁÓWNA FUNKCJA URUCHAMIAJĄCA OSTATECZNY MODEL
# =====================================================================

def run_optimal_model(name, X_train, y_train, X_test, y_test, num_classes, best_config):
    print(f"\n{'=' * 60}")
    print(f"TRENOWANIE OPTYMALNEGO MODELU: {name}")
    print(f"Użyte hiperparametry: {best_config}")
    print(f"{'=' * 60}")
    set_seed()
    train_loader, val_loader, test_loader, input_size = prepare_dataloaders(X_train, y_train, X_test, y_test)

    # Inicjalizacja modelu z optymalnymi hiperparametrami
    model = MLP(input_size=input_size,
                hidden_size=best_config['hidden_size'],
                num_classes=num_classes,
                use_relu=best_config['use_relu'])

    # Trening (używamy zbioru walidacyjnego do Early Stoppingu, żeby nie przeuczyć sieci!)
    trained_model = train_model(model, train_loader, val_loader,
                                epochs=300,
                                lr=best_config['lr'],
                                patience=5)

    # Ewaluacja i rysowanie macierzy pomyłek na zbiorze TRENINGOWYM (Train) i TESTOWYM (Test)
    print("\n--- GENEROWANIE WYNIKÓW ---")
    evaluate_and_plot_cm(trained_model, train_loader, name, "Treningowy")
    evaluate_and_plot_cm(trained_model, test_loader, name, "Testowy")


# =====================================================================
# 3. ZDEFINIOWANE EKSPERYMENTY
# =====================================================================

def main():
    # 1. Zbiór IRIS
    iris = load_iris()
    X_train_ir, X_test_ir, y_train_ir, y_test_ir = train_test_split(iris.data, iris.target, test_size=0.2,
                                                                    random_state=42)

    iris_optimal_config = {'hidden_size': 16, 'lr': 0.01, 'use_relu': True}
    run_optimal_model("IRIS", X_train_ir, y_train_ir, X_test_ir, y_test_ir, 3, iris_optimal_config)

    # 2. Zbiór WINE
    wine = load_wine()
    X_train_wi, X_test_wi, y_train_wi, y_test_wi = train_test_split(wine.data, wine.target, test_size=0.2,
                                                                    random_state=42)

    wine_optimal_config = {'hidden_size': 8, 'lr': 0.01, 'use_relu': True}
    run_optimal_model("WINE", X_train_wi, y_train_wi, X_test_wi, y_test_wi, 3, wine_optimal_config)

    # 3. Zbiór MNIST (Flatten - 784 cechy)
    try:
        y_train_mn = np.load("features/y_train.npy")
        y_test_mn = np.load("features/y_test.npy")
        X_tr_flat = np.load("features/X_flat_train.npy")
        X_te_flat = np.load("features/X_flat_test.npy")

        # UZUPEŁNIJ TE WARTOŚCI, GDY SKOŃCZY SIĘ GRID SEARCH DLA MNIST:
        mnist_optimal_config = {'hidden_size': 100, 'lr': 0.01, 'use_relu': True}

        # Opcjonalnie: Zwiększ batch_size w prepare_dataloaders dla MNIST, żeby było szybciej
        run_optimal_model("MNIST (Flatten)", X_tr_flat, y_train_mn, X_te_flat, y_test_mn, 10, mnist_optimal_config)

    except FileNotFoundError:
        print(
            "\n[INFO] Pliki MNIST nie zostały wczytane. Jeśli chcesz przetestować MNIST, upewnij się, że pliki .npy istnieją w folderze 'features/'.")


if __name__ == "__main__":
    main()