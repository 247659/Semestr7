import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importy z Twoich własnych plików
from dataset import NumpyDataset
from model import MLP
from train import train_model


# =====================================================================
# 1. FUNKCJE POMOCNICZE I EWALUACYJNE
# =====================================================================

def evaluate_accuracy(model, data_loader):
    """Oblicza skuteczność (Accuracy) modelu na podanym zbiorze danych."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total if total > 0 else 0


def prepare_dataloaders(X_train, y_train, X_test, y_test, batch_size=16):
    """
    Standaryzuje dane, wydziela zbiór walidacyjny i tworzy DataLoadery PyTorch.
    """
    # Standaryzacja
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Wydzielenie zbioru walidacyjnego (20% ze zbioru treningowego)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    # Tworzenie Datasetów
    train_dataset = NumpyDataset(X_tr, y_tr)
    val_dataset = NumpyDataset(X_val, y_val)
    test_dataset = NumpyDataset(X_test_scaled, y_test)

    # Tworzenie DataLoaderów
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_tr.shape[1]


# =====================================================================
# 2. SILNIK PRZESZUKIWANIA SIATKI (GRID SEARCH)
# =====================================================================

def find_best_model(train_loader, val_loader, input_size, num_classes):
    """
    Testuje różne konfiguracje hiperparametrów i zwraca najlepszy model
    (na podstawie zbioru walidacyjnego).
    """
    print("Rozpoczynam poszukiwanie optymalnego modelu...")

    # Zdefiniowana przestrzeń poszukiwań
    hidden_sizes = [8, 16, 32, 64, 100]
    learning_rates = [0.01, 0.001]
    use_relu_options = [True, False]

    best_accuracy = 0.0
    best_config = {}
    best_model_state = None

    for hidden in hidden_sizes:
        for lr in learning_rates:
            for relu in use_relu_options:

                print(f"\nTestuję: Hidden={hidden}, LR={lr}, ReLU={relu}")

                # Inicjalizacja modelu
                model = MLP(input_size=input_size,
                            hidden_size=hidden,
                            num_classes=num_classes,
                            use_relu=relu)

                # Trening z Early Stopping (val_loader decyduje o przerwaniu!)
                trained_model = train_model(model, train_loader, val_loader, epochs=200, lr=lr, patience=5)

                # Ewaluacja na zbiorze walidacyjnym
                val_acc = evaluate_accuracy(trained_model, val_loader)
                print(f"-> Skuteczność walidacyjna: {val_acc:.4f}")

                # Zapisujemy, jeśli jest to najlepszy wynik do tej pory
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_config = {'hidden_size': hidden, 'lr': lr, 'use_relu': relu}
                    best_model_state = trained_model.state_dict()

    print("\n" + "-" * 50)
    print(f"ZAKOŃCZONO POSZUKIWANIA. Najlepsza skuteczność walidacyjna: {best_accuracy:.4f}")
    print(f"Zwycięskie hiperparametry: {best_config}")
    print("-" * 50)

    # Odtworzenie najlepszego modelu z zapisanych wag
    optimal_model = MLP(input_size=input_size,
                        hidden_size=best_config['hidden_size'],
                        num_classes=num_classes,
                        use_relu=best_config['use_relu'])
    optimal_model.load_state_dict(best_model_state)

    return optimal_model, best_config


# =====================================================================
# 3. GŁÓWNY PIPELINE DLA ZBIORU
# =====================================================================

def run_full_pipeline(name, X_train, y_train, X_test, y_test, num_classes):
    """Kompleksowo przetwarza jeden zbiór danych."""
    print(f"\n\n{'=' * 60}")
    print(f"EKSPERYMENT: {name}")
    print(f"{'=' * 60}")

    # 1. Przygotowanie danych
    train_loader, val_loader, test_loader, input_size = prepare_dataloaders(X_train, y_train, X_test, y_test)

    # 2. Szukanie najlepszego modelu (na Train i Val)
    best_model, best_params = find_best_model(train_loader, val_loader, input_size=input_size, num_classes=num_classes)

    # 3. OSTATECZNY TEST (Na niewidzianym zbiorze Test)
    final_acc = evaluate_accuracy(best_model, test_loader)

    print(f"\n[PODSUMOWANIE OSTATECZNE - {name}]")
    print(f"Najlepsza architektura: {best_params}")
    print(f"Skuteczność (Accuracy) na ZBIORZE TESTOWYM: {final_acc:.4f}")


# =====================================================================
# 4. URUCHOMIENIE WSZYSTKICH EKSPERYMENTÓW
# =====================================================================

def main():
    # ---------------------------------------------------------
    # Eksperyment 1: Zbiór IRIS (3 klasy)
    # ---------------------------------------------------------
    # iris = load_iris()
    # X_train_ir, X_test_ir, y_train_ir, y_test_ir = train_test_split(iris.data, iris.target, test_size=0.2,
    #                                                                 random_state=42)
    # run_full_pipeline("Zbiór IRIS", X_train_ir, y_train_ir, X_test_ir, y_test_ir, num_classes=3)
    #
    # # ---------------------------------------------------------
    # # Eksperyment 2: Zbiór WINE (3 klasy)
    # # ---------------------------------------------------------
    # wine = load_wine()
    # X_train_wi, X_test_wi, y_train_wi, y_test_wi = train_test_split(wine.data, wine.target, test_size=0.2,
    #                                                                 random_state=42)
    # run_full_pipeline("Zbiór WINE", X_train_wi, y_train_wi, X_test_wi, y_test_wi, num_classes=3)

    # ---------------------------------------------------------
    # Eksperyment 3: Zbiór MNIST (10 klas)
    # ---------------------------------------------------------
    try:
        y_train_mn = np.load("features/y_train.npy")
        y_test_mn = np.load("features/y_test.npy")

        # # Metoda 1_1 (2 Cechy)
        # X_tr_m21 = np.load("features/X1_train.npy")
        # X_te_m21 = np.load("features/X1_test.npy")
        # run_full_pipeline("MNIST - 2 Cechy (Suma, Balance)", X_tr_m21, y_train_mn, X_te_m21, y_test_mn,
        #                   num_classes=10)
        #
        # #Metoda 1_2 (20 cech)
        # X_tr_pca = np.load("features/X_pca_train.npy")
        # X_te_pca = np.load("features/X_pca_test.npy")
        # run_full_pipeline("MNIST - 20 Cech (PCA)", X_tr_pca, y_train_mn, X_te_pca, y_test_mn, num_classes=10)

        # Metoda 2_1 (2 Cechy)
        X_tr_m21 = np.load("features/X2_train.npy")
        X_te_m21 = np.load("features/X2_test.npy")
        run_full_pipeline("MNIST - 2 Cechy (Pętle, Wariancja)", X_tr_m21, y_train_mn, X_te_m21, y_test_mn,
                          num_classes=10)

        # Metoda 2_2 (6 Cech - jeśli nadpisałeś ten wariant)
        X_tr_m22 = np.load("features/X3_train.npy")
        X_te_m22 = np.load("features/X3_test.npy")
        run_full_pipeline("MNIST - 6 Cech Intuicyjnych", X_tr_m22, y_train_mn, X_te_m22, y_test_mn, num_classes=10)

        X_flatten_train = np.load("features/X_flatten.npy")
        X_flatten_test = np.load("features/X_flatten_test.npy")
        run_full_pipeline("Flatten", X_flatten_train, y_train_mn, X_flatten_test, y_test_mn, num_classes=10)

    except FileNotFoundError:
        print("\n[BŁĄD] Nie znaleziono plików .npy dla MNIST w folderze 'features/'.")
        print("Upewnij się, że uruchomiłeś skrypt ekstrakcji cech przed odpaleniem treningu.")


if __name__ == "__main__":
    main()