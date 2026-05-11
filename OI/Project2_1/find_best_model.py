import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset import NumpyDataset
from model import MLP
from train import train_model

def evaluate_accuracy(model, data_loader):
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
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().transform(X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    train_dataset = NumpyDataset(X_tr, y_tr)
    val_dataset = NumpyDataset(X_val, y_val)
    test_dataset = NumpyDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_tr.shape[1]

def find_best_model(train_loader, val_loader, input_size, num_classes):
    print("Rozpoczynam poszukiwanie optymalnego modelu...")

    hidden_sizes = [8, 16, 32, 64, 100]
    use_relu_options = [True, False]

    best_accuracy = 0.0
    best_config = {}
    best_model_state = None

    for hidden in hidden_sizes:
        for relu in use_relu_options:

            print(f"\nTestuję: Hidden={hidden}, ReLU={relu}")

            model = MLP(input_size=input_size,
                        hidden_size=hidden,
                        num_classes=num_classes,
                        use_relu=relu)

            trained_model = train_model(model, train_loader, val_loader, epochs=200, lr=0.01, patience=5)

            val_acc = evaluate_accuracy(trained_model, val_loader)
            print(f"-> Skuteczność walidacyjna: {val_acc:.4f}")

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_config = {'hidden_size': hidden, 'use_relu': relu}
                best_model_state = trained_model.state_dict()

    print(f"ZAKOŃCZONO POSZUKIWANIA. Najlepsza skuteczność walidacyjna: {best_accuracy:.4f}")
    print(f"Zwycięskie hiperparametry: {best_config}")

    optimal_model = MLP(input_size=input_size,
                        hidden_size=best_config['hidden_size'],
                        num_classes=num_classes,
                        use_relu=best_config['use_relu'])
    optimal_model.load_state_dict(best_model_state)

    return optimal_model, best_config

def run_full_pipeline(name, X_train, y_train, X_test, y_test, num_classes):
    print(f"EKSPERYMENT: {name}")

    train_loader, val_loader, test_loader, input_size = prepare_dataloaders(X_train, y_train, X_test, y_test)
    best_model, best_params = find_best_model(train_loader, val_loader, input_size=input_size, num_classes=num_classes)
    final_acc = evaluate_accuracy(best_model, test_loader)

    print(f"\n[PODSUMOWANIE OSTATECZNE - {name}]")
    print(f"Najlepsza architektura: {best_params}")
    print(f"Skuteczność (Accuracy) na ZBIORZE TESTOWYM: {final_acc:.4f}")

def main():
    iris = load_iris()
    X_train_ir, X_test_ir, y_train_ir, y_test_ir = train_test_split(iris.data, iris.target, test_size=0.2,
                                                                    random_state=42)
    run_full_pipeline("Zbiór IRIS", X_train_ir, y_train_ir, X_test_ir, y_test_ir, num_classes=3)

    wine = load_wine()
    X_train_wi, X_test_wi, y_train_wi, y_test_wi = train_test_split(wine.data, wine.target, test_size=0.2,
                                                                    random_state=42)
    run_full_pipeline("Zbiór WINE", X_train_wi, y_train_wi, X_test_wi, y_test_wi, num_classes=3)

    y_train_mn = np.load("features/y_train.npy")
    y_test_mn = np.load("features/y_test.npy")

    X_tr_m21 = np.load("features/X1_train.npy")
    X_te_m21 = np.load("features/X1_test.npy")
    run_full_pipeline("MNIST - 2 Cechy (Suma, Balance)", X_tr_m21, y_train_mn, X_te_m21, y_test_mn,
                      num_classes=10)

    X_tr_pca = np.load("features/X_pca_train.npy")
    X_te_pca = np.load("features/X_pca_test.npy")
    run_full_pipeline("MNIST - 20 Cech (PCA)", X_tr_pca, y_train_mn, X_te_pca, y_test_mn, num_classes=10)

    X_tr_m21 = np.load("features/X2_train.npy")
    X_te_m21 = np.load("features/X2_test.npy")
    run_full_pipeline("MNIST - 2 Cechy (Pętle, Wariancja)", X_tr_m21, y_train_mn, X_te_m21, y_test_mn,
                      num_classes=10)

    X_tr_m22 = np.load("features/X3_train.npy")
    X_te_m22 = np.load("features/X3_test.npy")
    run_full_pipeline("MNIST - 6 Cech Intuicyjnych", X_tr_m22, y_train_mn, X_te_m22, y_test_mn, num_classes=10)

    X_flatten_train = np.load("features/X_flatten.npy")
    X_flatten_test = np.load("features/X_flatten_test.npy")
    run_full_pipeline("Flatten", X_flatten_train, y_train_mn, X_flatten_test, y_test_mn, num_classes=10)

if __name__ == "__main__":
    main()