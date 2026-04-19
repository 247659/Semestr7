import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix

def load(path):
    data = np.loadtxt(path, delimiter=';')
    X = data[:, :2]
    y = data[:, 2]
    return X, y

def plot_decision(model, X_train, X_test, y_train, y_test, title):
    fig, ax = plt.subplots()

    DecisionBoundaryDisplay.from_estimator(
        model, X_train, response_method="predict", alpha=0.5, ax=ax, eps=0.5
    )

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', label='Train')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='^', edgecolor='red', label='Test')

    ax.set_title(title)
    ax.legend()
    plt.show()

def plot_confusion(model, X_test, y_test, title):
    fig, ax = plt.subplots()

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(title)

    plt.show()

def run_experiment_1(datasets):
    C_values = [0.1, 0.5, 1.0, 4, 10, 25, 50, 100]
    hidden_layer_sizes = [10, 20, 35, 50, 75, 100]

    for dataset in datasets:
        X, y = load(dataset)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

        for kernel in ['linear', 'rbf']:
            best_svm, best_acc, best_C = None, -1, None
            for C in C_values:
                svc = SVC(kernel=kernel, C=C, random_state=42)
                svc.fit(X_train, y_train)
                acc = accuracy_score(y_train, svc.predict(X_train))
                if acc > best_acc:
                    best_acc, best_C, best_svm = acc, C, svc

                plot_decision(svc, X_train, X_test, y_train, y_test,
                              f"SVM ({kernel}) C={C}, acc={acc:.2f}, dane={dataset}")

            print(f"Najlepszy SVM ({kernel}): C={best_C}, Train Acc={best_acc:.2f}")

        for activation in ['identity', 'relu']:
            best_mlp, best_acc, best_hidden = None, -1, None
            print(f"Trenowanie MLP ({activation})... to może potrwać.")
            for size in hidden_layer_sizes:
                mlp = MLPClassifier(activation=activation, hidden_layer_sizes=size,
                                    max_iter=100000, tol=0, n_iter_no_change=100000, solver='sgd', random_state=42)
                mlp.fit(X_train, y_train)
                acc = accuracy_score(y_train, mlp.predict(X_train))
                if acc > best_acc:
                    best_acc, best_hidden, best_mlp = acc, size, mlp

                plot_decision(mlp, X_train, X_test, y_train, y_test,
                          f"MLP ({activation}) Layers={size} acc={acc:.2f}, dane={dataset}")

            print(f"Najlepszy MLP ({activation}): layers={best_hidden}, Train Acc={best_acc:.2f}")

def run_experiment_2_or_5(datasets, train_size, test_size, page5 = True):
    neighbors_range = range(1, 21)

    for dataset in datasets:
        X, y = load(dataset)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)

        train_acc, test_acc = [], []

        for k in neighbors_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            train_acc.append(accuracy_score(y_train, knn.predict(X_train)))
            test_acc.append(accuracy_score(y_test, knn.predict(X_test)))

        k_min = neighbors_range[0]
        k_max = neighbors_range[-1]
        k_best = neighbors_range[np.argmax(test_acc)]

        plt.figure(figsize=(8, 4))
        plt.plot(neighbors_range, train_acc, label='Accuracy Treningowe')
        plt.plot(neighbors_range, test_acc, label='Accuracy Testowe', linewidth=2)
        plt.xlabel('n_neighbors')
        plt.ylabel('Accuracy')
        plt.xticks(np.arange(min(neighbors_range), max(neighbors_range) + 1, 2))
        plt.legend()
        plt.title(f"Zmiana Accuracy K-NN - {dataset}")
        plt.grid(True)
        plt.show()

        for i, k in enumerate([k_min, k_best, k_max]):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            current_test_acc = accuracy_score(y_test, knn.predict(X_test))
            suffix = ""
            if (page5):
                if i == 0:
                    suffix = "min"
                elif i == 1:
                    suffix = "best"
                else:
                    suffix = "max"

            plot_decision(knn, X_train, X_test, y_train, y_test, f"K-NN Granica decyzyjna (K={k} {suffix})\n accuracy = {current_test_acc:.2f} ")
            plot_confusion(knn, X_test, y_test, f"K-NN Macierz pomyłek (K={k})")

def run_experiment_3_and_6(datasets, train_size, test_size):
    C_values = np.logspace(-1, 2, 15)

    for dataset in datasets:
        X, y = load(dataset)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)

        train_acc, test_acc = [], []

        for C in C_values:
            svc = SVC(kernel='rbf', C=C, random_state=42)
            svc.fit(X_train, y_train)
            train_acc.append(accuracy_score(y_train, svc.predict(X_train)))
            test_acc.append(accuracy_score(y_test, svc.predict(X_test)))

        c_min = C_values[0]
        c_max = C_values[-1]
        c_best = C_values[np.argmax(test_acc)]

        plt.figure(figsize=(8, 4))
        plt.plot(C_values, train_acc, label='Accuracy Treningowe')
        plt.plot(C_values, test_acc, label='Accuracy Testowe', linewidth=2)
        plt.xscale('log')
        plt.xlabel('log(C)')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f"Zmiana Accuracy SVM - {dataset}")
        plt.grid(True)
        plt.show()

        for i, c in enumerate([c_min, c_best, c_max]):
            svc = SVC(kernel='rbf', C=c)
            svc.fit(X_train, y_train)
            current_test_acc = accuracy_score(y_test, svc.predict(X_test))
            if i == 0:
                suffix = "min"
            elif i == 1:
                suffix = "best"
            else:
                suffix = "max"
            plot_decision(svc, X_train, X_test, y_train, y_test, f"SVM Granica decyzyjna (C={c:.2f} ({suffix}))\n accuracy = {current_test_acc:.2f}")
            plot_confusion(svc, X_test, y_test, f"SVM Macierz pomyłek (C={c:.2f}) dla {suffix}")


def run_experiment_4_and_7(datasets, train_size, test_size):
    neurons_range = [5, 10, 20, 50, 75, 100]

    for dataset in datasets:
        print(f"\n--- Zbiór: {dataset} ---")
        X, y = load(dataset)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)

        train_acc, test_acc = [], []

        for neurons in neurons_range:

            mlp = MLPClassifier(
                hidden_layer_sizes=(neurons,), activation='relu',
                max_iter=100000, tol=0, n_iter_no_change=100000, solver='sgd', random_state=42
            )

            mlp.fit(X_train, y_train)
            train_acc.append(accuracy_score(y_train, mlp.predict(X_train)))
            test_acc.append(accuracy_score(y_test, mlp.predict(X_test)))

        n_min = neurons_range[0]
        n_max = neurons_range[-1]
        n_best = neurons_range[np.argmax(test_acc)]

        plt.figure(figsize=(8, 4))
        plt.plot(neurons_range, train_acc, label='Accuracy Treningowe')
        plt.plot(neurons_range, test_acc, label='Accuracy Testowe', linewidth=2)
        plt.xlabel('Liczba neuronów w warstwie ukrytej')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f"Zmiana Accuracy MLP - {dataset}")
        plt.grid(True)
        plt.show()

        for i, neurons in enumerate([n_min, n_best, n_max]):
            mlp = MLPClassifier(
                hidden_layer_sizes=(neurons,), activation='relu',
                max_iter=100000, tol=0, n_iter_no_change=100000, solver='sgd', random_state=42
            )
            mlp.fit(X_train, y_train)

            current_test_acc = accuracy_score(y_test, mlp.predict(X_test))

            if i == 0:
                suffix = "min"
            elif i == 1:
                suffix = "best"
            else:
                suffix = "max"

            plot_decision(
                mlp, X_train, X_test, y_train, y_test,
                f"MLP Granica decyzyjna: {neurons} neuronów ({suffix})\nTest Accuracy = {current_test_acc:.2f}"
            )
            plot_confusion(mlp, X_test, y_test, f"MLP Macierz pomyłek: {neurons} neuronów ({suffix})")

def run_experiment_8(dataset = "2_3.csv"):
    X, y = load(dataset)
    X = StandardScaler().fit_transform(X)

    neurons_count = 75
    max_epochs = 100000

    train_sizes = [0.8, 0.2]

    for train_size in train_sizes:
        print(f"\n>> WARIANT: Wielkość zbioru treningowego = {train_size}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, test_size=0.2, random_state=42
        )

        print(f"Rysowanie wykresów i granic decyzyjnych (pojedynczy bieg)...")
        mlp_single = MLPClassifier(
            hidden_layer_sizes=(neurons_count,), activation='relu',
            solver='sgd', random_state=42
        )

        classes = np.unique(y_train)
        train_acc_history = []
        test_acc_history = []
        models_history = []
        best_epoch_acc = -1
        best_epoch_num = 0
        best_model = None

        for epoch in range(max_epochs):
            mlp_single.partial_fit(X_train, y_train, classes=classes)

            acc_train = accuracy_score(y_train, mlp_single.predict(X_train))
            acc_test = accuracy_score(y_test, mlp_single.predict(X_test))

            train_acc_history.append(acc_train)
            test_acc_history.append(acc_test)

            if epoch == 0:
                from copy import deepcopy
                models_history.append(('Pierwsza epoka (1)', deepcopy(mlp_single)))

            if acc_test > best_epoch_acc:
                best_epoch_acc = acc_test
                best_epoch_num = epoch
                best_model = deepcopy(mlp_single)

        models_history.append((f'Najlepsza epoka ({best_epoch_num})', best_model))
        models_history.append((f'Ostatnia epoka ({max_epochs - 1})', deepcopy(mlp_single)))

        plt.figure(figsize=(8, 4))
        plt.plot(range(max_epochs), train_acc_history, label='Accuracy Treningowe')
        plt.plot(range(max_epochs), test_acc_history, label='Accuracy Testowe')
        plt.xlabel('Numer Epoki')
        plt.ylabel('Accuracy')
        plt.title(f"Postęp nauki MLP (train_size={train_size})")
        plt.legend()
        plt.grid(True)
        plt.show()

        for title, model in models_history:
            current_test_acc = accuracy_score(y_test, model.predict(X_test))
            plot_decision(
                model, X_train, X_test, y_train, y_test,
                f"MLP (train={train_size}) - {title}\n accuracy = {current_test_acc:.2f}"
            )

        print(f"Uruchamianie 10 biegów z różnymi inicjalizacjami wag...")
        results_table = []

        for run_id in range(10):
            mlp = MLPClassifier(
                hidden_layer_sizes=(neurons_count,), activation='relu',
                solver='sgd', random_state=run_id
            )

            best_acc_test = -1
            best_ep_num = 0
            first_ep_train, first_ep_test = 0, 0
            last_ep_train, last_ep_test = 0, 0
            best_ep_train = 0

            for epoch in range(max_epochs):
                mlp.partial_fit(X_train, y_train, classes=classes)
                acc_tr = accuracy_score(y_train, mlp.predict(X_train))
                acc_te = accuracy_score(y_test, mlp.predict(X_test))

                if epoch == 0:
                    first_ep_train, first_ep_test = acc_tr, acc_te

                if acc_te > best_acc_test:
                    best_acc_test = acc_te
                    best_ep_num = epoch
                    best_ep_train = acc_tr

                if epoch == max_epochs - 1:
                    last_ep_train, last_ep_test = acc_tr, acc_te

            results_table.append({
                'Bieg': run_id,
                'Start (Train)': f"{first_ep_train:.2f}",
                'Start (Test)': f"{first_ep_test:.2f}",
                'Best (Epoka)': best_ep_num,
                'Best (Train)': f"{best_ep_train:.2f}",
                'Best (Test)': f"{best_acc_test:.2f}",
                'Koniec (Train)': f"{last_ep_train:.2f}",
                'Koniec (Test)': f"{last_ep_test:.2f}"
            })

        df_results = pd.DataFrame(results_table)
        print(f"\nTABELA WYNIKÓW DLA train_size={train_size}:")
        print(df_results.to_string(index=False))


def run_experiment_real_data():

    datasets = {
        "Iris (4 cechy, 3 klasy)": load_iris(),
        "Wine (13 cech, 3 klasy)": load_wine()
    }

    results_table = []

    for name, data in datasets.items():
        print(f"\nAnaliza zbioru: {name}")
        X, y = data.data, data.target

        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8, test_size=0.2, random_state=42)

        knn = KNeighborsClassifier(n_neighbors=3)

        svm = SVC(kernel='rbf', C=2.0, random_state=42)

        mlp = MLPClassifier(hidden_layer_sizes=(75,), activation='relu',
                            max_iter=100000, tol=0, n_iter_no_change=100000, solver='sgd', random_state=42)

        models = {
            "K-NN (k=3)": knn,
            "SVM (RBF, C=2)": svm,
            "MLP (ReLU, 75)": mlp
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

            results_table.append({
                "Zbiór": name.split()[0],
                "Klasyfikator": model_name,
                "Accuracy": f"{acc:.4f}",
                "Precision": f"{prec:.4f}",
                "Recall": f"{rec:.4f}",
                "F1-Score": f"{f1:.4f}"
            })

            plot_confusion(model, X_test, y_test, f"{name} accuracy = {acc:.4f}\nMacierz pomyłek: {model_name}")

    df_results = pd.DataFrame(results_table)
    print("\nTABELA WYNIKÓW (ZBIORY RZECZYWISTE):")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    datasets_all = ["2_1.csv", "2_2.csv", "2_3.csv"]
    datasets_subset = ["2_2.csv", "2_3.csv"]

    run_experiment_1(datasets_all)
    run_experiment_2_or_5(datasets_subset, 0.8, 0.2)
    run_experiment_3_and_6(datasets_subset, 0.8, 0.2)
    run_experiment_4_and_7(datasets_subset, 0.8, 0.2)
    run_experiment_2_or_5(datasets_subset, 0.2, 0.2, True)
    run_experiment_3_and_6(datasets_subset, 0.2, 0.2)
    run_experiment_4_and_7(datasets_subset, 0.2, 0.2)
    run_experiment_8()
    run_experiment_real_data()