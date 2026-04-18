import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
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
    print("\n" + "=" * 40 + "\nEKSPERYMENT 1: SVM i MLP (Best Train Acc)\n" + "=" * 40)
    C_values = [0.1, 0.5, 1.0, 4, 10, 25, 50, 100]
    hidden_layer_sizes = [10, 20, 35, 50, 75, 100]

    for dataset in datasets:
        print(f"\n--- Zbiór: {dataset} ---")
        X, y = load(dataset)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

        # 1. SVM
        # for kernel in ['linear', 'rbf']:
        #     best_svm, best_acc, best_C = None, -1, None
        #     for C in C_values:
        #         svc = SVC(kernel=kernel, C=C)
        #         svc.fit(X_train, y_train)
        #         acc = accuracy_score(y_train, svc.predict(X_train))
        #         if acc > best_acc:
        #             best_acc, best_C, best_svm = acc, C, svc
        #
        #         plot_decision(svc, X_train, X_test, y_train, y_test,
        #                       f"SVM ({kernel}) C={C}, acc={acc:.2f}, dane={dataset}")
        #
        #     print(f"Najlepszy SVM ({kernel}): C={best_C}, Train Acc={best_acc:.2f}")

        # 2. MLP
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
                          f"MLP ({activation}) Layers={size}\nTrain Acc={acc:.2f}")

            print(f"Najlepszy MLP ({activation}): layers={best_hidden}, Train Acc={best_acc:.2f}")

def run_experiment_2_or_5(datasets, train_size, test_size, page5 = False):
    print("\n" + "=" * 40 + "\nEKSPERYMENT 2: K-NN (n_neighbors)\n" + "=" * 40)
    neighbors_range = range(1, 21)  # od 1 do 20

    for dataset in datasets:
        print(f"\n--- Zbiór: {dataset} ---")
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

        # Wykres zmian Accuracy
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

        # Rysowanie granic i macierzy dla wybranych K
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

def run_experiment_3(datasets):
    print("\n" + "=" * 40 + "\nEKSPERYMENT 3: SVM (Parametr C)\n" + "=" * 40)
    C_values = np.logspace(-1, 2, 15)  # 10 wartości od 0.01 do 100

    for dataset in datasets:
        print(f"\n--- Zbiór: {dataset} ---")
        X, y = load(dataset)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

        train_acc, test_acc = [], []

        for C in C_values:
            svc = SVC(kernel='rbf', C=C)  # Używamy elastycznego rbf
            svc.fit(X_train, y_train)
            train_acc.append(accuracy_score(y_train, svc.predict(X_train)))
            test_acc.append(accuracy_score(y_test, svc.predict(X_test)))

        c_min = C_values[0]
        c_max = C_values[-1]
        c_best = C_values[np.argmax(test_acc)]  # Najlepsze na teście

        # Wykres zmian Accuracy (skala logarytmiczna)
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

        # Rysowanie granic i macierzy dla wybranych C
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


def run_experiment_4(datasets):
    print("\n" + "=" * 40 + "\nEKSPERYMENT 4: MLP (Liczba neuronów w warstwie ukrytej)\n" + "=" * 40)

    neurons_range = [5, 10, 20, 50, 75, 100]

    for dataset in datasets:
        print(f"\n--- Zbiór: {dataset} ---")
        X, y = load(dataset)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

        train_acc, test_acc = [], []

        for neurons in neurons_range:
            print(f"Trenowanie MLP dla {neurons} neuronów (to może potrwać)...")

            # Używamy activation='relu' bo jest bardziej elastyczne (wniosek z Exp 1)
            mlp = MLPClassifier(
                hidden_layer_sizes=(neurons,), activation='relu',
                max_iter=100000, tol=0, n_iter_no_change=100000, solver='sgd', random_state=42
            )

            mlp.fit(X_train, y_train)
            train_acc.append(accuracy_score(y_train, mlp.predict(X_train)))
            test_acc.append(accuracy_score(y_test, mlp.predict(X_test)))

        n_min = neurons_range[0]
        n_max = neurons_range[-1]
        n_best = neurons_range[np.argmax(test_acc)]  # Najlepsze na teście

        # --- WYKRES 1: Zmiana Accuracy ---
        plt.figure(figsize=(8, 4))
        plt.plot(neurons_range, train_acc, label='Accuracy Treningowe')
        plt.plot(neurons_range, test_acc, label='Accuracy Testowe', linewidth=2)
        plt.xlabel('Liczba neuronów w warstwie ukrytej')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f"Zmiana Accuracy MLP - {dataset}")
        plt.grid(True)
        plt.show()

        # --- WYKRESY 2: Granice i Macierze ---
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

if __name__ == "__main__":
    datasets_all = ["2_1.csv", "2_2.csv", "2_3.csv"]
    datasets_subset = ["2_2.csv", "2_3.csv"]

    # run_experiment_1(datasets_all)
    # run_experiment_2_or_5(datasets_subset, 0.8, 0.2)
    # run_experiment_3(datasets_subset)
    # run_experiment_4(datasets_subset)
    run_experiment_2_or_5(datasets_subset, 0.2, 0.2, True)