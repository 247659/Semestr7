import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score
from sklearn.preprocessing import StandardScaler
import warnings
import torch

# Importujemy funkcję z Twojego pliku visualize.py
from visualize import voronoi_diagram

# Ignorujemy ostrzeżenia o zmianach w bibliotekach
warnings.filterwarnings('ignore')


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


def get_most_confused_classes(X, y):
    """
    Znajduje dwie najbardziej mylone klasy poprzez obliczenie najmniejszej
    odległości euklidesowej pomiędzy ich średnimi (środkami ciężkości).
    """
    centroids = {}
    classes = np.unique(y)

    for c in classes:
        centroids[c] = np.mean(X[y == c], axis=0)

    min_dist = float('inf')
    confused_pair = (-1, -1)

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            c1, c2 = classes[i], classes[j]
            dist = np.linalg.norm(centroids[c1] - centroids[c2])
            if dist < min_dist:
                min_dist = dist
                confused_pair = (c1, c2)

    return confused_pair


def run_experiment(X, y, feature_name):
    set_seed(42)
    print(f"\n{'=' * 60}")
    print(f"EKSPERYMENT: {feature_name} (Wymiary: {X.shape[1]})")
    print(f"{'=' * 60}")

    X_sub = X
    y_sub = y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    sil_true = silhouette_score(X_scaled, y_sub)
    print(f"[Prawdziwe etykiety] Wskaźnik Silhouette (ogólna separowalność): {sil_true:.4f}")

    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X_scaled)

    sil_kmeans = silhouette_score(X_scaled, y_kmeans)
    ari_kmeans = adjusted_rand_score(y_sub, y_kmeans)
    hom_kmeans = homogeneity_score(y_sub, y_kmeans)
    comp_kmeans = completeness_score(y_sub, y_kmeans)

    print("\n--- Wyniki dla KMeans (k=10) ---")
    print(f"Silhouette Score: {sil_kmeans:.4f}")
    print(f"Adjusted Rand Index (ARI): {ari_kmeans:.4f}")
    print(f"Homogeneity: {hom_kmeans:.4f}")
    print(f"Completeness: {comp_kmeans:.4f}")

    c1, c2 = get_most_confused_classes(X_scaled, y_sub)
    print(f"\n[!] Najbardziej mylone klasy (najbliższe w przestrzeni cech): Cyfra {c1} oraz Cyfra {c2}")

    # Zwracamy słownik z metrykami do późniejszego wyrysowania
    return {
        "Silhouette": sil_true,
        "ARI": ari_kmeans,
        "Homogeneity": hom_kmeans,
        "Completeness": comp_kmeans
    }


def plot_metrics_vs_features(feature_counts, metrics_list):
    """
    Rysuje wykres liniowy pokazujący zmianę wskaźników w zależności od liczby cech.
    """
    plt.figure(figsize=(10, 6))

    # Wyciągamy nazwy metryk ze słownika
    metric_names = metrics_list[0].keys()

    # Rysujemy linię dla każdej metryki
    for name in metric_names:
        values = [m[name] for m in metrics_list]
        plt.plot(feature_counts, values, marker='o', linewidth=2, markersize=8, label=name)

    # Ustawienia osi X na skalę logarytmiczną ze względu na skok (2 -> 20 -> 784)
    plt.xscale('log')
    plt.xticks(feature_counts, labels=[str(x) for x in feature_counts])

    plt.xlabel('Liczba cech')
    plt.ylabel('Wartość wskaźnika')
    plt.title('Wpływ liczby cech na metryki separowalności')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ograniczamy oś Y by lepiej pokazywała zakres wartości metryk (-1 do 1 zazwyczaj, choć ARI/Homogeneity od 0)
    plt.ylim(min(-0.2, min([min(m.values()) for m in metrics_list]) - 0.1), 1.05)

    plt.tight_layout()
    plt.show()


def second_set():
    try:
        print("Wczytywanie danych...")
        X1_train = np.load("features/X2_train.npy")  # 2 cechy
        X2_train = np.load("features/X3_train.npy")  # 6 cech
        X3_train = np.load("features/X_flat_train.npy")  # 784 cechy
        y_train = np.load("features/y_train.npy")
    except FileNotFoundError:
        print("Błąd: Nie znaleziono plików .npy. Uruchom najpierw pipeline z 'features.py'!")
        return

    # Uruchamiamy eksperymenty i zbieramy ich wyniki
    metrics_2 = run_experiment(X1_train, y_train, "Metoda 2_1: 2 Cechy (Ilość pętli, Wariancja szerokości)")
    metrics_20 = run_experiment(X2_train, y_train, "Metoda 2_2: 6 Cech")
    metrics_784 = run_experiment(X3_train, y_train, "Metoda: 784 Cechy (Flatten)")

    # ---------------------------------------------------------
    # Nowy krok: Wykres porównawczy
    # ---------------------------------------------------------
    print("\nGenerowanie wykresu porównawczego metryk...")
    feature_counts = [X1_train.shape[1], X2_train.shape[1], X3_train.shape[1]]
    all_metrics = [metrics_2, metrics_20, metrics_784]
    plot_metrics_vs_features(feature_counts, all_metrics)

    # ---------------------------------------------------------
    # Generowanie Diagramów Woronoja
    # ---------------------------------------------------------
    print("\nGenerowanie Diagramów Woronoja dla przestrzeni dwuwymiarowych...")
    voronoi_diagram(X1_train, y_train, num_samples=100, title="Diagram Woronoja - 2 Cechy (Ilość pętli, Wariacja szerokości)")

def first_set():
    try:
        print("Wczytywanie danych...")
        X1_train = np.load("features/X1_train.npy")  # 2 cechy
        X2_train = np.load("features/X_pca_train.npy")  # 20 cech
        X3_train = np.load("features/X_flat_train.npy")  # 784 cechy
        y_train = np.load("features/y_train.npy")
    except FileNotFoundError:
        print("Błąd: Nie znaleziono plików .npy. Uruchom najpierw pipeline z 'features.py'!")
        return

    # Uruchamiamy eksperymenty i zbieramy ich wyniki
    metrics_2 = run_experiment(X1_train, y_train, "Metoda 1_1: 2 Cechy (Suma pikseli, balans)")
    metrics_20 = run_experiment(X2_train, y_train, "Metoda 1_2: 20 Cech (PCA)")
    metrics_784 = run_experiment(X3_train, y_train, "Metoda: 784 Cechy (Flatten)")

    # ---------------------------------------------------------
    # Nowy krok: Wykres porównawczy
    # ---------------------------------------------------------
    print("\nGenerowanie wykresu porównawczego metryk...")
    feature_counts = [X1_train.shape[1], X2_train.shape[1], X3_train.shape[1]]
    all_metrics = [metrics_2, metrics_20, metrics_784]
    plot_metrics_vs_features(feature_counts, all_metrics)

    # ---------------------------------------------------------
    # Generowanie Diagramów Woronoja
    # ---------------------------------------------------------
    print("\nGenerowanie Diagramów Woronoja dla przestrzeni dwuwymiarowych...")
    voronoi_diagram(X1_train, y_train, num_samples=500, title="Diagram Woronoja - 2 Cechy (Suma, Balans)")


if __name__ == "__main__":
    # first_set()
    second_set()