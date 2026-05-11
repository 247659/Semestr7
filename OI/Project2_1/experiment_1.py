import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, homogeneity_score, completeness_score
from visualize import voronoi_diagram


def kmeans(X, k=10):
    model = KMeans(n_clusters=k, random_state=42).fit(X)
    return model.labels_

def dbscan(X, eps):
    model = DBSCAN(eps=eps, min_samples=2).fit(X)
    return model.labels_

def get_most_confused_classes(X, y):
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

def analyze(X_train, y_train, draw_voronoi=False):
    X_train_scaled = StandardScaler().fit_transform(X_train)

    if draw_voronoi and X_train.shape[1] == 2:
        voronoi_diagram(X_train, y_train)

    clusters = kmeans(X_train_scaled)
    silhouette = silhouette_score(X_train_scaled, y_train)
    ARI = adjusted_rand_score(y_train, clusters)
    homogeneity = homogeneity_score(y_train, clusters)
    completeness = completeness_score(y_train, clusters)

    print(
        f"Cecha: {X_train.shape[1]}D | Silhouette: {silhouette:.4f}, ARI: {ARI:.4f}, Homogeneity: {homogeneity:.4f}, Completeness: {completeness:.4f}")

    c1, c2 = get_most_confused_classes(X_train_scaled, y_train)
    print(f"\n[!] Najbardziej mylone klasy (najbliższe w przestrzeni cech): Cyfra {c1} oraz Cyfra {c2}")

    return silhouette, ARI, homogeneity, completeness


def plot_metrics_chart(features_list, silhouette_list, ari_list, hom_list, comp_list):
    plt.figure(figsize=(10, 6))

    plt.plot(features_list, silhouette_list, marker='o', label='Silhouette', linewidth=2)
    plt.plot(features_list, ari_list, marker='o', label='ARI', linewidth=2)
    plt.plot(features_list, hom_list, marker='o', label='Homogeneity', linewidth=2)
    plt.plot(features_list, comp_list, marker='o', label='Completeness', linewidth=2)

    plt.xscale('log')
    plt.xticks(features_list, [str(f) for f in features_list])

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.title('Wpływ liczby cech na metryki separowalności')
    plt.xlabel('Liczba cech')
    plt.ylabel('Wartość wskaźnika')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def first():
    y_train = np.load("features/y_train.npy")

    features_counts = [2, 20, 784]
    sil_scores = []
    ari_scores = []
    hom_scores = []
    comp_scores = []

    X_train_method1_1 = np.load("features/X1_train.npy")
    s, a, h, c = analyze(X_train_method1_1, y_train, draw_voronoi=True)
    sil_scores.append(s)
    ari_scores.append(a)
    hom_scores.append(h)
    comp_scores.append(c)

    X_train_method1_2 = np.load("features/X_pca_train.npy")
    s, a, h, c = analyze(X_train_method1_2, y_train, draw_voronoi=False)
    sil_scores.append(s)
    ari_scores.append(a)
    hom_scores.append(h)
    comp_scores.append(c)

    X_train_flatten = np.load("features/X_flat_train.npy")
    s, a, h, c = analyze(X_train_flatten, y_train, draw_voronoi=False)
    sil_scores.append(s)
    ari_scores.append(a)
    hom_scores.append(h)
    comp_scores.append(c)

    plot_metrics_chart(features_counts, sil_scores, ari_scores, hom_scores, comp_scores)

def second():
    y_train = np.load("features/y_train.npy")

    features_counts = [2, 20, 784]
    sil_scores = []
    ari_scores = []
    hom_scores = []
    comp_scores = []

    X_train_method2_1 = np.load("features/X2_train.npy")
    s, a, h, c = analyze(X_train_method2_1, y_train, draw_voronoi=True)
    sil_scores.append(s)
    ari_scores.append(a)
    hom_scores.append(h)
    comp_scores.append(c)

    X_train_method2_2 = np.load("features/X3_train.npy")
    s, a, h, c = analyze(X_train_method2_2, y_train, draw_voronoi=False)
    sil_scores.append(s)
    ari_scores.append(a)
    hom_scores.append(h)
    comp_scores.append(c)

    X_train_flatten = np.load("features/X_flat_train.npy")
    s, a, h, c = analyze(X_train_flatten, y_train, draw_voronoi=False)
    sil_scores.append(s)
    ari_scores.append(a)
    hom_scores.append(h)
    comp_scores.append(c)

    plot_metrics_chart(features_counts, sil_scores, ari_scores, hom_scores, comp_scores)

if __name__ == "__main__":
    # first()
    second()