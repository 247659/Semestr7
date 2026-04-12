import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score

def load(path):
    data = np.loadtxt(path, delimiter=';')
    X = data[:, :2]
    y = data[:, 2]
    return X, y

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Funkcja pomocnicza, która "zamyka" nieskończone regiony Voronoia,
    rozciągając je daleko poza główny zbiór punktów.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Wymaga danych 2D")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points) * 2

    # Mapowanie krawędzi (ridges) dla każdego punktu
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Rekonstrukcja nieskończonych regionów
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            # Obliczanie kierunku wektora uciekającego w nieskończoność
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # wektor normalny

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Sortowanie wierzchołków przeciwnie do ruchu wskazówek zegara
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def voronoi_diagram(X, clusters):
    vor = Voronoi(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.tab10

    # Używamy naszej nowej funkcji, aby uzyskać wszystkie regiony (zamknięte)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # Rysujemy i wypełniamy regiony
    for i, region in enumerate(regions):
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.3,
                color=cmap(clusters[i] % 10),
                edgecolor='gray', linewidth=1)

    # Rysujemy punkty
    point_colors = [cmap(c % 10) for c in clusters]
    ax.scatter(X[:, 0], X[:, 1], c=point_colors, edgecolor='black', zorder=3)

    # KLUCZOWY KROK: Ustawiamy sztywne ramy wykresu.
    # To właśnie to robi faktyczny "clipping" i ukrywa te wyciągnięte daleko wierzchołki.
    margin_x = (X[:, 0].max() - X[:, 0].min()) * 0.1
    margin_y = (X[:, 1].max() - X[:, 1].min()) * 0.1
    ax.set_xlim(X[:, 0].min() - margin_x, X[:, 0].max() + margin_x)
    ax.set_ylim(X[:, 1].min() - margin_y, X[:, 1].max() + margin_y)

    plt.show()

def kmeans(X, k):
    kmeans = KMeans(n_clusters=k).fit(X)
    clusters = kmeans.labels_
    voronoi_diagram(X, clusters)
    score = silhouette_score(X, clusters)
    return score

def dbscan(X, eps, min_samples=1):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    clusters = dbscan.labels_
    unique = np.unique(clusters)
    voronoi_diagram(X, clusters)

    if len(unique) > 1:
        score = silhouette_score(X, clusters)
    else:
        score = 0

    return score, len(unique)

if __name__ == "__main__":
    for i in range (1, 4):

        X, y = load(f"./3_{i}.csv")

        X = StandardScaler().fit_transform(X)

        # silhouette_score_kmeans = []
        # clusters_values = range(2, 10)
        # for k in clusters_values:
        #     score = kmeans(X, k)
        #     silhouette_score_kmeans.append(score)
        #     print(f"k={k}, silhouette={score}")
        #
        # plt.plot(clusters_values, silhouette_score_kmeans, label='Silhouette score')
        # plt.xlabel("Liczba klastrów")
        # plt.ylabel("Silhouette score")
        # plt.legend()
        # plt.show()

        eps = np.linspace(0.01, 2.00, 9)
        silhouette_score_dbscan = []
        clusters_counts = []
        for ep in eps:
            score, clusters = dbscan(X, ep)
            silhouette_score_dbscan.append(score)
            clusters_counts.append(clusters)
            print(f"eps={ep:.2f}, clusters={clusters}, score={score}")

        plt.plot(eps, silhouette_score_dbscan, label='Silhouette score')

        for i, eps in enumerate(eps):
            plt.text(eps, silhouette_score_dbscan[i], str(clusters_counts[i]))

        plt.xlabel("eps")
        plt.ylabel("Silhouette score")

        plt.legend()
        plt.show()



