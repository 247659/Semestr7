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
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine

def load(path):
    data = np.loadtxt(path, delimiter=';')
    X = data[:, :2]
    y = data[:, 2]
    return X, y

#FUNKCJA WYGENEROWANA PRZEZ AI
def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Wymaga danych 2D")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points) * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

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

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def voronoi_diagram(X, clusters, y):
    vor = Voronoi(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.tab10

    #FRAGMENT WYGENEROWANY PRZEZ AI
    regions, vertices = voronoi_finite_polygons_2d(vor)

    for i, region in enumerate(regions):
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.3,
                color=cmap(clusters[i] % 10),
                edgecolor='gray', linewidth=1)

    if y is None:
        point_colors = [cmap(c % 10) for c in clusters]
    else:
        point_colors = [cmap(c % 10) for c in y]
    ax.scatter(X[:, 0], X[:, 1], c=point_colors, edgecolor='black', zorder=3)

    #FRAGMENT WYGENEROWANY PRZEZ AI
    margin_x = (X[:, 0].max() - X[:, 0].min()) * 0.1
    margin_y = (X[:, 1].max() - X[:, 1].min()) * 0.1
    ax.set_xlim(X[:, 0].min() - margin_x, X[:, 0].max() + margin_x)
    ax.set_ylim(X[:, 1].min() - margin_y, X[:, 1].max() + margin_y)

    plt.show()

def kmeans(X, k, voronoi, y=None):
    kmeans = KMeans(n_clusters=k).fit(X)
    clusters = kmeans.labels_
    if voronoi:
        voronoi_diagram(X, clusters, y)
    score = silhouette_score(X, clusters)
    return score, clusters

def dbscan(X, eps, voronoi, y=None):
    dbscan = DBSCAN(eps=eps, min_samples=1).fit(X)
    clusters = dbscan.labels_
    unique = np.unique(clusters)
    if voronoi:
        voronoi_diagram(X, clusters, y)

    if 1 < len(unique) < len(X):
        score = silhouette_score(X, clusters)
    else:
        score = 0

    return score, len(unique), clusters

def calculate_adjusted_rand(clusters, true_labels):
    return adjusted_rand_score(true_labels, clusters)

def calculate_homogeneity(clusters, true_labels):
    return homogeneity_score(true_labels, clusters)

def calculate_completeness(clusters, true_labels):
    return completeness_score(true_labels, clusters)

def experiment_1(X, voronoi):
    silhouette_score_kmeans = []
    clusters_values = range(2, 10)
    for k in clusters_values:
        score, _ = kmeans(X, k, voronoi)
        silhouette_score_kmeans.append(score)
        print(f"k={k}, silhouette={score}")

    plt.plot(clusters_values, silhouette_score_kmeans, label='Silhouette score')
    plt.xlabel("Liczba klastrów")
    plt.ylabel("Silhouette score")
    plt.legend()
    plt.show()

def experiment_2(X, voronoi):
    eps = np.linspace(0.01, 2.00, 9)
    silhouette_score_dbscan = []
    clusters_counts = []
    for ep in eps:
        score, clusters_amount, _ = dbscan(X, ep, voronoi)
        silhouette_score_dbscan.append(score)
        clusters_counts.append(clusters_amount)
        print(f"eps={ep:.2f}, clusters={clusters_amount}, score={score}")

    plt.plot(eps, silhouette_score_dbscan, label='Silhouette score')

    for i, eps in enumerate(eps):
        plt.text(eps, silhouette_score_dbscan[i], str(clusters_counts[i]))

    plt.xlabel("eps")
    plt.ylabel("Silhouette score")

    plt.legend()
    plt.show()

def experiment_3(X, y, voronoi):
    adjusted_rand_score_kmeans = []
    homogeneity_score_kmeans = []
    completeness_kmeans = []

    clusters_values = range(2, 10)
    for k in clusters_values:
        _, clusters = kmeans(X, k, voronoi, y)
        adjusted_rand = calculate_adjusted_rand(clusters, y)
        adjusted_rand_score_kmeans.append(adjusted_rand)
        homogeneity_score = calculate_homogeneity(clusters, y)
        homogeneity_score_kmeans.append(homogeneity_score)
        completeness = calculate_completeness(clusters, y)
        completeness_kmeans.append(completeness)
        print(f"k={k} adjusted_rand: {adjusted_rand}, homogenity: {homogeneity_score}, completeness: {completeness}")

    plt.plot(clusters_values, adjusted_rand_score_kmeans, label='Adjust rand')
    plt.plot(clusters_values, homogeneity_score_kmeans, label='Homogenity')
    plt.plot(clusters_values, completeness_kmeans, label='Completeness')
    plt.xlabel("Liczba klastrów")
    plt.legend()
    plt.show()

def experiment_4(X, y, voronoi):
    adjusted_rand_score_kmeans = []
    homogeneity_score_kmeans = []
    completeness_kmeans = []
    eps = np.linspace(0.01, 2.00, 9)
    clusters_counts = []

    for ep in eps:
        _, clusters_amount, clusters = dbscan(X, ep, voronoi, y)
        adjusted_rand = calculate_adjusted_rand(clusters, y)
        adjusted_rand_score_kmeans.append(adjusted_rand)
        homogeneity_score = calculate_homogeneity(clusters, y)
        homogeneity_score_kmeans.append(homogeneity_score)
        completeness = calculate_completeness(clusters, y)
        completeness_kmeans.append(completeness)
        clusters_counts.append(clusters_amount)
        print(f"eps={ep} adjusted_rand: {adjusted_rand}, homogenity: {homogeneity_score}, completeness: {completeness}")

    plt.plot(eps, adjusted_rand_score_kmeans, label='Adjust rand')
    plt.plot(eps, homogeneity_score_kmeans, label='Homogenity')
    plt.plot(eps, completeness_kmeans, label='Completeness')
    for i, eps in enumerate(eps):
        plt.text(eps, adjusted_rand_score_kmeans[i], str(clusters_counts[i]))

    plt.xlabel("eps")
    plt.ylabel("Silhouette score")

    plt.legend()
    plt.show()

def experiment5(X, y):
    experiment_1(X, False)
    experiment_2(X, False)
    experiment_3(X, y.astype(int), False)
    experiment_4(X, y.astype(int), False)

if __name__ == "__main__":
    for i in range (1, 4):

        X, y = load(f"./2_{i}.csv")
        X = StandardScaler().fit_transform(X)

        experiment_1(X, True)
        experiment_2(X, True)
        experiment_3(X, y.astype(int), True)
        experiment_4(X, y.astype(int), True)

    data = load_iris()
    print(data.target_names)
    experiment5(data.data, data.target)
    data = load_wine()
    print(data.target_names)
    experiment5(data.data, data.target)







