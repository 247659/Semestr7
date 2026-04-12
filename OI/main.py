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

def voronoi_diagram(X, clusters):
    vor = Voronoi(X)
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', show_points=False)
    plt.scatter(X[:, 0], X[:, 1], c=clusters)
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



