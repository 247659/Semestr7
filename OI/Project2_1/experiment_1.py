import numpy as np
from sklearn.preprocessing import StandardScaler

from visualize import voronoi_diagram
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.cluster import KMeans

def kmeans(X, k = 10):
    kmeans = KMeans(n_clusters=k).fit(X)
    clusters = kmeans.labels_
    return clusters

# def dbscan(X, eps, voronoi, y=None):
#     dbscan = DBSCAN(eps=eps, min_samples=1).fit(X)
#     clusters = dbscan.labels_
#     unique = np.unique(clusters)
#     if voronoi:
#         voronoi_diagram(X, clusters, y)
#
#     if 1 < len(unique) < len(X):
#         score = silhouette_score(X, clusters)
#     else:
#         score = 0
#
#     return score, len(unique), clusters

##########################################

X_train_method1_1 = np.load("features/X1_train.npy")
y_train = np.load("features/y_train.npy")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_method1_1)

voronoi_diagram(X_train_scaled, y_train)
clusters = kmeans(X_train_scaled)
silhouette = silhouette_score(X_train_scaled, y_train)
ARI = adjusted_rand_score(X_train_scaled, clusters)
homogeneity = homogeneity_score(X_train_scaled, clusters)
completeness = completeness_score(X_train_scaled, clusters)

print(f"Silhouette score: {silhouette}, ARI: {ARI}, homogeneity: {homogeneity}, completeness: {completeness}")

############################################

X_train_method1_2 = np.load("features/X_pca_train.npy")

X_train_scaled = scaler.fit_transform(X_train_method1_2)

voronoi_diagram(X_train_scaled, y_train)
clusters = kmeans(X_train_scaled)
silhouette = silhouette_score(X_train_scaled, y_train)
ARI = adjusted_rand_score(X_train_scaled, clusters)
homogeneity = homogeneity_score(X_train_scaled, clusters)
completeness = completeness_score(X_train_scaled, clusters)

print(f"Silhouette score: {silhouette}, ARI: {ARI}, homogeneity: {homogeneity}, completeness: {completeness}")


############################################

X_train_flatten = np.load("features/X_flat_test.npy")
X_train_scaled = scaler.fit_transform(X_train_flatten)

voronoi_diagram(X_train_scaled, y_train)
clusters = kmeans(X_train_scaled)
silhouette = silhouette_score(X_train_scaled, y_train)
ARI = adjusted_rand_score(X_train_scaled, clusters)
homogeneity = homogeneity_score(X_train_scaled, clusters)
completeness = completeness_score(X_train_scaled, clusters)

print(f"Silhouette score: {silhouette}, ARI: {ARI}, homogeneity: {homogeneity}, completeness: {completeness}")
