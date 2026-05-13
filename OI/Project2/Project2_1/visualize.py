import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.lines import Line2D

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


def voronoi_diagram(X, y, num_samples=500, title="Diagram Woronoja"):
    if len(X) > num_samples:
        indices = np.random.choice(len(X), num_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y

    vor = Voronoi(X_sample)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.cm.get_cmap('tab10', 10)

    for i, region in enumerate(regions):
        polygon = vertices[region]
        color = cmap(y_sample[i])
        ax.fill(*zip(*polygon), alpha=0.3, color=color, edgecolor='black', linewidth=0.5)

    for class_id in np.unique(y_sample):
        idx = (y_sample == class_id)
        ax.scatter(X_sample[idx, 0], X_sample[idx, 1],
                   c=[cmap(class_id)], edgecolor='black', s=25,
                   zorder=3, label=f"Cyfra {class_id}")

    margin_x = (X_sample[:, 0].max() - X_sample[:, 0].min()) * 0.1
    margin_y = (X_sample[:, 1].max() - X_sample[:, 1].min()) * 0.1
    ax.set_xlim(X_sample[:, 0].min() - margin_x, X_sample[:, 0].max() + margin_x)
    ax.set_ylim(X_sample[:, 1].min() - margin_y, X_sample[:, 1].max() + margin_y)

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_decision(model, X_train, X_test, y_train, y_test, title):

    fig, ax = plt.subplots(figsize=(10, 8))

    DecisionBoundaryDisplay.from_estimator(
        model,
        X_train,
        response_method="predict",
        alpha=0.35,
        ax=ax,
        eps=0.5
    )
    scatter_train = ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="tab10",
        edgecolor='k',
        s=20
    )
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="tab10",
        marker='^',
        s=25
    )
    colors = scatter_train.cmap(
        np.linspace(0, 1, 10)
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label=f'Cyfra {i}',
            markerfacecolor=colors[i],
            markersize=10
        )
        for i in range(10)
    ]

    ax.legend(
        handles=legend_elements,
        title="Klasy",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    ax.set_title(title)
    ax.set_xlabel("Suma pikseli")
    ax.set_ylabel("Balans pikseli w pionie")

    plt.tight_layout()
    plt.show()