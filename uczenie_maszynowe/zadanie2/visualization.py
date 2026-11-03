"""
Moduł odpowiedzialny za wizualizację wytrenowanego drzewa decyzyjnego (tree.png)
przy użyciu biblioteki Matplotlib.

Automatycznie oblicza rozstawienie współrzędnych (x, y) na podstawie liczby liści
w poddrzewach, dzięki czemu żadne węzły nie nachodzą na siebie, a grafika jest
w pełni czytelna i prezentuje:
- warunek podziału (w węzłach wewnętrznych),
- przewidywaną ocenę (w liściach, wyróżnioną kolorystycznie),
- wartość nieczystości (gini/entropy), liczbę próbek (samples) oraz rozkład klas (counts).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from decision_tree import DecisionTreeClassifierCustom, TreeNode


# Paleta kolorów dla klas ocen 0..5 (w liściach) oraz węzłów wewnętrznych
LEAF_COLORS: Dict[int, str] = {
    0: "#f8d7da",  # jasnoczerwony (ocena 0)
    1: "#ffe5d0",  # jasnopomarańczowy (ocena 1)
    2: "#fff3cd",  # jasnożółty (ocena 2)
    3: "#e2f0d9",  # jasnozielony (ocena 3)
    4: "#d1ecf1",  # jasnobłękitny (ocena 4)
    5: "#cce5ff",  # jasnoniebieski (ocena 5)
}
INTERNAL_NODE_COLOR = "#f0f4f8"


def _count_leaves(node: TreeNode) -> int:
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return _count_leaves(node.left) + _count_leaves(node.right)


def _assign_positions(
    node: TreeNode,
    x_min: float,
    x_max: float,
    y: float,
    y_step: float,
    positions: Dict[int, Tuple[float, float]],
) -> None:
    if node is None:
        return
    if node.is_leaf:
        positions[id(node)] = (0.5 * (x_min + x_max), y)
        return

    left_leaves = max(1, _count_leaves(node.left))
    right_leaves = max(1, _count_leaves(node.right))
    total_leaves = left_leaves + right_leaves

    split_x = x_min + (x_max - x_min) * (left_leaves / float(total_leaves))
    positions[id(node)] = (split_x, y)

    _assign_positions(node.left, x_min, split_x, y - y_step, y_step, positions)
    _assign_positions(node.right, split_x, x_max, y - y_step, y_step, positions)


def _format_node_label(node: TreeNode, criterion_name: str = "gini") -> str:
    counts_str = "[" + ",".join(str(int(c)) for c in node.class_counts) + "]"
    if node.is_leaf:
        return (
            f"Predict: Rating {node.predicted_class}\n"
            f"{criterion_name} = {node.impurity:.3f} | n = {node.n_samples}\n"
            f"counts = {counts_str}"
        )
    else:
        if node.is_binary_feature:
            cond_str = f"{node.feature_name} == 0"
        else:
            cond_str = f"{node.feature_name} <= {node.threshold:.2f}"
        return (
            f"{cond_str}\n"
            f"{criterion_name} = {node.impurity:.3f} | n = {node.n_samples}\n"
            f"counts = {counts_str}\n"
            f"majority = {node.predicted_class}"
        )


def plot_decision_tree(
    tree_clf: DecisionTreeClassifierCustom,
    user_id: int,
    output_path: str = "tree.png",
    dpi: int = 250,
) -> None:
    """
    Rysuje wytrenowane drzewo decyzyjne dla wybranego użytkownika i zapisuje do pliku tree.png.
    """
    if tree_clf.root_ is None:
        raise RuntimeError("Drzewo nie posiada korzenia (nie zostało wytrenowane).")

    total_nodes, total_leaves, max_depth = tree_clf.count_nodes_and_leaves()
    fig_width = max(16.0, total_leaves * 2.5)
    fig_height = max(9.0, (max_depth + 1) * 2.3)

    positions: Dict[int, Tuple[float, float]] = {}
    y_step = 1.0 / max(1, max_depth + 1)
    _assign_positions(
        tree_clf.root_,
        x_min=0.04,
        x_max=0.96,
        y=0.90,
        y_step=y_step,
        positions=positions,
    )

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.set_title(
        f"Indywidualne Binarne Drzewo Decyzyjne — Użytkownik ID = {user_id} "
        f"(Węzły: {total_nodes}, Liście: {total_leaves}, Głębokość: {max_depth})",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )

    def draw_subtree(node: TreeNode):
        x_p, y_p = positions[id(node)]

        if not node.is_leaf:
            for child, is_left in [(node.left, True), (node.right, False)]:
                if child is None:
                    continue
                x_c, y_c = positions[id(child)]
                ax.annotate(
                    "",
                    xy=(x_c, y_c + 0.032),
                    xytext=(x_p, y_p - 0.032),
                    arrowprops=dict(
                        arrowstyle="->", color="#495057", lw=1.6, shrinkA=0, shrinkB=0
                    ),
                    zorder=1,
                )
                mid_x = 0.5 * (x_p + x_c)
                mid_y = 0.5 * (y_p + y_c)
                edge_lbl = "True (<=)" if is_left else "False (>)"
                edge_col = "#1b7837" if is_left else "#b2182b"
                ax.text(
                    mid_x,
                    mid_y + 0.01,
                    edge_lbl,
                    fontsize=8.5,
                    fontweight="bold",
                    color=edge_col,
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        fc="white",
                        ec="none",
                        alpha=0.85,
                    ),
                    zorder=2,
                )
                draw_subtree(child)

        box_color = (
            LEAF_COLORS.get(node.predicted_class, "#e2e3e5")
            if node.is_leaf
            else INTERNAL_NODE_COLOR
        )
        edge_color = "#2b5c8f" if not node.is_leaf else "#2f6f3e"
        label_text = _format_node_label(node, criterion_name=tree_clf.criterion)

        ax.text(
            x_p,
            y_p,
            label_text,
            ha="center",
            va="center",
            fontsize=8.5,
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor=box_color,
                edgecolor=edge_color,
                linewidth=1.6,
            ),
            zorder=3,
        )

    draw_subtree(tree_clf.root_)

    # Legenda kolorów klas na dole wykresu
    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fc=LEAF_COLORS[c],
            ec="#2f6f3e",
            label=f"Klasa (Rating) {c}",
        )
        for c in range(6)
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        frameon=True,
        fontsize=9.5,
        title="Kolory liści odpowiadające przewidywanej ocenie filmu (0–5):",
        title_fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
