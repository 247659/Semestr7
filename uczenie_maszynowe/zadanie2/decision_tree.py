"""
Własna implementacja Binarnego Drzewa Decyzyjnego (DecisionTreeClassifierCustom)
dla wieloklasowej klasyfikacji ocen filmów w skali 0..5.

Nie korzysta z żadnych gotowych implementacji drzew decyzyjnych (np. scikit-learn).
Obsługuje kryteria Gini Impurity oraz Entropy (Information Gain), wektoryzowane
przeszukiwanie progów podziału w NumPy, cechy binarne i ciągłe oraz deterministyczne
rozstrzyganie remisów w liściach w oparciu o rozkład a priori użytkownika.
"""

import numpy as np
from typing import Optional, List, Union, Tuple


class TreeNode:
    """Reprezentacja pojedynczego węzła lub liścia w binarnym drzewie decyzyjnym."""

    def __init__(
        self,
        is_leaf: bool = True,
        predicted_class: int = 3,
        class_counts: Optional[np.ndarray] = None,
        class_probs: Optional[np.ndarray] = None,
        n_samples: int = 0,
        impurity: float = 0.0,
        depth: int = 0,
        feature_idx: Optional[int] = None,   # Indeks kolumny (cechy $j$) w macierzy X, według której ten węzeł dzieli dane.
        feature_name: Optional[str] = None,
        threshold: Optional[float] = None,
        is_binary_feature: bool = False,
        impurity_decrease: float = 0.0,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
    ):
        self.is_leaf = is_leaf
        self.predicted_class = int(predicted_class)
        self.class_counts = (
            class_counts
            if class_counts is not None
            else np.zeros(6, dtype=np.int64)
        )
        self.class_probs = (
            class_probs
            if class_probs is not None
            else np.ones(6, dtype=np.float64) / 6.0
        )
        self.n_samples = int(n_samples)
        self.impurity = float(impurity)
        self.depth = int(depth)
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.threshold = threshold
        self.is_binary_feature = bool(is_binary_feature)
        self.impurity_decrease = float(impurity_decrease)
        self.left = left
        self.right = right


class DecisionTreeClassifierCustom:
    """
    Binarny klasyfikator drzewa decyzyjnego dla 6 klas ocen {0, 1, 2, 3, 4, 5}.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = 4,
        min_samples_split: int = 8,
        min_samples_leaf: int = 3,
        min_impurity_decrease: float = 0.005,
        max_features: Optional[Union[str, int, float]] = "all",
        max_thresholds: Optional[int] = 25,
        n_classes: int = 6,
        random_state: Optional[int] = 42,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = max(2, int(min_samples_split))
        self.min_samples_leaf = max(1, int(min_samples_leaf))
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.n_classes = int(n_classes)
        self.random_state = random_state

        self.root_: Optional[TreeNode] = None
        self.user_prior_: np.ndarray = np.zeros(self.n_classes, dtype=np.float64)
        self.user_mean_rating_: float = 3.0
        self.feature_names_: Optional[List[str]] = None
        self.is_binary_feature_: Optional[np.ndarray] = None
        self.rng_ = np.random.default_rng(random_state)

    @staticmethod
    def compute_gini(class_counts: np.ndarray) -> float:
        """Oblicza wskaźnik nieczystości Giniego: Gini(S) = 1 - sum(p_k^2)."""
        total = float(np.sum(class_counts))
        if total <= 0.0:
            return 0.0
        probs = class_counts / total
        return float(1.0 - np.sum(probs * probs))

    def _compute_node_impurity(self, class_counts: np.ndarray) -> float:
        return self.compute_gini(class_counts)

    def _compute_vectorized_impurity(
        self, counts_matrix: np.ndarray, totals: np.ndarray
    ) -> np.ndarray:
        """
        Wektoryzowane obliczanie nieczystości dla wielu podziałów jednocześnie.
        counts_matrix: kształt (T, n_classes)
        totals: kształt (T, 1)
        """
        probs = counts_matrix / np.maximum(totals, 1.0)
        return 1.0 - np.sum(probs * probs, axis=1)


    def _resolve_leaf_class(self, class_counts: np.ndarray) -> int:
        """
        Wyznacza klasę większościową w liściu z obsługą remisów (tie-breaking):
        1. Maksymalna liczebność w węźle.
        2. Wyższy rozkład a priori u danego użytkownika (user_prior_).
        3. Mniejsza odległość od średniej oceny użytkownika (user_mean_rating_).
        4. Preferencja globalnie najczęstszej klasy środkowej (3).
        """
        max_cnt = np.max(class_counts)
        candidates = np.flatnonzero(class_counts == max_cnt)
        if len(candidates) == 1:
            return int(candidates[0])

        # Tie-breaker 1: user prior
        priors = self.user_prior_[candidates]
        max_prior = np.max(priors)
        candidates = candidates[np.isclose(priors, max_prior)]
        if len(candidates) == 1:
            return int(candidates[0])

        # Tie-breaker 2: najbliżej średniej oceny użytkownika
        dists = np.abs(candidates - self.user_mean_rating_)
        min_dist = np.min(dists)
        candidates = candidates[np.isclose(dists, min_dist)]
        if len(candidates) == 1:
            return int(candidates[0])

        # Tie-breaker 3: najbliżej 3.0
        dists_global = np.abs(candidates - 3.0)
        return int(candidates[np.argmin(dists_global)])

    def _select_candidate_features(self, n_total_features: int) -> np.ndarray:
        """Wybiera indeksy cech rozpatrywanych przy podziale danego węzła."""
        if (
            self.max_features is None
            or self.max_features == "all"
            or self.max_features == n_total_features
        ):
            return np.arange(n_total_features)
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_total_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_total_features)))
        elif isinstance(self.max_features, float) and 0.0 < self.max_features <= 1.0:
            k = max(1, int(self.max_features * n_total_features))
        elif isinstance(self.max_features, int):
            k = max(1, min(n_total_features, self.max_features))
        else:
            k = n_total_features

        if k >= n_total_features:
            return np.arange(n_total_features)
        return self.rng_.choice(n_total_features, size=k, replace=False)

    def _find_best_split(
        self,
        X: np.ndarray,
        y_onehot: np.ndarray,
        parent_impurity: float,
        parent_counts: np.ndarray,
    ):
        """
        Przeszukuje kandydackie cechy i progi podziału za pomocą wektoryzacji NumPy.
        Zwraca (best_feat, best_thresh, best_gain).
        """
        n_samples, n_features = X.shape
        min_leaf = self.min_samples_leaf

        if n_samples < 2 * min_leaf:
            return None, None, 0.0

        candidate_features = self._select_candidate_features(n_features)

        best_gain = -1.0
        best_feat = None
        best_thresh = None

        def evaluate_feature_subset(feat_indices: np.ndarray):
            nonlocal best_gain, best_feat, best_thresh
            for j in feat_indices:
                col = X[:, j]
                is_bin = (
                    bool(self.is_binary_feature_[j])
                    if self.is_binary_feature_ is not None
                    else False
                )

                if is_bin:
                    # Cecha binarna: jedyny próg podziału to 0.5 (rozdziela 0 od 1)
                    left_mask = col <= 0.5
                    n_left = int(np.sum(left_mask))
                    n_right = n_samples - n_left
                    if n_left < min_leaf or n_right < min_leaf:
                        continue
                    left_counts = np.sum(y_onehot[left_mask], axis=0)
                    right_counts = parent_counts - left_counts
                    imp_l = self._compute_node_impurity(left_counts)
                    imp_r = self._compute_node_impurity(right_counts)
                    gain = parent_impurity - (
                        (n_left / n_samples) * imp_l + (n_right / n_samples) * imp_r
                    )
                    if gain > best_gain:
                        best_gain = float(gain)
                        best_feat = int(j)
                        best_thresh = 0.5
                else:
                    # Cecha ciągła / dyskretna: sortowanie i skumulowane sumy (np.cumsum)
                    order = np.argsort(col, kind="mergesort")
                    sorted_col = col[order]
                    if sorted_col[0] == sorted_col[-1]:
                        continue  # Cecha stała w węźle

                    sorted_onehot = y_onehot[order]
                    # Miejsca, w których wartość cechy faktycznie wzrasta
                    valid_split_pos = np.flatnonzero(sorted_col[:-1] < sorted_col[1:])
                    # Uwzględnienie ograniczenia min_samples_leaf
                    valid_split_pos = valid_split_pos[
                        (valid_split_pos + 1 >= min_leaf)
                        & (n_samples - (valid_split_pos + 1) >= min_leaf)
                    ]
                    if len(valid_split_pos) == 0:
                        continue

                    # Opcjonalne zawężenie do max_thresholds kwantyli
                    if (
                        self.max_thresholds is not None
                        and len(valid_split_pos) > self.max_thresholds
                    ):
                        q_idx = np.linspace(
                            0, len(valid_split_pos) - 1, self.max_thresholds
                        ).astype(int)
                        valid_split_pos = valid_split_pos[np.unique(q_idx)]

                    cum_counts = np.cumsum(sorted_onehot, axis=0)
                    left_counts_mat = cum_counts[valid_split_pos]
                    right_counts_mat = parent_counts - left_counts_mat

                    n_left_vec = (valid_split_pos + 1).reshape(-1, 1).astype(np.float64)
                    n_right_vec = float(n_samples) - n_left_vec

                    imp_l_vec = self._compute_vectorized_impurity(
                        left_counts_mat, n_left_vec
                    )
                    imp_r_vec = self._compute_vectorized_impurity(
                        right_counts_mat, n_right_vec
                    )

                    gains = parent_impurity - (
                        (n_left_vec[:, 0] / n_samples) * imp_l_vec
                        + (n_right_vec[:, 0] / n_samples) * imp_r_vec
                    )
                    max_idx = int(np.argmax(gains))
                    max_gain = float(gains[max_idx])
                    if max_gain > best_gain:
                        pos = int(valid_split_pos[max_idx])
                        best_gain = max_gain
                        best_feat = int(j)
                        best_thresh = float(
                            0.5 * (sorted_col[pos] + sorted_col[pos + 1])
                        )

        evaluate_feature_subset(candidate_features)

        # Edge-case: jeśli w Random Forest na wylosowanym podzbiorze cech nie znaleziono
        # żadnego dopuszczalnego podziału, próbujemy pozostałych cech
        if best_feat is None and len(candidate_features) < n_features:
            remaining = np.setdiff1d(np.arange(n_features), candidate_features)
            evaluate_feature_subset(remaining)

        return best_feat, best_thresh, best_gain

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, y_onehot: np.ndarray, depth: int
    ) -> TreeNode:
        n_samples = X.shape[0]
        class_counts = np.sum(y_onehot, axis=0).astype(np.int64)
        class_probs = class_counts / max(1.0, float(n_samples))
        impurity = self._compute_node_impurity(class_counts)
        predicted_class = self._resolve_leaf_class(class_counts)

        # Sprawdzenie warunków stopu
        if (
            impurity <= 1e-12
            or (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or n_samples < 2 * self.min_samples_leaf
        ):
            return TreeNode(
                is_leaf=True,
                predicted_class=predicted_class,
                class_counts=class_counts,
                class_probs=class_probs,
                n_samples=n_samples,
                impurity=impurity,
                depth=depth,
            )

        best_feat, best_thresh, best_gain = self._find_best_split(
            X, y_onehot, impurity, class_counts
        )

        if (
            best_feat is None
            or best_thresh is None
            or best_gain < self.min_impurity_decrease
            or best_gain <= 1e-12
        ):
            return TreeNode(
                is_leaf=True,
                predicted_class=predicted_class,
                class_counts=class_counts,
                class_probs=class_probs,
                n_samples=n_samples,
                impurity=impurity,
                depth=depth,
            )

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return TreeNode(
                is_leaf=True,
                predicted_class=predicted_class,
                class_counts=class_counts,
                class_probs=class_probs,
                n_samples=n_samples,
                impurity=impurity,
                depth=depth,
            )

        feat_name = (
            self.feature_names_[best_feat]
            if self.feature_names_ is not None and best_feat < len(self.feature_names_)
            else f"feature_{best_feat}"
        )
        is_bin = (
            bool(self.is_binary_feature_[best_feat])
            if self.is_binary_feature_ is not None
            else False
        )

        left_child = self._build_tree(
            X[left_mask], y[left_mask], y_onehot[left_mask], depth + 1
        )
        right_child = self._build_tree(
            X[right_mask], y[right_mask], y_onehot[right_mask], depth + 1
        )

        return TreeNode(
            is_leaf=False,
            predicted_class=predicted_class,
            class_counts=class_counts,
            class_probs=class_probs,
            n_samples=n_samples,
            impurity=impurity,
            depth=depth,
            feature_idx=best_feat,
            feature_name=feat_name,
            threshold=best_thresh,
            is_binary_feature=is_bin,
            impurity_decrease=best_gain,
            left=left_child,
            right=right_child,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        is_binary_feature: Optional[np.ndarray] = None,
    ) -> "DecisionTreeClassifierCustom":
        """Trenuje drzewo decyzyjne na macierzy cech X i wektorze ocen y."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)

        self.feature_names_ = feature_names
        self.is_binary_feature_ = is_binary_feature
        self.rng_ = np.random.default_rng(self.random_state)

        if X_arr.shape[0] == 0:
            # Zabezpieczenie dla pustego zbioru treningowego (zimny start)
            self.user_prior_ = np.ones(self.n_classes, dtype=np.float64) / self.n_classes
            self.user_mean_rating_ = 3.0
            self.root_ = TreeNode(
                is_leaf=True,
                predicted_class=3,
                class_counts=np.zeros(self.n_classes, dtype=np.int64),
                class_probs=self.user_prior_.copy(),
                n_samples=0,
                impurity=0.0,
                depth=0,
            )
            return self

        # Kodowanie one-hot wektora docelowego y w przestrzeni klas 0..5
        y_onehot = np.zeros((X_arr.shape[0], self.n_classes), dtype=np.int64)
        valid_y = np.clip(y_arr, 0, self.n_classes - 1)
        y_onehot[np.arange(X_arr.shape[0]), valid_y] = 1

        total_counts = np.sum(y_onehot, axis=0)
        self.user_prior_ = total_counts / float(X_arr.shape[0])
        self.user_mean_rating_ = float(np.mean(valid_y))

        self.root_ = self._build_tree(X_arr, valid_y, y_onehot, depth=0)
        return self

    def _predict_single_node(self, x_row: np.ndarray, node: TreeNode) -> TreeNode:
        while not node.is_leaf:
            if x_row[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Zwraca macierz prawdopodobieństw klas o wymiarach (N, 6)."""
        if self.root_ is None:
            raise RuntimeError("Drzewo nie zostało wytrenowane. Wywołaj najpierw fit().")
        X_arr = np.asarray(X, dtype=np.float64)
        probs = np.zeros((X_arr.shape[0], self.n_classes), dtype=np.float64)
        for i in range(X_arr.shape[0]):
            leaf = self._predict_single_node(X_arr[i], self.root_)
            probs[i, :] = leaf.class_probs
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Zwraca wektor przewidywanych klas całkowitych {0, 1, 2, 3, 4, 5}."""
        if self.root_ is None:
            raise RuntimeError("Drzewo nie zostało wytrenowane. Wywołaj najpierw fit().")
        X_arr = np.asarray(X, dtype=np.float64)
        preds = np.zeros(X_arr.shape[0], dtype=np.int64)
        for i in range(X_arr.shape[0]):
            leaf = self._predict_single_node(X_arr[i], self.root_)
            preds[i] = leaf.predicted_class
        return preds

    def count_nodes_and_leaves(self) -> Tuple[int, int, int]:
        """Zwraca (liczba_wszystkich_węzłów, liczba_liści, maksymalna_głębokość)."""
        if self.root_ is None:
            return 0, 0, 0
        total_nodes = 0
        leaf_nodes = 0
        max_d = 0
        stack = [self.root_]
        while stack:
            node = stack.pop()
            total_nodes += 1
            max_d = max(max_d, node.depth)
            if node.is_leaf:
                leaf_nodes += 1
            else:
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)
        return total_nodes, leaf_nodes, max_d
