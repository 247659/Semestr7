from typing import Dict, List, Tuple

import numpy as np

DEFAULT_K_CANDIDATES = (2, 3, 4, 5, 7, 9, 11, 15, 17)
POWER_CANDIDATES = (1.0, 2.0)


def calculate_loss(y_true: np.ndarray, y_pred: np.ndarray, loss_name: str = "mae") -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return float("inf")
    return float(np.mean(np.abs(y_pred - y_true)))


def rating_mode_without_current_label(y: np.ndarray) -> np.ndarray:
    # LOOCV fallback: dominanta dla każdego wiersza liczona bez jego własnej etykiety.
    y = np.asarray(y, dtype=int)
    total_counts = np.bincount(y, minlength=6)
    fallbacks = np.empty(len(y), dtype=int)
    for i, current_label in enumerate(y):
        counts = total_counts.copy()
        counts[current_label] -= 1
        fallbacks[i] = int(np.argmax(counts)) if counts.sum() else 3
    return fallbacks


class UserKNNClassifier:
    # Osobny kNN dla użytkownika; cechy oraz k/p dobierane wrapperem SFS i LOOCV.

    def __init__(
            self,
            user_id: int,
            k_candidates: Tuple[int, ...] = DEFAULT_K_CANDIDATES,
            max_selected_features: int = 8,
            eps: float = 1e-4,
            aggregation: str = "weighted_mean",
            loss_name: str = "mae",
    ):
        self.user_id = int(user_id)
        self.k_candidates = tuple(sorted({int(k) for k in k_candidates if int(k) >= 1})) or (1,)
        self.max_selected_features = max(0, int(max_selected_features))
        self.eps = float(eps)
        self.aggregation = aggregation
        self.loss_name = loss_name

        self.train_indices = np.array([], dtype=int)
        self.train_ratings = np.array([], dtype=int)
        self.selected_features: List[str] = []
        self.best_k = 1
        self.best_power = 1.0
        self.cv_loss = float("nan")
        self.user_mode_rating = 3

    def _aggregate_neighbors(self, neighbor_ratings: np.ndarray, weights: np.ndarray, fallback_ratings: np.ndarray
                             ) -> np.ndarray:
        # Agregacja weighted_mean: średnia ocen ważona podobieństwem sąsiadów.
        neighbor_ratings = np.asarray(neighbor_ratings, dtype=float)
        weights = np.asarray(weights, dtype=float)
        result = np.asarray(fallback_ratings, dtype=int).copy()
        weight_sum = weights.sum(axis=1)
        valid = weight_sum > 1e-12
        if np.any(valid):
            means = np.sum(neighbor_ratings[valid] * weights[valid], axis=1) / weight_sum[valid]
            result[valid] = np.clip(np.floor(means + 0.5), 0, 5).astype(int)
        return result

    def _predict_from_neighbors(self, sim_matrix: np.ndarray, train_ratings: np.ndarray, k: int, power: float,
                                fallback_rating: int
                                ) -> np.ndarray:
        # Predykcja z k najbliższych sąsiadów; fallback jest tylko dla braku wag.
        n_queries, n_train = sim_matrix.shape
        if n_queries == 0:
            return np.array([], dtype=int)
        if n_train == 0:
            return np.full(n_queries, int(fallback_rating), dtype=int)

        k_eff = min(max(1, int(k)), n_train)
        order = np.argsort(-sim_matrix, axis=1)[:, :k_eff]
        top_sims = np.clip(np.take_along_axis(sim_matrix, order, axis=1), 0.0, 1.0)
        top_ratings = train_ratings[order]
        distances = np.maximum(1.0 - top_sims, 0.0)
        weights = np.power(top_sims, power) / (distances + self.eps)
        fallback = np.full(n_queries, int(fallback_rating), dtype=int)
        return self._aggregate_neighbors(top_ratings, weights, fallback)

    def _evaluate_loocv(self, combined_sim_sub: np.ndarray, y: np.ndarray) -> Tuple[float, int, float]:
        # LOOCV dla konkretnego podzbioru cech; fallback nie używa odłożonej etykiety.
        y = np.asarray(y, dtype=int)
        n = len(y)
        if n < 2:
            return float("inf"), 1, 1.0

        similarities = np.asarray(combined_sim_sub, dtype=float).copy()
        np.fill_diagonal(similarities, -1.0)
        valid_ks = [k for k in self.k_candidates if k < n]
        if not valid_ks:
            valid_ks = [1]
        k_max = min(max(valid_ks), n - 1)
        order = np.argsort(-similarities, axis=1)[:, :k_max]
        sorted_sims = np.clip(np.take_along_axis(similarities, order, axis=1), 0.0, 1.0)
        sorted_y = y[order]

        fold_fallbacks = rating_mode_without_current_label(y)
        best_loss = float("inf")
        best_k = valid_ks[0]
        best_power = POWER_CANDIDATES[0]

        for power in POWER_CANDIDATES:
            weights_all = np.power(sorted_sims, power) / (
                    np.maximum(1.0 - sorted_sims, 0.0) + self.eps
            )
            for k in valid_ks:
                if k > weights_all.shape[1]:
                    continue
                weights_k = weights_all[:, :k]
                ratings_k = sorted_y[:, :k]
                predictions = self._aggregate_neighbors(ratings_k, weights_k, fold_fallbacks)
                loss = calculate_loss(y, predictions, self.loss_name)
                if loss < best_loss - 1e-12:
                    best_loss, best_k, best_power = loss, k, power

        return best_loss, int(best_k), float(best_power)

    def fit(self, train_movie_indices: np.ndarray, train_ratings: np.ndarray, sim_matrices: Dict[str, np.ndarray]
            ) -> "UserKNNClassifier":
        # Dopasowanie modelu użytkownika z SFS; zachowuje najlepszy niepusty podzbiór cech.
        self.train_indices = np.asarray(train_movie_indices, dtype=int)
        self.train_ratings = np.asarray(train_ratings, dtype=int)
        if len(self.train_indices) != len(self.train_ratings):
            raise ValueError("Liczba indeksów filmów nie zgadza się z liczbą ocen.")
        if np.any((self.train_ratings < 0) | (self.train_ratings > 5)):
            raise ValueError("Oceny treningowe muszą należeć do {0, 1, 2, 3, 4, 5}.")

        if len(self.train_ratings):
            vals, counts = np.unique(self.train_ratings, return_counts=True)
            self.user_mode_rating = int(vals[np.argmax(counts)])
        else:
            self.user_mode_rating = 3

        n = len(self.train_indices)
        if n < 2:
            self.selected_features = []
            self.best_k = 1
            self.best_power = 1.0
            self.cv_loss = float("nan")
            return self

        if self.max_selected_features == 0 or not sim_matrices:
            self.selected_features = []
            self.cv_loss, self.best_k, self.best_power = self._evaluate_loocv(
                np.zeros((n, n), dtype=float), self.train_ratings
            )
            return self

        candidate_features = list(sim_matrices.keys())
        user_sub_matrices = {
            feature: np.asarray(sim_matrices[feature])[np.ix_(self.train_indices, self.train_indices)]
            for feature in candidate_features
        }

        selected_path: List[str] = []
        remaining = list(candidate_features)
        current_sum_sim = np.zeros((n, n), dtype=float)

        best_loss = float("inf")
        best_k = self.k_candidates[0]
        best_power = 1.0
        best_selected: List[str] = []

        max_steps = min(self.max_selected_features, len(candidate_features))
        for _step in range(max_steps):
            step_best_feature = None
            step_best_loss = float("inf")
            step_best_k = 1
            step_best_power = 1.0

            for feature in remaining:
                candidate_sim = (current_sum_sim + user_sub_matrices[feature]) / (len(selected_path) + 1)
                loss, k_opt, power_opt = self._evaluate_loocv(candidate_sim, self.train_ratings)
                if loss < step_best_loss - 1e-12:
                    step_best_feature = feature
                    step_best_loss = loss
                    step_best_k = k_opt
                    step_best_power = power_opt

            if step_best_feature is None:
                break

            selected_path.append(step_best_feature)
            remaining.remove(step_best_feature)
            current_sum_sim += user_sub_matrices[step_best_feature]

            if step_best_loss < best_loss - 1e-12:
                best_loss = step_best_loss
                best_k = step_best_k
                best_power = step_best_power
                best_selected = selected_path.copy()

        self.selected_features = best_selected
        self.best_k = int(best_k)
        self.best_power = float(best_power)
        self.cv_loss = float(best_loss)
        return self

    def predict(self, query_movie_indices: np.ndarray, sim_matrices: Dict[str, np.ndarray]) -> np.ndarray:
        query_movie_indices = np.asarray(query_movie_indices, dtype=int)
        n_queries = len(query_movie_indices)
        n_train = len(self.train_indices)
        if n_queries == 0:
            return np.array([], dtype=int)
        if n_train == 0:
            return np.full(n_queries, 3, dtype=int)

        combined_sim = np.zeros((n_queries, n_train), dtype=float)
        if self.selected_features:
            for feature in self.selected_features:
                combined_sim += sim_matrices[feature][np.ix_(query_movie_indices, self.train_indices)]
            combined_sim /= len(self.selected_features)

        return self._predict_from_neighbors(sim_matrix=combined_sim, train_ratings=self.train_ratings, k=self.best_k,
                                            power=self.best_power, fallback_rating=self.user_mode_rating)
