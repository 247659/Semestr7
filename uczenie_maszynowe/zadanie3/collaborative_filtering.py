"""
Własna implementacja klasyfikatora / rekomendera User-Based Collaborative Filtering
(UserBasedCFRecommender).
"""

import numpy as np
from typing import Optional
from similarity import compute_user_similarity_matrix


class UserBasedCFRecommender:
    """
    Model rekomendacji oparty na podobieństwie użytkowników (User-Based k-NN CF).
    """

    def __init__(
        self,
        similarity_metric: str = "jaccard_adjcos",
        k_neighbors: int = 25,
        aggregation: str = "weighted_voting",
        shrinkage: float = 10.0,
        sim_exponent: float = 1.5,
        positive_only: bool = False,
        min_common: int = 2,
    ):
        self.similarity_metric = similarity_metric
        self.k_neighbors = int(k_neighbors)
        self.aggregation = aggregation
        self.shrinkage = float(shrinkage)
        self.sim_exponent = float(sim_exponent)
        self.positive_only = bool(positive_only)
        self.min_common = int(min_common)

        self.R_train_: Optional[np.ndarray] = None
        self.sim_matrix_: Optional[np.ndarray] = None
        self.user_means_: Optional[np.ndarray] = None
        self.user_stds_: Optional[np.ndarray] = None
        self.R_diff_: Optional[np.ndarray] = None
        self.mask_: Optional[np.ndarray] = None

    def fit(
        self, R_train: np.ndarray, precomputed_sim: Optional[np.ndarray] = None
    ) -> "UserBasedCFRecommender":
        """
        Zapisuje macierz ocen treningowych (U, M), wyznacza statystyki użytkowników
        oraz macierz podobieństwa między użytkownikami (U, U).
        """
        self.R_train_ = np.asarray(R_train, dtype=np.float64)
        self.mask_ = ~np.isnan(self.R_train_)
        mask_f = self.mask_.astype(np.float64)

        user_counts = np.maximum(mask_f.sum(axis=1), 1.0)
        self.user_means_ = np.nansum(self.R_train_, axis=1) / user_counts
        self.R_diff_ = np.where(
            self.mask_, self.R_train_ - self.user_means_[:, None], 0.0
        )
        self.user_stds_ = np.sqrt(
            np.maximum(
                np.sum((self.R_diff_ ** 2) * mask_f, axis=1) / user_counts, 1e-4
            )
        )

        if precomputed_sim is not None:
            self.sim_matrix_ = precomputed_sim
        else:
            self.sim_matrix_ = compute_user_similarity_matrix(
                self.R_train_,
                metric=self.similarity_metric,
                shrinkage=self.shrinkage,
                min_common=self.min_common,
            )
        return self

    def predict_for_user(self, u_idx: int) -> np.ndarray:
        """
        Wyznacza wektor przewidywanych ocen całkowitych {0, 1, 2, 3, 4, 5}
        o długości M dla pojedynczego użytkownika u_idx.
        Dla każdego filmu m brani są pod uwagę WYŁĄCZNIE ci sąsiedzi, którzy
        widzieli (ocenili) film m (Hint 3).
        """
        if self.R_train_ is None or self.sim_matrix_ is None:
            raise RuntimeError("Model nie został wytrenowany. Wywołaj fit().")

        U, M = self.R_train_.shape
        mask = self.mask_
        col_idx = np.arange(M)[None, :]

        u_sims = self.sim_matrix_[u_idx].copy()
        u_sims[u_idx] = 0.0

        use_pos_only = self.positive_only or (
            self.similarity_metric in ("cosine", "msd", "jaccard_agreement")
        )
        if use_pos_only:
            u_sims = np.maximum(0.0, u_sims)
            sort_scores = np.where(mask, u_sims[:, None], -1.0)
        else:
            sort_scores = np.where(mask, np.abs(u_sims[:, None]), -1.0)

        # Wybór k_neighbors najbardziej podobnych osób spośród tych, które WIDZIAŁY film m (Hint 3)
        k_eff = min(self.k_neighbors, U - 1) if self.k_neighbors is not None else U
        if 0 < k_eff < U:
            part_idx = np.argpartition(-sort_scores, kth=k_eff, axis=0)[:k_eff, :]
        else:
            part_idx = np.broadcast_to(np.arange(U)[:, None], (U, M))

        top_mask = mask[part_idx, col_idx]
        top_sims = u_sims[part_idx] * top_mask

        w = (np.abs(top_sims) ** self.sim_exponent) * np.sign(top_sims)
        abs_w_sum = np.sum(np.abs(w), axis=0)
        safe_abs_w_sum = np.maximum(abs_w_sum, 1e-12)
        fallback_rating = int(np.clip(np.rint(self.user_means_[u_idx]), 0, 5))

        if self.aggregation == "z_score":
            neigh_z = self.R_diff_[part_idx, col_idx] / self.user_stds_[part_idx]
            num = np.sum(w * neigh_z, axis=0)
            cont_pred = self.user_means_[u_idx] + self.user_stds_[u_idx] * np.where(
                abs_w_sum > 1e-12, num / safe_abs_w_sum, 0.0
            )
            return np.clip(np.rint(cont_pred), 0, 5).astype(np.int64)

        elif self.aggregation == "weighted_average":
            w_pos = np.maximum(0.0, w)
            w_sum = np.sum(w_pos, axis=0)
            safe_w_sum = np.maximum(w_sum, 1e-12)
            neigh_r = np.where(top_mask, self.R_train_[part_idx, col_idx], 0.0)
            num = np.sum(w_pos * neigh_r, axis=0)
            cont_pred = np.where(
                w_sum > 1e-12, num / safe_w_sum, self.user_means_[u_idx]
            )
            return np.clip(np.rint(cont_pred), 0, 5).astype(np.int64)

        elif self.aggregation == "weighted_voting":
            shifted_r = np.clip(
                np.rint(
                    self.user_means_[u_idx]
                    + np.sign(top_sims) * self.R_diff_[part_idx, col_idx]
                ),
                0,
                5,
            ).astype(np.int64)
            w_abs = np.abs(w)
            class_scores = np.zeros((6, M), dtype=np.float64)
            for c in range(6):
                class_scores[c] = np.sum(w_abs * (shifted_r == c) * top_mask, axis=0)
            best_c = np.argmax(class_scores, axis=0)
            return np.where(abs_w_sum > 1e-12, best_c, fallback_rating).astype(np.int64)

        else:
            raise ValueError(f"Nieznana metoda agregacji: {self.aggregation}")

    def predict_all_matrix(self) -> np.ndarray:
        """Zwraca pełną macierz przewidywanych ocen (U, M) w skali {0, 1, 2, 3, 4, 5}."""
        U, M = self.R_train_.shape
        preds = np.zeros((U, M), dtype=np.int64)
        for u in range(U):
            preds[u] = self.predict_for_user(u)
        return preds