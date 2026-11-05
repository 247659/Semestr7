"""
Własna implementacja Lasu Losowego (RandomForestClassifierCustom)
wykorzystująca wyłącznie własną klasę DecisionTreeClassifierCustom.

Zapewnia różnorodność drzew poprzez:
1. Bootstrap sampling (losowanie próbek ze zwracaniem dla każdego drzewa).
2. Losowy podzbiór cech rozpatrywany w każdym węźle (max_features='sqrt').
3. Opcjonalną dywersyfikację hiperparametrów między drzewami.

Obsługuje dwa warianty agregacji:
- 'soft': uśrednianie wektorów prawdopodobieństw klas po wszystkich drzewach.
- 'hard': głosowanie większościowe po przewidywanych klasach.
"""

import numpy as np
from typing import Optional, List, Union
from decision_tree import DecisionTreeClassifierCustom


class RandomForestClassifierCustom:
    """
    Zespół (ensemble) binarnych drzew decyzyjnych dla klasyfikacji wieloklasowej 0..5.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        criterion: str = "gini",
        max_depth: Optional[int] = 8,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[Union[str, int, float]] = "sqrt",
        max_thresholds: Optional[int] = 25,
        bootstrap: bool = True,
        voting: str = "soft",
        diversify_trees: bool = False,
        n_classes: int = 6,
        random_state: Optional[int] = 42,
    ):
        if voting not in ("soft", "hard"):
            raise ValueError("Parametr voting musi mieć wartość 'soft' lub 'hard'.")
        self.n_estimators = max(1, int(n_estimators))
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.bootstrap = bool(bootstrap)
        self.voting = voting
        self.diversify_trees = bool(diversify_trees)
        self.n_classes = int(n_classes)
        self.random_state = random_state

        self.trees_: List[DecisionTreeClassifierCustom] = []
        self.user_prior_: np.ndarray = np.zeros(self.n_classes, dtype=np.float64)
        self.user_mean_rating_: float = 3.0
        self.rng_ = np.random.default_rng(random_state)

    def _resolve_tie(self, scores_vec: np.ndarray) -> int:
        """Rozstrzyga ewentualne remisy w zagregowanym wyniku na korzyść rozkładu a priori użytkownika."""
        max_val = np.max(scores_vec)
        candidates = np.flatnonzero(np.isclose(scores_vec, max_val, atol=1e-12))
        if len(candidates) == 1:
            return int(candidates[0])

        priors = self.user_prior_[candidates]
        max_prior = np.max(priors)
        candidates = candidates[np.isclose(priors, max_prior, atol=1e-12)]
        if len(candidates) == 1:
            return int(candidates[0])

        dists = np.abs(candidates - self.user_mean_rating_)
        min_dist = np.min(dists)
        candidates = candidates[np.isclose(dists, min_dist, atol=1e-12)]
        return int(candidates[0])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        is_binary_feature: Optional[np.ndarray] = None,
    ) -> "RandomForestClassifierCustom":
        """Trenuje zespół n_estimators drzew decyzyjnych."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples = X_arr.shape[0]

        self.rng_ = np.random.default_rng(self.random_state)
        self.trees_ = []

        if n_samples == 0:
            self.user_prior_ = np.ones(self.n_classes, dtype=np.float64) / self.n_classes
            self.user_mean_rating_ = 3.0
            tree = DecisionTreeClassifierCustom(
                n_classes=self.n_classes, random_state=self.random_state
            )
            tree.fit(X_arr, y_arr, feature_names, is_binary_feature)
            self.trees_.append(tree)
            return self

        valid_y = np.clip(y_arr, 0, self.n_classes - 1)
        counts = np.bincount(valid_y, minlength=self.n_classes).astype(np.float64)
        self.user_prior_ = counts / float(n_samples)
        self.user_mean_rating_ = float(np.mean(valid_y))

        for b in range(self.n_estimators):
            tree_seed = int(self.rng_.integers(0, 2**31 - 1))

            if self.bootstrap:
                boot_indices = self.rng_.choice(n_samples, size=n_samples, replace=True)
                X_boot = X_arr[boot_indices]
                y_boot = valid_y[boot_indices]
            else:
                X_boot = X_arr
                y_boot = valid_y

            # Opcjonalna dywersyfikacja hiperparametrów między drzewami
            crit = self.criterion
            leaf_min = self.min_samples_leaf
            if self.diversify_trees:
                crit = "gini"
                leaf_min = int(self.rng_.choice([1, 2, 3]))

            tree = DecisionTreeClassifierCustom(
                criterion=crit,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=leaf_min,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_thresholds=self.max_thresholds,
                n_classes=self.n_classes,
                random_state=tree_seed,
            )
            tree.fit(X_boot, y_boot, feature_names, is_binary_feature)
            # Przekazanie pełnego rozkładu a priori użytkownika do rozstrzygania remisów w liściach
            tree.user_prior_ = self.user_prior_.copy()
            tree.user_mean_rating_ = self.user_mean_rating_
            self.trees_.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Zwraca uśrednioną macierz prawdopodobieństw klas o wymiarach (N, 6)."""
        if not self.trees_:
            raise RuntimeError("Las losowy nie został wytrenowany. Wywołaj najpierw fit().")
        X_arr = np.asarray(X, dtype=np.float64)
        total_probs = np.zeros((X_arr.shape[0], self.n_classes), dtype=np.float64)
        for tree in self.trees_:
            total_probs += tree.predict_proba(X_arr)
        return total_probs / float(len(self.trees_))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Zwraca wektor przewidywanych klas całkowitych {0, 1, 2, 3, 4, 5}."""
        if not self.trees_:
            raise RuntimeError("Las losowy nie został wytrenowany. Wywołaj najpierw fit().")
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = X_arr.shape[0]
        preds = np.zeros(n_samples, dtype=np.int64)

        if self.voting == "soft":
            avg_probs = self.predict_proba(X_arr)
            for i in range(n_samples):
                preds[i] = self._resolve_tie(avg_probs[i])
        else:
            # Hard voting: zliczanie głosów poszczególnych drzew
            votes = np.zeros((n_samples, self.n_classes), dtype=np.float64)
            for tree in self.trees_:
                t_preds = tree.predict(X_arr)
                votes[np.arange(n_samples), t_preds] += 1.0
            for i in range(n_samples):
                preds[i] = self._resolve_tie(votes[i])

        return preds
