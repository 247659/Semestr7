"""
Moduł walidacji krzyżowej per-user (Adaptive Stratified K-Fold) oraz obliczania metryk:
- Accuracy
- Macro F1-score
- Weighted F1-score
- Mean Absolute Error (MAE)
- Confusion Matrix (6 x 6)

Zapewnia całkowity brak przecieku danych (zero data leakage): każdy model
wewnątrz foldu trenuje wyłącznie na części treningowej danego użytkownika.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from decision_tree import DecisionTreeClassifierCustom
from random_forest import RandomForestClassifierCustom


def adaptive_stratified_kfold_indices(
    y: np.ndarray, n_splits: int = 3, random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Tworzy podział na n_splits foldów z zachowaniem proporcji klas (stratification).
    Bezpiecznie obsługuje klasy rzadkie o liczebności mniejszej niż n_splits (np. 1 lub 2 przykłady)
    poprzez cykliczne (round-robin) przypisywanie przetasowanych indeksów każdej klasy do kolejnych foldów.
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(y)
    folds: List[List[int]] = [[] for _ in range(n_splits)]

    # Grupowanie indeksów wg klasy i przetasowanie wewnątrz każdej klasy
    unique_classes = np.unique(y)
    current_fold = 0
    for c in unique_classes:
        c_idx = np.flatnonzero(y == c).copy()
        rng.shuffle(c_idx)
        for idx in c_idx:
            folds[current_fold % n_splits].append(int(idx))
            current_fold += 1

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(n_samples)
    for k in range(n_splits):
        val_idx = np.array(sorted(folds[k]), dtype=np.int64)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_indices[train_mask]
        splits.append((train_idx, val_idx))

    return splits


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 6
) -> Dict[str, Any]:
    """
    Oblicza kompletny zestaw metryk klasyfikacyjnych na parach (y_true, y_pred):
    - accuracy
    - macro_f1
    - weighted_f1
    - mae (średni błąd bezwzględny na skali 0..5)
    - confusion_matrix (kształt 6x6, wiersze: true, kolumny: pred)
    - per_class_f1 (wektor długości 6)
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    n = len(y_t)
    if n == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "mae": 0.0,
            "confusion_matrix": np.zeros((n_classes, n_classes), dtype=np.int64),
            "per_class_f1": np.zeros(n_classes, dtype=np.float64),
        }

    accuracy = float(np.mean(y_t == y_p))
    mae = float(np.mean(np.abs(y_t - y_p)))

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_t, y_p), 1)

    per_class_f1 = np.zeros(n_classes, dtype=np.float64)
    supports = np.sum(cm, axis=1).astype(np.float64)
    pred_totals = np.sum(cm, axis=0).astype(np.float64)

    for c in range(n_classes):
        tp = float(cm[c, c])
        denom = supports[c] + pred_totals[c]
        if denom > 0.0:
            per_class_f1[c] = (2.0 * tp) / denom
        else:
            per_class_f1[c] = 0.0

    # Macro F1 po klasach obecnych w y_true
    present_mask = supports > 0
    macro_f1 = (
        float(np.mean(per_class_f1[present_mask])) if np.any(present_mask) else 0.0
    )
    weighted_f1 = float(np.sum(per_class_f1 * supports) / float(n))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mae": mae,
        "confusion_matrix": cm,
        "per_class_f1": per_class_f1,
    }


def evaluate_per_user_cv(
    train_df: pd.DataFrame,
    X_movies: np.ndarray,
    movie_id_to_row: Dict[int, int],
    feature_names: List[str],
    is_binary_feature: np.ndarray,
    model_type: str = "tree",
    model_params: Dict[str, Any] = None,
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Przeprowadza walidację krzyżową (n_splits-Fold CV) oddzielnie dla każdego użytkownika.
    Zwraca zarówno metryki zagregowane na połączonych predykcjach (Pooled OOF),
    jak i średnie metryki liczone per użytkownik (Per-User Mean).
    """
    if model_params is None:
        model_params = {}

    oof_y_true_list: List[np.ndarray] = []
    oof_y_pred_list: List[np.ndarray] = []
    user_acc_list: List[float] = []
    user_wf1_list: List[float] = []

    grouped = train_df.groupby("user_id", sort=True)
    for user_id, group in grouped:
        movie_ids = group["movie_id"].to_numpy(dtype=np.int64)
        y_user = group["rating"].to_numpy(dtype=np.int64)
        row_indices = [movie_id_to_row[int(m)] for m in movie_ids]
        X_user = X_movies[row_indices]

        splits = adaptive_stratified_kfold_indices(
            y_user, n_splits=n_splits, random_state=random_state + int(user_id)
        )
        u_oof_preds = np.zeros_like(y_user)

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X_user[tr_idx], y_user[tr_idx]
            X_val = X_user[val_idx]

            seed = random_state + int(user_id) * 10 + fold_idx
            if model_type == "tree":
                clf = DecisionTreeClassifierCustom(random_state=seed, **model_params)
            elif model_type == "forest":
                clf = RandomForestClassifierCustom(random_state=seed, **model_params)
            else:
                raise ValueError(f"Nieznany model_type: {model_type}")

            clf.fit(X_tr, y_tr, feature_names, is_binary_feature)
            u_oof_preds[val_idx] = clf.predict(X_val)

        oof_y_true_list.append(y_user)
        oof_y_pred_list.append(u_oof_preds)

        u_metrics = compute_classification_metrics(y_user, u_oof_preds)
        user_acc_list.append(u_metrics["accuracy"])
        user_wf1_list.append(u_metrics["weighted_f1"])

    all_true = np.concatenate(oof_y_true_list)
    all_pred = np.concatenate(oof_y_pred_list)
    global_metrics = compute_classification_metrics(all_true, all_pred)

    return {
        "global_accuracy": global_metrics["accuracy"],
        "global_macro_f1": global_metrics["macro_f1"],
        "global_weighted_f1": global_metrics["weighted_f1"],
        "global_mae": global_metrics["mae"],
        "confusion_matrix": global_metrics["confusion_matrix"],
        "per_class_f1": global_metrics["per_class_f1"],
        "mean_user_accuracy": float(np.mean(user_acc_list)),
        "std_user_accuracy": float(np.std(user_acc_list)),
        "mean_user_weighted_f1": float(np.mean(user_wf1_list)),
        "std_user_weighted_f1": float(np.std(user_wf1_list)),
    }
