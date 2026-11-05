import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Any
from decision_tree import DecisionTreeClassifierCustom
from random_forest import RandomForestClassifierCustom


def adaptive_stratified_kfold_indices(
    y: np.ndarray, n_splits: int = 3, random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:

    rng = np.random.default_rng(random_state)
    n_samples = len(y)
    folds: List[List[int]] = [[] for _ in range(n_splits)]

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
) -> Dict[str, float]:

    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    n = len(y_t)
    if n == 0:
        return {"accuracy": 0.0, "weighted_f1": 0.0, "mae": 0.0}

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

    weighted_f1 = float(np.sum(per_class_f1 * supports) / float(n))

    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "mae": mae,
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
) -> Dict[int, Dict[str, float]]:

    if model_params is None:
        model_params = {}

    per_user_details: Dict[int, Dict[str, float]] = {}

    grouped = train_df.groupby("user_id", sort=True)
    for user_id, group in grouped:
        u_id = int(user_id)
        movie_ids = group["movie_id"].to_numpy(dtype=np.int64)
        y_user = group["rating"].to_numpy(dtype=np.int64)
        row_indices = [movie_id_to_row[int(m)] for m in movie_ids]
        X_user = X_movies[row_indices]

        splits = adaptive_stratified_kfold_indices(
            y_user, n_splits=n_splits, random_state=random_state + u_id
        )
        u_oof_preds = np.zeros_like(y_user)

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X_user[tr_idx], y_user[tr_idx]
            X_val = X_user[val_idx]

            seed = random_state + u_id * 10 + fold_idx
            if model_type == "tree":
                clf = DecisionTreeClassifierCustom(random_state=seed, **model_params)
            elif model_type == "forest":
                clf = RandomForestClassifierCustom(random_state=seed, **model_params)
            else:
                raise ValueError(f"Nieznany model_type: {model_type}")

            clf.fit(X_tr, y_tr, feature_names, is_binary_feature)
            u_oof_preds[val_idx] = clf.predict(X_val)

        per_user_details[u_id] = compute_classification_metrics(y_user, u_oof_preds)

    return per_user_details


def tune_hyperparameters_per_user(
    train_df: pd.DataFrame,
    X_movies: np.ndarray,
    movie_id_to_row: Dict[int, int],
    feature_names: List[str],
    is_binary_feature: np.ndarray,
    model_type: str,
    configs_grid: Dict[str, Dict[str, Any]],
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:

    config_results: Dict[str, Dict[int, Dict[str, float]]] = {}
    for cfg_name, cfg_params in configs_grid.items():
        config_results[cfg_name] = evaluate_per_user_cv(
            train_df=train_df,
            X_movies=X_movies,
            movie_id_to_row=movie_id_to_row,
            feature_names=feature_names,
            is_binary_feature=is_binary_feature,
            model_type=model_type,
            model_params=cfg_params,
            n_splits=n_splits,
            random_state=random_state,
        )

    user_ids = sorted(train_df["user_id"].astype(int).unique())
    best_config_name_per_user: Dict[int, str] = {}
    best_params_per_user: Dict[int, Dict[str, Any]] = {}

    for u_id in user_ids:
        best_cfg_u = None
        best_score_tuple = None

        for cfg_name in configs_grid.keys():
            u_det = config_results[cfg_name][u_id]
            score_tuple = (
                round(u_det["weighted_f1"], 6),
                round(u_det["accuracy"], 6),
                -round(u_det["mae"], 6),
            )
            if best_score_tuple is None or score_tuple > best_score_tuple:
                best_score_tuple = score_tuple
                best_cfg_u = cfg_name

        best_config_name_per_user[u_id] = best_cfg_u
        best_params_per_user[u_id] = configs_grid[best_cfg_u].copy()

    selection_counts = dict(Counter(best_config_name_per_user.values()).most_common())
    default_config_name = next(iter(selection_counts))

    return {
        "best_config_name_per_user": best_config_name_per_user,
        "best_params_per_user": best_params_per_user,
        "default_config_name": default_config_name,
        "config_selection_counts": selection_counts,
    }