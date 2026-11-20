import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Any
from similarity import compute_user_similarity_matrix
from collaborative_filtering import UserBasedCFRecommender


def create_per_user_stratified_folds(
    R: np.ndarray, n_splits: int = 3, random_state: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    U, M = R.shape
    fold_matrix = np.zeros((U, M), dtype=np.int64)

    for u in range(U):
        seen_cols = np.flatnonzero(~np.isnan(R[u]))
        y_u = R[u, seen_cols].astype(np.int64)
        current_fold = 0
        for c in np.unique(y_u):
            c_cols = seen_cols[y_u == c].copy()
            rng.shuffle(c_cols)
            for col in c_cols:
                fold_matrix[u, col] = (current_fold % n_splits) + 1
                current_fold += 1

    return fold_matrix


def compute_user_metrics(
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


def tune_cf_hyperparameters_per_user(
    R: np.ndarray,
    user_ids: List[int],
    configs_grid: Dict[str, Dict[str, Any]],
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    U, M = R.shape
    fold_matrix = create_per_user_stratified_folds(
        R, n_splits=n_splits, random_state=random_state
    )

    oof_preds_by_cfg: Dict[str, np.ndarray] = {
        cfg_name: np.zeros((U, M), dtype=np.int64) for cfg_name in configs_grid
    }

    for fold_id in range(1, n_splits + 1):
        val_mask = fold_matrix == fold_id
        R_tr = np.where(val_mask, np.nan, R)

        sim_cache: Dict[Tuple[str, float, int], np.ndarray] = {}

        for cfg_name, params in configs_grid.items():
            sim_key = (
                params.get("similarity_metric", "jaccard_adjcos"),
                int(params.get("min_common", 1)),
            )
            if sim_key not in sim_cache:
                sim_cache[sim_key] = compute_user_similarity_matrix(
                    R_tr,
                    metric=sim_key[0],
                    min_common=sim_key[1],
                )

            model = UserBasedCFRecommender(**params)
            model.fit(R_tr, precomputed_sim=sim_cache[sim_key])
            fold_preds = model.predict_all_matrix()
            oof_preds_by_cfg[cfg_name][val_mask] = fold_preds[val_mask]

    best_config_name_per_user: Dict[int, str] = {}
    best_params_per_user: Dict[int, Dict[str, Any]] = {}

    for u_idx, u_id in enumerate(user_ids):
        seen_mask = fold_matrix[u_idx] > 0
        y_true_u = R[u_idx, seen_mask].astype(np.int64)

        best_cfg_u = None
        best_score_tuple = None

        for cfg_name, params in configs_grid.items():
            y_pred_u = oof_preds_by_cfg[cfg_name][u_idx, seen_mask]
            u_metrics = compute_user_metrics(y_true_u, y_pred_u)

            score_tuple = (
                round(u_metrics["weighted_f1"], 6),
                round(u_metrics["accuracy"], 6),
                -round(u_metrics["mae"], 6),
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