"""
Moduł walidacji krzyżowej per-user (Adaptive Stratified K-Fold), obliczania metryk
oraz indywidualnego doboru hiperparametrów (Per-User Hyperparameter Tuning):
- Accuracy
- Macro F1-score
- Weighted F1-score
- Mean Absolute Error (MAE)
- Confusion Matrix (6 x 6)

Zapewnia całkowity brak przecieku danych (zero data leakage): każdy model
wewnątrz foldu trenuje wyłącznie na części treningowej danego użytkownika,
a wybór najlepszej konfiguracji hiperparametrów odbywa się niezależnie
dla każdego z 358 użytkowników na podstawie jego wyników walidacyjnych.
"""

import numpy as np
import pandas as pd
from collections import Counter
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
    Przeprowadza walidację krzyżową (n_splits-Fold CV) dla pojedynczej zadanej
    konfiguracji hiperparametrów oddzielnie dla każdego użytkownika.
    Zwraca zarówno metryki zagregowane globalnie, jak i szczegółowe wyniki każdego użytkownika.
    """
    if model_params is None:
        model_params = {}

    oof_y_true_list: List[np.ndarray] = []
    oof_y_pred_list: List[np.ndarray] = []
    user_acc_list: List[float] = []
    user_wf1_list: List[float] = []
    per_user_details: Dict[int, Dict[str, Any]] = {}

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

        oof_y_true_list.append(y_user)
        oof_y_pred_list.append(u_oof_preds)

        u_metrics = compute_classification_metrics(y_user, u_oof_preds)
        user_acc_list.append(u_metrics["accuracy"])
        user_wf1_list.append(u_metrics["weighted_f1"])

        per_user_details[u_id] = {
            "y_true": y_user,
            "y_pred": u_oof_preds,
            "accuracy": u_metrics["accuracy"],
            "weighted_f1": u_metrics["weighted_f1"],
            "macro_f1": u_metrics["macro_f1"],
            "mae": u_metrics["mae"],
        }

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
        "per_user_details": per_user_details,
    }


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
    """
    Przeprowadza indywidualny dobór hiperparametrów (Per-User Hyperparameter Tuning)
    oddzielnie dla każdego użytkownika spośród szerokiej siatki konfiguracji `configs_grid`.

    Dla każdego użytkownika wybierana jest ta konfiguracja, która osiąga dla niego:
    1. Najwyższy wynik Weighted F1-score w 3-Fold CV,
    2. W razie remisu: najwyższe Accuracy,
    3. W razie dalszego remisu: najniższy błąd bezwzględny (MAE).

    Zwraca:
    - per_config_summaries: wyniki każdej pojedynczej konfiguracji w skali całej populacji,
    - best_global_config_name: nazwa konfiguracji najlepszej globalnie (dla porównania),
    - best_config_name_per_user: słownik {user_id -> nazwa_najlepszej_konfiguracji_użytkownika},
    - best_params_per_user: słownik {user_id -> słownik_najlepszych_hiperparametrów_użytkownika},
    - config_selection_counts: licznik pokazujący, ilu użytkowników wybrało daną konfigurację,
    - personalized_metrics: zagregowane metryki Out-Of-Fold po zastosowaniu indywidualnie
      dobranych hiperparametrów dla każdego użytkownika.
    """
    config_results: Dict[str, Dict[str, Any]] = {}
    for cfg_name, cfg_params in configs_grid.items():
        res = evaluate_per_user_cv(
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
        config_results[cfg_name] = res

    # Wyznaczenie najlepszej pojedynczej konfiguracji globalnej (punkt odniesienia)
    best_global_cfg = max(
        config_results.keys(),
        key=lambda k: (
            config_results[k]["global_weighted_f1"],
            config_results[k]["global_accuracy"],
            -config_results[k]["global_mae"],
        ),
    )

    # Indywidualny wybór najlepszej konfiguracji dla każdego użytkownika osobno
    user_ids = sorted(train_df["user_id"].astype(int).unique())
    best_config_name_per_user: Dict[int, str] = {}
    best_params_per_user: Dict[int, Dict[str, Any]] = {}

    personalized_true_list: List[np.ndarray] = []
    personalized_pred_list: List[np.ndarray] = []
    personalized_user_acc: List[float] = []
    personalized_user_wf1: List[float] = []

    for u_id in user_ids:
        best_cfg_u = None
        best_score_tuple = None
        best_u_detail = None

        for cfg_name in configs_grid.keys():
            u_det = config_results[cfg_name]["per_user_details"][u_id]
            # Kryterium wyboru per-user: (weighted_f1, accuracy, -mae)
            score_tuple = (
                round(u_det["weighted_f1"], 6),
                round(u_det["accuracy"], 6),
                -round(u_det["mae"], 6),
            )
            if best_score_tuple is None or score_tuple > best_score_tuple:
                best_score_tuple = score_tuple
                best_cfg_u = cfg_name
                best_u_detail = u_det

        best_config_name_per_user[u_id] = best_cfg_u
        best_params_per_user[u_id] = configs_grid[best_cfg_u].copy()

        personalized_true_list.append(best_u_detail["y_true"])
        personalized_pred_list.append(best_u_detail["y_pred"])
        personalized_user_acc.append(best_u_detail["accuracy"])
        personalized_user_wf1.append(best_u_detail["weighted_f1"])

    all_pers_true = np.concatenate(personalized_true_list)
    all_pers_pred = np.concatenate(personalized_pred_list)
    pers_global_metrics = compute_classification_metrics(all_pers_true, all_pers_pred)

    selection_counts = dict(Counter(best_config_name_per_user.values()).most_common())

    return {
        "per_config_summaries": config_results,
        "best_global_config_name": best_global_cfg,
        "best_global_summary": config_results[best_global_cfg],
        "best_config_name_per_user": best_config_name_per_user,
        "best_params_per_user": best_params_per_user,
        "config_selection_counts": selection_counts,
        "personalized_metrics": {
            "global_accuracy": pers_global_metrics["accuracy"],
            "global_macro_f1": pers_global_metrics["macro_f1"],
            "global_weighted_f1": pers_global_metrics["weighted_f1"],
            "global_mae": pers_global_metrics["mae"],
            "confusion_matrix": pers_global_metrics["confusion_matrix"],
            "per_class_f1": pers_global_metrics["per_class_f1"],
            "mean_user_accuracy": float(np.mean(personalized_user_acc)),
            "std_user_accuracy": float(np.std(personalized_user_acc)),
            "mean_user_weighted_f1": float(np.mean(personalized_user_wf1)),
            "std_user_weighted_f1": float(np.std(personalized_user_wf1)),
        },
    }