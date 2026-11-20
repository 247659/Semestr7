import time
import numpy as np
from typing import Dict, Any, Counter

from data_loader import load_csv_datasets, build_rating_matrix
from similarity import compute_user_similarity_matrix
from collaborative_filtering import UserBasedCFRecommender
from evaluation import tune_cf_hyperparameters_per_user
from utils import (
    save_submission_file,
    verify_submission_output,
)

K_VALUES = [1, 2, 4, 6, 10]


def get_cf_configs_grid() -> Dict[str, Dict[str, Any]]:
    base_methods = {
        "CF_jaccard_adjcos_vote": {
            "similarity_metric": "jaccard_adjcos",
            "aggregation": "weighted_voting",
            "sim_exponent": 2.0,
        },
        "CF_jaccard_adjcos_zscore": {
            "similarity_metric": "jaccard_adjcos",
            "aggregation": "z_score",
            "sim_exponent": 1.5,
        },
        "CF_jaccard_adjcos_avarege": {
            "similarity_metric": "jaccard_adjcos",
            "aggregation": "weighted_average",
            "sim_exponent": 2.0,
        },
        "CF_jaccard_pearson_vote": {
            "similarity_metric": "jaccard_pearson",
            "aggregation": "weighted_voting",
            "sim_exponent": 1.5,
        },
        "CF_jaccard_pearson_average": {
            "similarity_metric": "jaccard_pearson",
            "aggregation": "weighted_average",
            "sim_exponent": 2.0,
        },
        "CF_jaccard_pearson_zscore": {
            "similarity_metric": "jaccard_pearson",
            "aggregation": "z_score",
            "sim_exponent": 1.5,
        },
        "CF_jaccard_agree_vote": {
            "similarity_metric": "jaccard_agreement",
            "aggregation": "weighted_voting",
            "sim_exponent": 2.0,
        },
        "CF_jaccard_agree_average": {
            "similarity_metric": "jaccard_agreement",
            "aggregation": "weighted_average",
            "sim_exponent": 1.5,
        },
        "CF_jaccard_agree_zscore": {
            "similarity_metric": "jaccard_agreement",
            "aggregation": "z_score",
            "sim_exponent": 2.0,
        },
    }

    configs_grid: Dict[str, Dict[str, Any]] = {}
    for base_name, base_params in base_methods.items():
        for k in K_VALUES:
            cfg_name = f"{base_name}_k{k}"
            params = base_params.copy()
            params["k_neighbors"] = k
            configs_grid[cfg_name] = params

    return configs_grid


def main() -> None:
    start_time = time.time()

    print("\n[1/4] Wczytywanie plików CSV i budowa macierzy ocen użytkownik-film...")
    movies_df, train_df, task_df = load_csv_datasets(
        "movie.csv", "train.csv", "task.csv"
    )
    R_matrix, user_ids, movie_ids, user_to_idx, movie_to_idx = build_rating_matrix(
        train_df, movies_df
    )
    print(
        f"  - Liczba użytkowników: {len(user_ids)}, liczba filmów: {len(movie_ids)}"
    )
    print(
        f"  - Rekordów treningowych (train.csv): {len(train_df)}, "
        f"rekordów do uzupełnienia (task.csv): {len(task_df)}"
    )

    cf_configs = get_cf_configs_grid()
    print(
        f"\n[2/4] Indywidualny dobór hiperparametrów (3-Fold CV, {len(cf_configs)} konfiguracji, "
        f"k={K_VALUES}) dla każdego z 358 użytkowników..."
    )
    cf_tuning = tune_cf_hyperparameters_per_user(
        R=R_matrix,
        user_ids=user_ids,
        configs_grid=cf_configs,
        n_splits=3,
        random_state=42,
    )

    print("\n  - Rozkład wybranych pełnych konfiguracji (Top 10):")
    for cfg_name, cnt in list(cf_tuning["config_selection_counts"].items())[:10]:
        print(f"      {cfg_name:<35}: {cnt} użytkowników")

    best_params_list = list(cf_tuning["best_params_per_user"].values())
    for param_name in ["similarity_metric", "aggregation", "k_neighbors", "sim_exponent"]:
        counts = dict(Counter(p[param_name] for p in best_params_list).most_common())
        print(f"  - Wybory dla [{param_name:<17}]: {counts}")

    print("\n[3/4] Wyznaczanie podobieństw na pełnym zbiorze train.csv i predykcja dla task.csv...")
    best_params_per_user = cf_tuning["best_params_per_user"]
    default_params = cf_configs[cf_tuning["default_config_name"]]

    full_sim_cache = {}
    full_preds_matrix = np.zeros_like(R_matrix, dtype=np.int64)

    for u_id in user_ids:
        u_idx = user_to_idx[u_id]
        u_params = best_params_per_user.get(u_id, default_params)
        sim_key = (
            u_params.get("similarity_metric", "jaccard_adjcos"),
            int(u_params.get("min_common", 1)),
        )
        if sim_key not in full_sim_cache:
            full_sim_cache[sim_key] = compute_user_similarity_matrix(
                R_matrix,
                metric=sim_key[0],
                min_common=sim_key[1],
            )

        recommender = UserBasedCFRecommender(**u_params)
        recommender.fit(R_matrix, precomputed_sim=full_sim_cache[sim_key])
        full_preds_matrix[u_idx] = recommender.predict_for_user(u_idx)

    task_u_indices = [user_to_idx[int(u)] for u in task_df["user_id"]]
    task_m_indices = [movie_to_idx[int(m)] for m in task_df["movie_id"]]
    task_predictions = full_preds_matrix[task_u_indices, task_m_indices]

    save_submission_file(task_df, task_predictions, "submission.csv")
    print("  - Zapisano plik: submission.csv")

    print("\n[4/4] Weryfikacja wygenerowanego pliku submission.csv...")
    report = verify_submission_output("task.csv", "submission.csv")
    print(f"  - submission.csv: {report}")

    elapsed = time.time() - start_time
    print(f"\nProces zakończony pomyślnie w {elapsed:.2f} s.")


if __name__ == "__main__":
    main()
