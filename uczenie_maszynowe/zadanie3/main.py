"""
Główny moduł wykonawczy (main.py) dla zadania rekomendacji filmów
na podstawie podobieństwa między osobami (User-Based Collaborative Filtering).
"""

import time
import numpy as np
from typing import Dict, Any, Counter

from data_loader import load_csv_datasets, build_rating_matrix
from similarity import compute_user_similarity_matrix
from collaborative_filtering import UserBasedCFRecommender
from evaluation import tune_cf_hyperparameters_per_user
from utils import (
    run_all_unit_tests,
    save_submission_file,
    verify_submission_output,
)

# Tablica różnych wartości liczby sąsiadów k sprawdzana dla każdego sposobu
K_VALUES = [2, 4, 6, 10, 15, 20]


def get_cf_configs_grid() -> Dict[str, Dict[str, Any]]:
    """
    Buduje siatkę konfiguracji hiperparametrów, w której każdy sposób
    (kombinacja miary podobieństwa, w tym wariantów Jaccarda, oraz metody agregacji)
    jest testowany dla całej tablicy różnych wartości k (K_VALUES).
    """
    base_methods = {
        # 1. Warianty oparte na mierze Jaccarda (Jaccard-weighted & Jaccard Agreement)
        "CF_jaccard_adjcos_vote": {
            "similarity_metric": "jaccard_adjcos",
            "aggregation": "weighted_voting",
            "sim_exponent": 2.0,
            "positive_only": False,
        },
        "CF_jaccard_adjcos_zscore": {
            "similarity_metric": "jaccard_adjcos",
            "aggregation": "z_score",
            "sim_exponent": 1.5,
            "positive_only": False,
        },
        "CF_jaccard_adjcos_avarege": {
            "similarity_metric": "jaccard_adjcos",
            "aggregation": "weighted_average",
            "sim_exponent": 2.0,
            "positive_only": False,
        },
        "CF_jaccard_pearson_vote": {
            "similarity_metric": "jaccard_pearson",
            "aggregation": "weighted_voting",
            "sim_exponent": 1.5,
            "positive_only": False,
        },
        "CF_jaccard_pearson_average": {
            "similarity_metric": "jaccard_pearson",
            "aggregation": "weighted_average",
            "sim_exponent": 2.0,
            "positive_only": False,
        },
        "CF_jaccard_pearson_zscore": {
            "similarity_metric": "jaccard_pearson",
            "aggregation": "z_score",
            "sim_exponent": 1.5,
            "positive_only": False,
        },
        "CF_jaccard_agree_vote": {
            "similarity_metric": "jaccard_agreement",
            "aggregation": "weighted_voting",
            "sim_exponent": 2.0,
            "positive_only": True,
        },
        "CF_jaccard_agree_average": {
            "similarity_metric": "jaccard_agreement",
            "aggregation": "weighted_average",
            "sim_exponent": 1.5,
            "positive_only": True,
        },
        "CF_jaccard_agree_zscore": {
            "similarity_metric": "jaccard_agreement",
            "aggregation": "z_score",
            "sim_exponent": 2.0,
            "positive_only": True,
        },
    }

    configs_grid: Dict[str, Dict[str, Any]] = {}
    for base_name, base_params in base_methods.items():
        for k in K_VALUES:
            for pos_only in [True, False]:
                if base_params["similarity_metric"] == "jaccard_agreement" and not pos_only:
                    continue
                pos_tag = "pos" if pos_only else "all"
                cfg_name = f"{base_name}_k{k}_{pos_tag}"
                params = base_params.copy()
                params["k_neighbors"] = k
                params["positive_only"] = pos_only
                configs_grid[cfg_name] = params

    return configs_grid


def main() -> None:
    start_time = time.time()
    print("=" * 80)
    print("SYSTEM REKOMENDACJI FILMÓW OPARTY NA PODOBIEŃSTWIE OSÓB (USER-BASED CF)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # KROK 1: Testy jednostkowe (Hints 1-4)
    # -------------------------------------------------------------------------
    # print("\n[1/5] Uruchamianie testów jednostkowych...")
    # test_results = run_all_unit_tests()
    # for test_name, passed in test_results.items():
    #     print(f"  - {test_name}: {'PASSED' if passed else 'FAILED'}")

    # -------------------------------------------------------------------------
    # KROK 2: Wczytanie danych i budowa macierzy ocen R (bez cech filmów - Hint 1)
    # -------------------------------------------------------------------------
    print("\n[2/5] Wczytywanie plików CSV i budowa macierzy ocen użytkownik-film...")
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

    # -------------------------------------------------------------------------
    # KROK 3: Walidacja krzyżowa 3-Fold CV i dobór hiperparametrów PER UŻYTKOWNIK
    # -------------------------------------------------------------------------
    cf_configs = get_cf_configs_grid()
    print(
        f"\n[3/5] Indywidualny dobór hiperparametrów (3-Fold CV, {len(cf_configs)} konfiguracji, "
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

    # Podsumowanie ile razy wybrane zostało każde pojedyncze ustawienie (hiperparametr)
    best_params_list = list(cf_tuning["best_params_per_user"].values())
    for param_name in ["similarity_metric", "aggregation", "k_neighbors", "positive_only", "sim_exponent"]:
        counts = dict(Counter(p[param_name] for p in best_params_list).most_common())
        print(f"  - Wybory dla [{param_name:<17}]: {counts}")

    # -------------------------------------------------------------------------
    # KROK 4: Predykcja dla zbioru task.csv i zapis submission.csv
    # -------------------------------------------------------------------------
    print("\n[4/5] Wyznaczanie podobieństw na pełnym zbiorze train.csv i predykcja dla task.csv...")
    best_params_per_user = cf_tuning["best_params_per_user"]
    default_params = cf_configs[cf_tuning["default_config_name"]]

    full_sim_cache = {}
    full_preds_matrix = np.zeros_like(R_matrix, dtype=np.int64)

    for u_id in user_ids:
        u_idx = user_to_idx[u_id]
        u_params = best_params_per_user.get(u_id, default_params)
        sim_key = (
            u_params.get("similarity_metric", "jaccard_adjcos"),
            float(u_params.get("shrinkage", 0.0)),
            int(u_params.get("min_common", 2)),
        )
        if sim_key not in full_sim_cache:
            full_sim_cache[sim_key] = compute_user_similarity_matrix(
                R_matrix,
                metric=sim_key[0],
                shrinkage=sim_key[1],
                min_common=sim_key[2],
            )

        recommender = UserBasedCFRecommender(**u_params)
        recommender.fit(R_matrix, precomputed_sim=full_sim_cache[sim_key])
        full_preds_matrix[u_idx] = recommender.predict_for_user(u_idx)

    task_u_indices = [user_to_idx[int(u)] for u in task_df["user_id"]]
    task_m_indices = [movie_to_idx[int(m)] for m in task_df["movie_id"]]
    task_predictions = full_preds_matrix[task_u_indices, task_m_indices]

    save_submission_file(task_df, task_predictions, "submission.csv")
    print("  - Zapisano plik: submission.csv")

    # -------------------------------------------------------------------------
    # KROK 5: Końcowa weryfikacja pliku submission.csv
    # -------------------------------------------------------------------------
    print("\n[5/5] Weryfikacja wygenerowanego pliku submission.csv...")
    report = verify_submission_output("task.csv", "submission.csv")
    print(f"  - submission.csv: {report}")

    elapsed = time.time() - start_time
    print(f"\nProces zakończony pomyślnie w {elapsed:.2f} s.")


if __name__ == "__main__":
    main()
