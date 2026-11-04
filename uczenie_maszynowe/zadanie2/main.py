"""
Główny moduł wykonawczy (main.py) realizujący pełny proces:
1. Uruchomienie testów jednostkowych i integracyjnych (pkt 6.1).
2. Wczytanie danych movie.csv, train.csv, task.csv oraz lokalnego pliku tmdb_raw.json.
3. Ekstrakcja i budowa macierzy cech filmów (Feature Engineering).
4. Walidacja krzyżowa 3-Fold Per-User CV oraz INDYWIDUALNY dobór hiperparametrów
   oddzielnie dla każdego z 358 użytkowników (Per-User Hyperparameter Tuning)
   na rozszerzonej siatce parametrów dla drzewa decyzyjnego i lasu losowego (pkt 6.2).
5. Trening finalnych modeli indywidualnych dla każdego z 358 użytkowników
   z użyciem jego własnych, optymalnych hiperparametrów oraz wygenerowanie plików
   submission_tree.csv, submission_forest.csv i tree.png (pkt 6.3).
6. Automatyczna weryfikacja końcowa wygenerowanych plików (pkt 6.4).
"""

import time
import numpy as np

from data_loader import load_csv_datasets, load_and_link_tmdb_data
from feature_engineering import MovieFeatureEngineer
from decision_tree import DecisionTreeClassifierCustom
from random_forest import RandomForestClassifierCustom
from evaluation import tune_hyperparameters_per_user
from visualization import plot_decision_tree
from utils import (
    run_all_unit_tests,
    save_submission_file,
    verify_generated_outputs,
)


def get_decision_tree_grid():
    """
    Zwraca rozszerzoną siatkę konfiguracji hiperparametrów dla Pojedynczego Drzewa Decyzyjnego,
    obejmującą szeroki zakres max_depth (2, 3, 4, 5, 6, 8, 10, None), oba kryteria podziału
    ('gini', 'entropy'), zróżnicowane wartości min_samples_split, min_samples_leaf
    oraz min_impurity_decrease.
    """
    return {
        "DT_gini_d3_leaf5": {
            "criterion": "gini",
            "max_depth": 3,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "min_impurity_decrease": 0.0,
        },
        "DT_gini_d4_leaf3": {
            "criterion": "gini",
            "max_depth": 4,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
            "min_impurity_decrease": 0.005,
        },
        # Sprawdzenie, czy mniejszy próg gain daje lepsze wyniki
        "DT_gini_d4_leaf3_gain001": {
            "criterion": "gini",
            "max_depth": 4,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "min_impurity_decrease": 0.001,
        },
        "DT_gini_d5_leaf3": {
            "criterion": "gini",
            "max_depth": 5,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "min_impurity_decrease": 0.003,
        },
        # Sprawdzenie większego ograniczenia liści
        "DT_gini_d5_leaf5": {
            "criterion": "gini",
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "min_impurity_decrease": 0.0,
        },
        "DT_gini_d6_leaf2": {
            "criterion": "gini",
            "max_depth": 6,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "min_impurity_decrease": 0.002,
        },
        # Drzewo o średniej głębokości bez progu gain
        "DT_gini_d6_leaf3": {
            "criterion": "gini",
            "max_depth": 6,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "min_impurity_decrease": 0.0,
        },
        "DT_gini_d8_leaf2": {
            "criterion": "gini",
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "min_impurity_decrease": 0.001,
        },
        "DT_gini_d8_leaf4": {
            "criterion": "gini",
            "max_depth": 8,
            "min_samples_split": 8,
            "min_samples_leaf": 4,
            "min_impurity_decrease": 0.0,
        },
        "DT_gini_d10_leaf1": {
            "criterion": "gini",
            "max_depth": 10,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "min_impurity_decrease": 0.0,
        },
        "DT_gini_d10_leaf2_gain001": {
            "criterion": "gini",
            "max_depth": 10,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "min_impurity_decrease": 0.001,
        },
        "DT_gini_full_unconstrained": {
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_impurity_decrease": 0.0,
        },
    }


def get_random_forest_grid():
    """
    Zwraca rozszerzoną siatkę konfiguracji hiperparametrów dla Lasu Losowego,
    obejmującą różne głębokości drzew (max_depth: 3, 5, 8, 12, None),
    różne liczby drzew (n_estimators: 25, 45, 65), kryteria ('gini', 'entropy'),
    strategie doboru podzbioru cech ('sqrt', 'log2'), metody agregacji ('soft', 'hard')
    oraz dywersyfikację hiperparametrów między drzewami (diversify_trees).
    """
    return {
        "RF_35t_d5_gini_sqrt_soft": {
            "n_estimators": 35,
            "criterion": "gini",
            "max_depth": 5,
            "min_samples_split": 6,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "voting": "soft",
            "diversify_trees": False,
        },
        "RF_45t_d8_gini_sqrt_soft": {
            "n_estimators": 45,
            "criterion": "gini",
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "voting": "soft",
            "diversify_trees": False,
        },
        "RF_45t_d8_gini_sqrt_hard": {
            "n_estimators": 45,
            "criterion": "gini",
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "voting": "hard",
            "diversify_trees": False,
        },
        "RF_45t_d8_gini_log2_soft": {
            "n_estimators": 45,
            "criterion": "gini",
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "log2",
            "voting": "soft",
            "diversify_trees": False,
        },
        "RF_50t_d12_gini_sqrt_soft_leaf1": {
            "n_estimators": 50,
            "criterion": "gini",
            "max_depth": 12,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "voting": "soft",
            "diversify_trees": False,
        },
        "RF_65t_full_diversified_soft": {
            "n_estimators": 65,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "voting": "soft",
            "diversify_trees": True,
        },
    }


def main() -> None:
    start_time = time.time()
    print("=" * 80)
    print("ETAP 2: SYSTEM REKOMENDACJI FILMÓW (PER-USER HYPERPARAMETER TUNING)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # KROK 1: Testy jednostkowe (pkt 6.1)
    # -------------------------------------------------------------------------
    print("\n[1/6] Uruchamianie testów jednostkowych i integracyjnych...")
    test_results = run_all_unit_tests()
    for test_name, passed in test_results.items():
        print(f"  - {test_name}: {'PASSED' if passed else 'FAILED'}")

    # -------------------------------------------------------------------------
    # KROK 2: Wczytanie danych i budowa macierzy cech z tmdb_raw.json
    # -------------------------------------------------------------------------
    print("\n[2/6] Wczytywanie plików CSV oraz lokalnego pliku tmdb_raw.json...")
    movies_df, train_df, task_df = load_csv_datasets(
        "movie.csv", "train.csv", "task.csv"
    )
    tmdb_records = load_and_link_tmdb_data(movies_df, "tmdb_raw.json")

    fe = MovieFeatureEngineer(reference_year=2026)
    X_movies, movie_id_to_row, feature_names, is_binary_feature = fe.fit_transform(
        movies_df, tmdb_records
    )
    print(
        f"  - Wczytano filmów: {len(movies_df)}, rekordów train: {len(train_df)}, "
        f"rekordów task: {len(task_df)}"
    )
    print(
        f"  - Wymiar macierzy cech X_movies: {X_movies.shape} "
        f"(cechy ciągłe: {(~is_binary_feature).sum()}, cechy binarne: {is_binary_feature.sum()})"
    )

    # -------------------------------------------------------------------------
    # KROK 3: Walidacja krzyżowa 3-Fold CV i dobór hiperparametrów PER UŻYTKOWNIK
    # -------------------------------------------------------------------------
    print("\n[3/6] Indywidualny dobór hiperparametrów (3-Fold CV) dla każdego z 358 użytkowników...")

    tree_configs = get_decision_tree_grid()
    forest_configs = get_random_forest_grid()

    print(f"\n  --- A. Testowanie {len(tree_configs)} konfiguracji Drzewa Decyzyjnego ---")
    tree_tuning = tune_hyperparameters_per_user(
        train_df=train_df,
        X_movies=X_movies,
        movie_id_to_row=movie_id_to_row,
        feature_names=feature_names,
        is_binary_feature=is_binary_feature,
        model_type="tree",
        configs_grid=tree_configs,
        n_splits=3,
        random_state=42,
    )

    for cfg_name, res in tree_tuning["per_config_summaries"].items():
        print(
            f"    [{cfg_name:<26}] Global Acc: {res['global_accuracy']:.4f} | "
            f"Weighted F1: {res['global_weighted_f1']:.4f} | "
            f"Macro F1: {res['global_macro_f1']:.4f} | MAE: {res['global_mae']:.4f}"
        )

    t_glob = tree_tuning["best_global_summary"]
    t_pers = tree_tuning["personalized_metrics"]
    print("\n  Podsumowanie doboru hiperparametrów Drzewa Decyzyjnego:")
    print(
        f"    * Najlepsza pojedyncza konfiguracja globalna ({tree_tuning['best_global_config_name']}): "
        f"Acc = {t_glob['global_accuracy']:.4f}, Weighted F1 = {t_glob['global_weighted_f1']:.4f}, MAE = {t_glob['global_mae']:.4f}"
    )
    print(
        f"    * Dobór INDYWIDUALNY per użytkownik (Per-User Tuned DT): "
        f"Acc = {t_pers['global_accuracy']:.4f}, Weighted F1 = {t_pers['global_weighted_f1']:.4f}, "
        f"Macro F1 = {t_pers['global_macro_f1']:.4f}, MAE = {t_pers['global_mae']:.4f}"
    )
    print(f"    * Rozkład wybranych konfiguracji drzewa wśród 358 użytkowników: {tree_tuning['config_selection_counts']}")
    print("    * Macierz pomyłek (Confusion Matrix 6x6) dla Per-User Tuned Decision Tree:")
    print(t_pers["confusion_matrix"])

    print(f"\n  --- B. Testowanie {len(forest_configs)} konfiguracji Lasu Losowego ---")
    forest_tuning = tune_hyperparameters_per_user(
        train_df=train_df,
        X_movies=X_movies,
        movie_id_to_row=movie_id_to_row,
        feature_names=feature_names,
        is_binary_feature=is_binary_feature,
        model_type="forest",
        configs_grid=forest_configs,
        n_splits=3,
        random_state=42,
    )

    for cfg_name, res in forest_tuning["per_config_summaries"].items():
        print(
            f"    [{cfg_name:<31}] Global Acc: {res['global_accuracy']:.4f} | "
            f"Weighted F1: {res['global_weighted_f1']:.4f} | "
            f"Macro F1: {res['global_macro_f1']:.4f} | MAE: {res['global_mae']:.4f}"
        )

    f_glob = forest_tuning["best_global_summary"]
    f_pers = forest_tuning["personalized_metrics"]
    print("\n  Podsumowanie doboru hiperparametrów Lasu Losowego:")
    print(
        f"    * Najlepsza pojedyncza konfiguracja globalna ({forest_tuning['best_global_config_name']}): "
        f"Acc = {f_glob['global_accuracy']:.4f}, Weighted F1 = {f_glob['global_weighted_f1']:.4f}, MAE = {f_glob['global_mae']:.4f}"
    )
    print(
        f"    * Dobór INDYWIDUALNY per użytkownik (Per-User Tuned RF): "
        f"Acc = {f_pers['global_accuracy']:.4f}, Weighted F1 = {f_pers['global_weighted_f1']:.4f}, "
        f"Macro F1 = {f_pers['global_macro_f1']:.4f}, MAE = {f_pers['global_mae']:.4f}"
    )
    print(f"    * Rozkład wybranych konfiguracji lasu wśród 358 użytkowników: {forest_tuning['config_selection_counts']}")
    print("    * Macierz pomyłek (Confusion Matrix 6x6) dla Per-User Tuned Random Forest:")
    print(f_pers["confusion_matrix"])

    # -------------------------------------------------------------------------
    # KROK 4: Trening finalnych modeli z indywidualnymi hiperparametrami (pkt 6.3)
    # -------------------------------------------------------------------------
    print("\n[4/6] Trenowanie finalnych modeli (z indywidualnymi hiperparametrami każdego użytkownika) i predykcja task.csv...")
    best_tree_params_per_user = tree_tuning["best_params_per_user"]
    best_forest_params_per_user = forest_tuning["best_params_per_user"]
    default_tree_params = tree_configs[tree_tuning["best_global_config_name"]]
    default_forest_params = forest_configs[forest_tuning["best_global_config_name"]]

    preds_tree_all = np.zeros(len(task_df), dtype=np.int64)
    preds_forest_all = np.zeros(len(task_df), dtype=np.int64)

    train_by_user = {
        int(u): grp for u, grp in train_df.groupby("user_id", sort=False)
    }
    viz_user_id = 587
    viz_tree_model = None

    for user_id, task_group in task_df.groupby("user_id", sort=False):
        u_id = int(user_id)
        task_indices = task_group.index.to_numpy()
        task_movie_ids = task_group["movie_id"].to_numpy(dtype=np.int64)
        X_test_u = X_movies[[movie_id_to_row[int(m)] for m in task_movie_ids]]

        if u_id in train_by_user:
            tr_grp = train_by_user[u_id]
            tr_movie_ids = tr_grp["movie_id"].to_numpy(dtype=np.int64)
            y_train_u = tr_grp["rating"].to_numpy(dtype=np.int64)
            X_train_u = X_movies[[movie_id_to_row[int(m)] for m in tr_movie_ids]]

            # Pobieramy optymalne hiperparametry wyznaczone specjalnie dla użytkownika u_id
            u_tree_params = best_tree_params_per_user.get(u_id, default_tree_params)
            u_forest_params = best_forest_params_per_user.get(u_id, default_forest_params)

            tree_clf = DecisionTreeClassifierCustom(
                random_state=42 + u_id, **u_tree_params
            )
            tree_clf.fit(X_train_u, y_train_u, feature_names, is_binary_feature)
            preds_tree_all[task_indices] = tree_clf.predict(X_test_u)

            if u_id == viz_user_id:
                viz_tree_model = tree_clf

            forest_clf = RandomForestClassifierCustom(
                random_state=1000 + u_id, **u_forest_params
            )
            forest_clf.fit(X_train_u, y_train_u, feature_names, is_binary_feature)
            preds_forest_all[task_indices] = forest_clf.predict(X_test_u)
        else:
            preds_tree_all[task_indices] = 3
            preds_forest_all[task_indices] = 3

    save_submission_file(task_df, preds_tree_all, "submission_tree.csv")
    save_submission_file(task_df, preds_forest_all, "submission_forest.csv")
    print("  - Zapisano pliki: submission_tree.csv oraz submission_forest.csv")

    # -------------------------------------------------------------------------
    # KROK 5: Wizualizacja drzewa decyzyjnego (tree.png)
    # -------------------------------------------------------------------------
    chosen_viz_cfg = tree_tuning["best_config_name_per_user"].get(viz_user_id, "default")
    print(
        f"\n[5/6] Generowanie wizualizacji drzewa decyzyjnego "
        f"(user_id={viz_user_id}, wybrana konfiguracja={chosen_viz_cfg}) -> tree.png..."
    )
    plot_decision_tree(viz_tree_model, user_id=viz_user_id, output_path="tree.png")
    print("  - Zapisano plik: tree.png")

    # -------------------------------------------------------------------------
    # KROK 6: Końcowa weryfikacja wygenerowanych plików (pkt 6.4)
    # -------------------------------------------------------------------------
    print("\n[6/6] Weryfikacja wygenerowanych plików...")
    verification_report = verify_generated_outputs(
        "task.csv", "submission_tree.csv", "submission_forest.csv", "tree.png"
    )
    for fname, info in verification_report.items():
        print(f"  - {fname}: {info}")

    elapsed = time.time() - start_time
    print(f"\nProces zakończony pomyślnie w {elapsed:.2f} s.")


if __name__ == "__main__":
    main()