import time
import numpy as np

from data_loader import load_csv_datasets, load_and_link_tmdb_data
from feature_engineering import MovieFeatureEngineer
from decision_tree import DecisionTreeClassifierCustom
from random_forest import RandomForestClassifierCustom
from evaluation import tune_hyperparameters_per_user
from visualization import plot_decision_tree
from utils import (
    save_submission_file,
    verify_generated_outputs,
)


def get_decision_tree_grid():
    return {
        "DT_gini_d3_leaf5": {
            "max_depth": 3,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "min_impurity_decrease": 0.0,
        },
        "DT_gini_d4_leaf3": {
            "max_depth": 4,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
            "min_impurity_decrease": 0.005,
        },
        "DT_gini_d4_leaf3_gain001": {
            "max_depth": 4,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "min_impurity_decrease": 0.001,
        },
        "DT_gini_d5_leaf3": {
            "max_depth": 5,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "min_impurity_decrease": 0.003,
        },
        "DT_gini_d5_leaf5": {
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "min_impurity_decrease": 0.0,
        },
        "DT_gini_d6_leaf2": {
            "max_depth": 6,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "min_impurity_decrease": 0.002,
        },
        "DT_gini_d6_leaf3": {
            "max_depth": 6,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "min_impurity_decrease": 0.0,
        },
        "DT_gini_d8_leaf2": {
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "min_impurity_decrease": 0.001,
        },
        "DT_gini_d8_leaf4": {
            "max_depth": 8,
            "min_samples_split": 8,
            "min_samples_leaf": 4,
            "min_impurity_decrease": 0.0,
        },
        "DT_gini_d10_leaf1": {
            "max_depth": 10,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "min_impurity_decrease": 0.0,
        },
        "DT_gini_d10_leaf2_gain001": {
            "max_depth": 10,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "min_impurity_decrease": 0.001,
        },
        "DT_gini_full_unconstrained": {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_impurity_decrease": 0.0,
        },
    }


def get_random_forest_grid():
    return {
        "RF_35t_d5_gini_sqrt_soft": {
            "n_estimators": 35,
            "max_depth": 5,
            "min_samples_split": 6,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "voting": "soft",
            "diversify_trees": False,
        },
        "RF_35t_d5_gini_sqrt_hard": {
            "n_estimators": 35,
            "max_depth": 5,
            "min_samples_split": 6,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "voting": "hard",
            "diversify_trees": False,
        },
        "RF_45t_d8_gini_sqrt_soft": {
            "n_estimators": 45,
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "voting": "soft",
            "diversify_trees": False,
        },
        "RF_45t_d8_gini_sqrt_hard": {
            "n_estimators": 45,
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "voting": "hard",
            "diversify_trees": False,
        },
        "RF_45t_d8_gini_log2_soft": {
            "n_estimators": 45,
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "log2",
            "voting": "soft",
            "diversify_trees": False,
        },
        "RF_50t_d12_gini_sqrt_soft_leaf1": {
            "n_estimators": 50,
            "max_depth": 12,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "voting": "soft",
            "diversify_trees": False,
        },
        "RF_65t_full_diversified_soft": {
            "n_estimators": 65,
            "max_depth": None,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "voting": "soft",
            "diversify_trees": True,
        },
        "RF_65t_full_diversified_hard": {
            "n_estimators": 65,
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

    # -------------------------------------------------------------------------
    # KROK 1: Wczytanie danych i budowa macierzy cech z tmdb_raw.json
    # -------------------------------------------------------------------------
    print("\n[1/5] Wczytywanie plików CSV oraz lokalnego pliku tmdb_raw.json...")
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
    )

    # -------------------------------------------------------------------------
    # KROK 2: Walidacja krzyżowa 3-Fold CV i dobór hiperparametrów PER UŻYTKOWNIK
    # -------------------------------------------------------------------------
    print("\n[2/5] Indywidualny dobór hiperparametrów (3-Fold CV) dla każdego z 358 użytkowników...")

    tree_configs = get_decision_tree_grid()
    forest_configs = get_random_forest_grid()

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
    print(
        f"  - Rozkład wybranych konfiguracji drzewa wśród 358 użytkowników: "
        f"{tree_tuning['config_selection_counts']}"
    )

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
    print(
        f"  - Rozkład wybranych konfiguracji lasu wśród 358 użytkowników: "
        f"{forest_tuning['config_selection_counts']}"
    )

    # -------------------------------------------------------------------------
    # KROK 3: Trening finalnych modeli z indywidualnymi hiperparametrami (pkt 6.3)
    # -------------------------------------------------------------------------
    print(
        "\n[3/5] Trenowanie finalnych modeli (z indywidualnymi hiperparametrami każdego użytkownika) i predykcja task.csv...")
    best_tree_params_per_user = tree_tuning["best_params_per_user"]
    best_forest_params_per_user = forest_tuning["best_params_per_user"]
    default_tree_params = tree_configs[tree_tuning["default_config_name"]]
    default_forest_params = forest_configs[forest_tuning["default_config_name"]]

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
    # KROK 4: Wizualizacja drzewa decyzyjnego
    # -------------------------------------------------------------------------
    chosen_viz_cfg = tree_tuning["best_config_name_per_user"].get(viz_user_id, "default")
    print(
        f"\n[4/5] Generowanie wizualizacji drzewa decyzyjnego "
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