"""
Główny moduł wykonawczy (main.py) realizujący pełny proces:
1. Uruchomienie testów jednostkowych i integracyjnych (pkt 6.1).
2. Wczytanie danych movie.csv, train.csv, task.csv oraz lokalnego pliku tmdb_raw.json.
3. Ekstrakcja i budowa macierzy cech filmów (Feature Engineering).
4. Walidacja krzyżowa 3-Fold Per-User CV oraz porównanie hiperparametrów
   pojedynczego drzewa decyzyjnego i lasu losowego (pkt 6.2).
5. Trening finalnych modeli indywidualnych dla każdego z 358 użytkowników
   oraz wygenerowanie plików submission_tree.csv, submission_forest.csv i tree.png (pkt 6.3).
6. Automatyczna weryfikacja końcowa wygenerowanych plików (pkt 6.4).
"""

import time
import numpy as np

from data_loader import load_csv_datasets, load_and_link_tmdb_data
from feature_engineering import MovieFeatureEngineer
from decision_tree import DecisionTreeClassifierCustom
from random_forest import RandomForestClassifierCustom
from evaluation import evaluate_per_user_cv
from visualization import plot_decision_tree
from utils import (
    run_all_unit_tests,
    save_submission_file,
    verify_generated_outputs,
)


def main() -> None:
    start_time = time.time()
    print("=" * 75)
    print("ETAP 2: SYSTEM REKOMENDACJI FILMÓW (DECISION TREE & RANDOM FOREST)")
    print("=" * 75)

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
    # KROK 3: Walidacja krzyżowa 3-Fold CV per użytkownik (pkt 6.2)
    # -------------------------------------------------------------------------
    print("\n[3/6] Przeprowadzanie walidacji krzyżowej 3-Fold CV (358 użytkowników)...")

    tree_configs = {
        "DT_gini_depth4": {
            "criterion": "gini",
            "max_depth": 4,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
            "min_impurity_decrease": 0.005,
        },
        "DT_entropy_depth4": {
            "criterion": "entropy",
            "max_depth": 4,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
            "min_impurity_decrease": 0.005,
        },
        "DT_gini_depth6": {
            "criterion": "gini",
            "max_depth": 6,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "min_impurity_decrease": 0.002,
        },
    }

    forest_configs = {
        "RF_50trees_soft": {
            "n_estimators": 50,
            "criterion": "gini",
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "voting": "soft",
        },
        "RF_50trees_hard": {
            "n_estimators": 50,
            "criterion": "gini",
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "voting": "hard",
        },
    }

    best_tree_name, best_tree_res = None, None
    for name, params in tree_configs.items():
        res = evaluate_per_user_cv(
            train_df,
            X_movies,
            movie_id_to_row,
            feature_names,
            is_binary_feature,
            model_type="tree",
            model_params=params,
            n_splits=3,
            random_state=42,
        )
        print(
            f"  [Tree: {name}] Global Acc: {res['global_accuracy']:.4f} | "
            f"Weighted F1: {res['global_weighted_f1']:.4f} | "
            f"Macro F1: {res['global_macro_f1']:.4f} | MAE: {res['global_mae']:.4f} | "
            f"User Acc (mean±std): {res['mean_user_accuracy']:.4f}±{res['std_user_accuracy']:.4f}"
        )
        if (
            best_tree_res is None
            or res["global_weighted_f1"] > best_tree_res["global_weighted_f1"]
        ):
            best_tree_name, best_tree_res = name, res

    best_forest_name, best_forest_res = None, None
    for name, params in forest_configs.items():
        res = evaluate_per_user_cv(
            train_df,
            X_movies,
            movie_id_to_row,
            feature_names,
            is_binary_feature,
            model_type="forest",
            model_params=params,
            n_splits=3,
            random_state=42,
        )
        print(
            f"  [Forest: {name}] Global Acc: {res['global_accuracy']:.4f} | "
            f"Weighted F1: {res['global_weighted_f1']:.4f} | "
            f"Macro F1: {res['global_macro_f1']:.4f} | MAE: {res['global_mae']:.4f} | "
            f"User Acc (mean±std): {res['mean_user_accuracy']:.4f}±{res['std_user_accuracy']:.4f}"
        )
        if (
            best_forest_res is None
            or res["global_weighted_f1"] > best_forest_res["global_weighted_f1"]
        ):
            best_forest_name, best_forest_res = name, res

    print(f"\n  Najlepsza konfiguracja Drzewa Decyzyjnego: {best_tree_name}")
    print("  Macierz pomyłek (Confusion Matrix 6x6 - wiersze: true, kolumny: pred):")
    print(best_tree_res["confusion_matrix"])

    print(f"\n  Najlepsza konfiguracja Lasu Losowego: {best_forest_name}")
    print("  Macierz pomyłek (Confusion Matrix 6x6 - wiersze: true, kolumny: pred):")
    print(best_forest_res["confusion_matrix"])

    # -------------------------------------------------------------------------
    # KROK 4: Trening finalnych modeli per użytkownik i predykcja task.csv (pkt 6.3)
    # -------------------------------------------------------------------------
    print("\n[4/6] Trenowanie finalnych modeli na pełnym zbiorze train.csv i predykcja task.csv...")
    final_tree_params = tree_configs[best_tree_name]
    final_forest_params = forest_configs[best_forest_name]

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

            tree_clf = DecisionTreeClassifierCustom(
                random_state=42 + u_id, **final_tree_params
            )
            tree_clf.fit(X_train_u, y_train_u, feature_names, is_binary_feature)
            preds_tree_all[task_indices] = tree_clf.predict(X_test_u)

            if u_id == viz_user_id:
                viz_tree_model = tree_clf

            forest_clf = RandomForestClassifierCustom(
                random_state=1000 + u_id, **final_forest_params
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
    print(f"\n[5/6] Generowanie wizualizacji drzewa decyzyjnego (user_id={viz_user_id}) -> tree.png...")
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
