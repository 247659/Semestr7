"""
Moduł narzędziowy (utils.py) zawierający:
1. Zestaw testów jednostkowych i integracyjnych (pkt 6.1 specyfikacji):
   - poprawność obliczania Gini/Entropy,
   - poprawność wyznaczania podziałów,
   - poprawność budowy drzewa,
   - poprawność predykcji,
   - poprawność działania bootstrap sampling,
   - poprawność agregacji wyników lasu,
   - obsługę brakujących wartości w metadanych TMDB,
   - obsługę użytkowników z małą liczbą ocen / jedną klasą ocen,
   - przynależność wszystkich predykcji do klas 0..5.
2. Zapis oraz automatyczną weryfikację plików wynikowych (pkt 6.4 specyfikacji):
   - submission_tree.csv,
   - submission_forest.csv,
   - tree.png.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any
from decision_tree import DecisionTreeClassifierCustom
from random_forest import RandomForestClassifierCustom
from feature_engineering import MovieFeatureEngineer


def run_all_unit_tests() -> Dict[str, bool]:
    """
    Uruchamia kompletny zestaw testów weryfikujących poprawność implementacji
    przed uruchomieniem właściwego eksperymentu.
    """
    results: Dict[str, bool] = {}

    # 1. Test poprawności obliczania Gini i Entropy
    pure_counts = np.array([0, 0, 0, 50, 0, 0], dtype=np.int64)
    two_equal = np.array([25, 25, 0, 0, 0, 0], dtype=np.int64)
    six_equal = np.array([10, 10, 10, 10, 10, 10], dtype=np.int64)

    assert np.isclose(DecisionTreeClassifierCustom.compute_gini(pure_counts), 0.0)
    assert np.isclose(DecisionTreeClassifierCustom.compute_gini(two_equal), 0.5)
    assert np.isclose(
        DecisionTreeClassifierCustom.compute_gini(six_equal), 1.0 - 1.0 / 6.0
    )

    assert np.isclose(DecisionTreeClassifierCustom.compute_entropy(pure_counts), 0.0)
    assert np.isclose(DecisionTreeClassifierCustom.compute_entropy(two_equal), 1.0)
    assert np.isclose(
        DecisionTreeClassifierCustom.compute_entropy(six_equal), np.log2(6.0)
    )
    results["1_gini_entropy_calculation"] = True

    # 2. Test poprawności wyznaczania podziałów (cecha ciągła i binarna)
    X_split = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0],
        ],
        dtype=np.float64,
    )
    y_split = np.array([1, 1, 1, 5, 5, 5], dtype=np.int64)
    tree_split = DecisionTreeClassifierCustom(
        criterion="gini",
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        random_state=42,
    )
    tree_split.fit(
        X_split,
        y_split,
        feature_names=["cont_feat", "bin_feat"],
        is_binary_feature=np.array([False, True]),
    )
    assert not tree_split.root_.is_leaf
    assert tree_split.root_.left.is_leaf and tree_split.root_.right.is_leaf
    assert tree_split.root_.left.predicted_class == 1
    assert tree_split.root_.right.predicted_class == 5
    results["2_split_selection"] = True

    # 3. Test poprawności budowy drzewa i respektowania ograniczeń (max_depth, min_samples_leaf)
    rng = np.random.default_rng(123)
    X_syn = rng.normal(size=(90, 12))
    y_syn = rng.integers(0, 6, size=90)
    tree_depth = DecisionTreeClassifierCustom(
        criterion="entropy",
        max_depth=3,
        min_samples_split=6,
        min_samples_leaf=4,
        random_state=123,
    )
    tree_depth.fit(X_syn, y_syn)
    n_nodes, n_leaves, actual_depth = tree_depth.count_nodes_and_leaves()
    assert actual_depth <= 3 and n_nodes >= 1 and n_leaves >= 1
    # Sprawdzenie min_samples_leaf w każdym liściu
    stack = [tree_depth.root_]
    while stack:
        nd = stack.pop()
        if nd.is_leaf:
            assert nd.n_samples >= 4
        else:
            stack.extend([nd.left, nd.right])
    results["3_tree_construction_constraints"] = True

    # 4. Test poprawności predykcji oraz przynależności do klas 0..5
    preds_tree = tree_depth.predict(X_syn)
    probs_tree = tree_depth.predict_proba(X_syn)
    assert preds_tree.shape == (90,)
    assert probs_tree.shape == (90, 6)
    assert np.all(np.isin(preds_tree, [0, 1, 2, 3, 4, 5]))
    assert np.allclose(np.sum(probs_tree, axis=1), 1.0)
    results["4_tree_predictions_range_0_5"] = True

    # 5. Test poprawności działania bootstrap sampling oraz różnorodności w Random Forest
    rf = RandomForestClassifierCustom(
        n_estimators=15,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        voting="soft",
        random_state=42,
    )
    rf.fit(X_syn, y_syn)
    assert len(rf.trees_) == 15
    # Sprawdzenie, że drzewa w lesie różnią się między sobą (dzięki bootstrap + random subspaces)
    root_feats = {t.root_.feature_idx for t in rf.trees_ if not t.root_.is_leaf}
    assert len(root_feats) > 1
    results["5_bootstrap_and_forest_diversity"] = True

    # 6. Test poprawności agregacji wyników lasu (soft voting vs hard voting)
    preds_soft = rf.predict(X_syn)
    rf.voting = "hard"
    preds_hard = rf.predict(X_syn)
    assert preds_soft.shape == (90,) and preds_hard.shape == (90,)
    assert np.all(np.isin(preds_soft, [0, 1, 2, 3, 4, 5]))
    assert np.all(np.isin(preds_hard, [0, 1, 2, 3, 4, 5]))
    results["6_forest_aggregation_soft_and_hard"] = True

    # 7. Test obsługi brakujących wartości w metadanych TMDB
    dummy_movies = pd.DataFrame(
        {"id": [1, 2], "tmdb_id": [101, 102], "title": ["Movie A", "Movie B"]}
    )
    dummy_tmdb = {
        1: {},  # całkowicie pusty rekord
        2: {"budget": 0, "revenue": None, "runtime": None, "release_date": ""},
    }
    fe = MovieFeatureEngineer()
    X_m, _, _, _ = fe.fit_transform(dummy_movies, dummy_tmdb)
    assert X_m.shape[0] == 2
    assert not np.any(np.isnan(X_m)) and not np.any(np.isinf(X_m))
    results["7_missing_metadata_imputation"] = True

    # 8. Test obsługi użytkowników z małą liczbą ocen / stałą klasą ocen
    X_small = rng.normal(size=(3, 10))
    y_const = np.array([4, 4, 4], dtype=np.int64)
    tree_const = DecisionTreeClassifierCustom(random_state=1).fit(X_small, y_const)
    rf_const = RandomForestClassifierCustom(n_estimators=5, random_state=1).fit(
        X_small, y_const
    )
    assert np.all(tree_const.predict(X_syn[:10]) == 4)
    assert np.all(rf_const.predict(X_syn[:10]) == 4)

    # Użytkownik z pojedynczą próbką
    X_one = rng.normal(size=(1, 10))
    y_one = np.array([2], dtype=np.int64)
    assert DecisionTreeClassifierCustom().fit(X_one, y_one).predict(X_syn[:5])[0] == 2
    assert RandomForestClassifierCustom(n_estimators=5).fit(X_one, y_one).predict(X_syn[:5])[0] == 2
    results["8_small_and_constant_user_edge_cases"] = True

    return results


def save_submission_file(
    task_df: pd.DataFrame, predictions: np.ndarray, output_path: str
) -> None:
    """
    Zapisuje plik wynikowy CSV zachowując oryginalną strukturę task.csv:
    - identyczna kolejność rekordów i kolumny (id;user_id;movie_id;rating),
    - zachowanie istniejących ocen, jeżeli nie są NULL,
    - zastąpienie wyłącznie wartości NULL przewidzianymi ocenami całkowitymi 0..5,
    - brak wiersza nagłówkowego i separator ';'.
    """
    out_df = task_df.copy()
    null_mask = out_df["rating"].isna()
    preds_int = np.asarray(predictions, dtype=np.int64)

    if len(preds_int) == len(out_df):
        out_df.loc[null_mask, "rating"] = preds_int[null_mask]
    elif len(preds_int) == int(np.sum(null_mask)):
        out_df.loc[null_mask, "rating"] = preds_int
    else:
        raise ValueError("Niezgodny wymiar wektora predykcji względem task_df.")

    out_df["rating"] = out_df["rating"].astype(int)
    out_df.to_csv(output_path, sep=";", header=False, index=False)


def verify_generated_outputs(
    task_path: str = "task.csv",
    sub_tree_path: str = "submission_tree.csv",
    sub_forest_path: str = "submission_forest.csv",
    tree_img_path: str = "tree.png",
) -> Dict[str, Any]:
    """
    Przeprowadza końcową weryfikację wygenerowanych plików (pkt 6.4 specyfikacji):
    - istnienie i rozmiar plików,
    - zgodność liczby rekordów i ich kolejności z task.csv,
    - brak wartości NULL,
    - poprawność zakresu ocen 0..5,
    - poprawność wygenerowania pliku graficznego tree.png.
    """
    orig_task = pd.read_csv(
        task_path,
        sep=";",
        header=None,
        names=["id", "user_id", "movie_id", "rating"],
        na_values=["NULL", "null", ""],
    )

    report: Dict[str, Any] = {}
    for label, path in [
        ("submission_tree.csv", sub_tree_path),
        ("submission_forest.csv", sub_forest_path),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Plik wynikowy {path} nie został utworzony!")

        sub_df = pd.read_csv(
            path,
            sep=";",
            header=None,
            names=["id", "user_id", "movie_id", "rating"],
            na_values=["NULL", "null", ""],
        )

        assert len(sub_df) == len(orig_task), f"Niezgodna liczba wierszy w {label}"
        assert (sub_df["id"].values == orig_task["id"].values).all(), (
            f"Niezgodna kolejność id w {label}"
        )
        assert (sub_df["user_id"].values == orig_task["user_id"].values).all(), (
            f"Niezgodna kolumna user_id w {label}"
        )
        assert (sub_df["movie_id"].values == orig_task["movie_id"].values).all(), (
            f"Niezgodna kolumna movie_id w {label}"
        )
        assert sub_df["rating"].isna().sum() == 0, f"Występują wartości NULL w {label}"

        # Sprawdzenie zachowania pierwotnych niepustych ocen (jeśli występowały)
        non_null_mask = orig_task["rating"].notna()
        if non_null_mask.any():
            assert (
                sub_df.loc[non_null_mask, "rating"].astype(int).values
                == orig_task.loc[non_null_mask, "rating"].astype(int).values
            ).all(), f"Zmodyfikowano istniejące oceny w {label}"

        valid_range = sub_df["rating"].isin([0, 1, 2, 3, 4, 5]).all()
        assert valid_range, f"Oceny poza zakresem 0..5 w {label}"

        dist = sub_df["rating"].value_counts().sort_index().to_dict()
        report[label] = {
            "rows": int(len(sub_df)),
            "nulls": int(sub_df["rating"].isna().sum()),
            "valid_range_0_5": bool(valid_range),
            "file_size_bytes": int(os.path.getsize(path)),
            "rating_distribution": {int(k): int(v) for k, v in dist.items()},
        }

    assert os.path.exists(tree_img_path), "Plik tree.png nie został utworzony!"
    img_size = os.path.getsize(tree_img_path)
    assert img_size > 10_000, "Plik tree.png jest zbyt mały lub uszkodzony!"
    report["tree.png"] = {
        "exists": True,
        "file_size_bytes": int(img_size),
    }

    return report
