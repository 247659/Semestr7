import os
import numpy as np
import pandas as pd
from typing import Dict, Any


def save_submission_file(
    task_df: pd.DataFrame, predictions: np.ndarray, output_path: str
) -> None:
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
) -> Dict[str, Any]:

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
            "nulls": int(sub_df["rating"].isna().sum()),
            "valid_range_0_5": bool(valid_range),
            "rating_distribution": {int(k): int(v) for k, v in dist.items()},
        }

    return report
