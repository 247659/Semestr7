import os
import numpy as np
import pandas as pd
from typing import Dict, Any

def save_submission_file(
    task_df: pd.DataFrame, predictions: np.ndarray, output_path: str = "submission.csv"
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


def verify_submission_output(
    task_path: str = "task.csv",
    submission_path: str = "submission.csv",
) -> Dict[str, Any]:

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Plik wynikowy {submission_path} nie został utworzony!")

    orig_task = pd.read_csv(
        task_path,
        sep=";",
        header=None,
        names=["id", "user_id", "movie_id", "rating"],
        na_values=["NULL", "null", ""],
    )
    sub_df = pd.read_csv(
        submission_path,
        sep=";",
        header=None,
        names=["id", "user_id", "movie_id", "rating"],
        na_values=["NULL", "null", ""],
    )

    assert len(sub_df) == len(orig_task), "Niezgodna liczba wierszy w submission.csv"
    assert (sub_df["id"].values == orig_task["id"].values).all(), "Niezgodne id"
    assert (sub_df["user_id"].values == orig_task["user_id"].values).all(), "Niezgodne user_id"
    assert (sub_df["movie_id"].values == orig_task["movie_id"].values).all(), "Niezgodne movie_id"
    assert sub_df["rating"].isna().sum() == 0, "Występują wartości NULL w submission.csv"

    valid_range = sub_df["rating"].isin([0, 1, 2, 3, 4, 5]).all()
    assert valid_range, "Oceny poza dozwolonym zakresem {0, 1, 2, 3, 4, 5}"

    dist = sub_df["rating"].value_counts().sort_index().to_dict()
    return {
        "rows": int(len(sub_df)),
        "nulls": int(sub_df["rating"].isna().sum()),
        "valid_range_0_5": bool(valid_range),
        "rating_distribution": {int(k): int(v) for k, v in dist.items()},
    }