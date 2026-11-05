import os
import json
import pandas as pd
from typing import Dict, Tuple, Any


def load_csv_datasets(
    movie_path: str = "movie.csv",
    train_path: str = "train.csv",
    task_path: str = "task.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    with open(movie_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    has_header = first_line.lower().startswith("id;")

    if has_header:
        movies = pd.read_csv(movie_path, sep=";")
        train = pd.read_csv(train_path, sep=";", na_values=["NULL", "null", ""])
        task = pd.read_csv(task_path, sep=";", na_values=["NULL", "null", ""])
    else:
        movies = pd.read_csv(
            movie_path, sep=";", header=None, names=["id", "tmdb_id", "title"]
        )
        train = pd.read_csv(
            train_path,
            sep=";",
            header=None,
            names=["id", "user_id", "movie_id", "rating"],
            na_values=["NULL", "null", ""],
        )
        task = pd.read_csv(
            task_path,
            sep=";",
            header=None,
            names=["id", "user_id", "movie_id", "rating"],
            na_values=["NULL", "null", ""],
        )

    movies["id"] = movies["id"].astype(int)
    movies["tmdb_id"] = movies["tmdb_id"].astype(int)
    train["id"] = train["id"].astype(int)
    train["user_id"] = train["user_id"].astype(int)
    train["movie_id"] = train["movie_id"].astype(int)
    train["rating"] = train["rating"].astype(int)

    task["id"] = task["id"].astype(int)
    task["user_id"] = task["user_id"].astype(int)
    task["movie_id"] = task["movie_id"].astype(int)

    return movies, train, task


def load_and_link_tmdb_data(
    movies_df: pd.DataFrame, tmdb_json_path: str = "tmdb_raw.json"
) -> Dict[int, Dict[str, Any]]:

    if not os.path.exists(tmdb_json_path):
        raise FileNotFoundError(f"Brak pliku {tmdb_json_path} w katalogu roboczym.")

    with open(tmdb_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    by_inner_tmdb_id: Dict[int, Dict[str, Any]] = {}
    by_title: Dict[str, Dict[str, Any]] = {}

    if isinstance(raw_data, dict):
        items = raw_data.items()
    elif isinstance(raw_data, list):
        items = [(str(i), rec) for i, rec in enumerate(raw_data)]
    else:
        items = []

    for key, rec in items:
        if not isinstance(rec, dict):
            continue
        inner_id = rec.get("id")
        if inner_id is not None:
            try:
                by_inner_tmdb_id[int(inner_id)] = rec
            except (ValueError, TypeError):
                pass
        title = rec.get("title") or rec.get("original_title")
        if isinstance(title, str) and title.strip():
            by_title[title.strip().lower()] = rec

    linked_records: Dict[int, Dict[str, Any]] = {}
    for _, row in movies_df.iterrows():
        m_id = int(row["id"])
        t_id = int(row["tmdb_id"])
        title = str(row["title"]).strip().lower()

        rec = None
        if t_id in by_inner_tmdb_id:
            rec = by_inner_tmdb_id[t_id]
        elif isinstance(raw_data, dict) and str(t_id) in raw_data:
            rec = raw_data[str(t_id)]
        elif isinstance(raw_data, dict) and str(m_id) in raw_data:
            rec = raw_data[str(m_id)]
        elif title in by_title:
            rec = by_title[title]

        linked_records[m_id] = rec if isinstance(rec, dict) else {}

    return linked_records
