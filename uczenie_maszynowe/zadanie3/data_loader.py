"""
Moduł odpowiedzialny za wczytywanie danych z plików:
- movie.csv
- train.csv
- task.csv
oraz budowę macierzy ocen użytkownik-film (R_matrix).
Zgodnie ze Wskazówką 1 (Hint 1), cechy filmów (np. z tmdb_raw.json) nie są
brane pod uwagę — wykorzystywane są wyłącznie oceny wystawione przez użytkowników.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List


def load_csv_datasets(
    movie_path: str = "movie.csv",
    train_path: str = "train.csv",
    task_path: str = "task.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Wczytuje pliki movie.csv, train.csv oraz task.csv z separatorem ';'
    i brakiem wiersza nagłówkowego (z obsługą ewentualnego nagłówka).
    """
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


def build_rating_matrix(
    train_df: pd.DataFrame, movies_df: pd.DataFrame
) -> Tuple[np.ndarray, List[int], List[int], Dict[int, int], Dict[int, int]]:
    """
    Buduje macierz ocen R o wymiarach (U, M), gdzie:
    - U to liczba unikalnych użytkowników (358),
    - M to liczba filmów (200),
    - R[u, m] zawiera ocenę z zakresu {0, 1, 2, 3, 4, 5} lub np.nan, jeśli
      użytkownik u nie widział (nie ocenił) filmu m w zbiorze treningowym.
    """
    user_ids = sorted(train_df["user_id"].astype(int).unique())
    movie_ids = sorted(movies_df["id"].astype(int).unique())

    user_to_idx = {u_id: idx for idx, u_id in enumerate(user_ids)}
    movie_to_idx = {m_id: idx for idx, m_id in enumerate(movie_ids)}

    R = np.full((len(user_ids), len(movie_ids)), np.nan, dtype=np.float64)

    u_indices = [user_to_idx[int(u)] for u in train_df["user_id"]]
    m_indices = [movie_to_idx[int(m)] for m in train_df["movie_id"]]
    ratings = train_df["rating"].to_numpy(dtype=np.float64)

    R[u_indices, m_indices] = ratings

    return R, user_ids, movie_ids, user_to_idx, movie_to_idx