import time
import argparse
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

from tmdb_client import TMDBRawClient
from feature_processor import RawFeatureProcessor
from similarity import build_component_similarity_matrices
from knn_classifier import UserKNNClassifier

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predykcja ocen filmów kNN z agregacją weighted_mean i funkcją straty MAE."
    )
    parser.add_argument("--cache-file", default="tmdb_raw.json", help="Cache surowych danych TMDB")
    parser.add_argument("--api-key", default=None, help="Klucz API v3 TMDB")
    parser.add_argument("--bearer-token", default=None, help="Token Bearer API v4 TMDB")
    args = parser.parse_args()

    movies_df = pd.read_csv(
        "movie.csv", sep=";", header=None, names=["movie_id", "tmdb_id", "title"],
        dtype={"movie_id": "Int64", "tmdb_id": "Int64"},
    )
    train_df = pd.read_csv(
        "train.csv", sep=";", header=None, names=["id", "user_id", "movie_id", "rating"],
        na_values=["NULL", "null", ""],
    )
    task_df = pd.read_csv(
        "task.csv", sep=";", header=None, names=["id", "user_id", "movie_id", "rating"],
        na_values=["NULL", "null", ""],
    )

    movies_df["movie_id"] = movies_df["movie_id"].astype(int)
    movies_df["tmdb_id"] = movies_df["tmdb_id"].astype(int)
    for frame in (train_df, task_df):
        for col in ("id", "user_id", "movie_id"):
            frame[col] = frame[col].astype(int)

    missing_task_df = task_df.loc[task_df["rating"].isna()].copy()
    print(
        f"[INFO] Wczytano {len(movies_df)} filmów, {len(train_df)} ocen treningowych "
        f"({train_df['user_id'].nunique()} użytkowników), {len(task_df)} wierszy task "
    )

    raw_client = TMDBRawClient(
        api_key=args.api_key,
        bearer_token=args.bearer_token,
        raw_cache_path=args.cache_file,
    )
    raw_movies_data = raw_client.load_or_fetch_raw(movies_df)
    print("[INFO] Przetwarzanie surowych danych TMDB i ekstrakcja cech...")
    features_by_movie = RawFeatureProcessor.transform_all(raw_movies_data, movies_df)

    sorted_movie_ids = sorted(movies_df["movie_id"].astype(int).tolist())
    movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(sorted_movie_ids)}
    sim_matrices = build_component_similarity_matrices(sorted_movie_ids, features_by_movie)
    print(f"[INFO] Wygenerowano {len(sim_matrices)} macierzy podobieństwa.")

    best_aggregation = "weighted_mean"
    best_loss = "mae"

    all_vals, all_counts = np.unique(train_df["rating"].to_numpy(dtype=int), return_counts=True)
    global_mode = int(all_vals[np.argmax(all_counts)]) if len(all_vals) else 3

    train_grouped = {int(uid): group.copy() for uid, group in train_df.groupby("user_id", sort=False)}
    predictions_map: Dict[int, int] = {}
    cv_losses: List[float] = []
    chosen_k_counter: Counter = Counter()
    chosen_feat_counter: Counter = Counter()
    cold_start_users = set()
    start = time.time()

    for user_id, user_task in missing_task_df.groupby("user_id", sort=False):
        user_id = int(user_id)
        if user_id not in train_grouped:
            cold_start_users.add(user_id)
            preds = np.full(len(user_task), global_mode, dtype=int)
        else:
            user_train = train_grouped[user_id]
            train_m_idx = np.array([movie_id_to_idx[int(m)] for m in user_train["movie_id"]], dtype=int)
            train_y = user_train["rating"].to_numpy(dtype=int)
            clf = UserKNNClassifier(
                user_id=user_id,
                aggregation=best_aggregation,
                loss_name=best_loss,
            )
            clf.fit(train_m_idx, train_y, sim_matrices)
            if np.isfinite(clf.cv_loss):
                cv_losses.append(clf.cv_loss)
            chosen_k_counter[clf.best_k] += 1
            for feature in clf.selected_features:
                chosen_feat_counter[feature] += 1
            query_indices = np.array([movie_id_to_idx[int(m)] for m in user_task["movie_id"]], dtype=int)
            preds = clf.predict(query_indices, sim_matrices)

        for row_id, prediction in zip(user_task["id"], preds):
            predictions_map[int(row_id)] = int(np.clip(prediction, 0, 5))

    submission_df = task_df.copy()
    missing_mask = submission_df["rating"].isna()
    if missing_mask.any():
        submission_df.loc[missing_mask, "rating"] = submission_df.loc[missing_mask, "id"].map(predictions_map)
    unresolved = submission_df.loc[missing_mask & submission_df["rating"].isna(), "id"].tolist()
    if unresolved:
        raise RuntimeError(f"Nie wygenerowano predykcji dla wierszy task.csv: {unresolved[:20]}")

    submission_df["rating"] = pd.to_numeric(submission_df["rating"], errors="raise").astype(int)
    if not submission_df["rating"].between(0, 5).all():
        raise RuntimeError("Predykcje poza dopuszczalnym zakresem 0..5.")
    submission_df.to_csv("submission.csv", sep=";", header=False, index=False)

    print(f"[INFO] Rozkład wybranego k: {dict(chosen_k_counter.most_common())}")
    print(f"[INFO] Najczęściej wybierane cechy: {chosen_feat_counter.most_common(10)}")


if __name__ == "__main__":
    main()