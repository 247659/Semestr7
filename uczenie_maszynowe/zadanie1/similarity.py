from typing import Dict, List, Set, Tuple, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_numeric_similarity_matrix(values: List[Any]) -> np.ndarray:
    # Znormalizowana miara podobieństwa [0, 1] dla cechy liczbowej: 1 - |x - y| / (max - min).
    arr = np.array([float(v) if v is not None else np.nan for v in values], dtype=float)
    valid_mask = ~np.isnan(arr)
    n = len(arr)
    if np.sum(valid_mask) < 2:
        return np.eye(n, dtype=float)

    vmin = np.min(arr[valid_mask])
    vmax = np.max(arr[valid_mask])
    if vmax <= vmin:
        return np.ones((n, n), dtype=float)

    median_val = np.median(arr[valid_mask])
    arr_filled = np.where(valid_mask, arr, median_val)

    diff = np.abs(arr_filled[:, None] - arr_filled[None, :])
    sim = 1.0 - (diff / (vmax - vmin))
    return np.clip(sim, 0.0, 1.0)


def compute_set_similarity_matrix(sets_list: List[Set[str]]) -> np.ndarray:
    # Współczynnik podobieństwa Jaccarda [0, 1] dla cech reprezentowanych jako zbiory.
    n = len(sets_list)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        set_i = sets_list[i]
        if not set_i:
            continue
        for j in range(i + 1, n):
            set_j = sets_list[j]
            if not set_j:
                continue
            inter = len(set_i & set_j)
            if inter > 0:
                union = len(set_i | set_j)
                val = inter / union
                sim[i, j] = val
                sim[j, i] = val
    return sim


def compute_top_people_similarity_matrix(people_lists: List[Set[str]]) -> np.ndarray:
    # Miara podobieństwa z nasyceniem dla głównych aktorów (set_cast): 1 osoba = 0.70, >=2 = 1.00."""
    n = len(people_lists)
    sim = np.zeros((n, n), dtype=float)

    for i in range(n):
        sim[i, i] = 1.0
        set_i = people_lists[i]
        if not set_i:
            continue
        for j in range(i + 1, n):
            set_j = people_lists[j]
            if not set_j:
                continue

            common = len(set_i & set_j)
            if common >= 2:
                val = 1.0
            elif common == 1:
                val = 0.70
            else:
                val = 0.0

            sim[i, j] = val
            sim[j, i] = val

    return sim


def compute_categorical_similarity_matrix(categories: List[Any]) -> np.ndarray:
    # Znormalizowana miara podobieństwa [0, 1] dla cech kategorycznych.
    n = len(categories)
    sim = np.zeros((n, n), dtype=float)

    normalized_sets: List[Set[str]] = []
    for c in categories:
        if c is None or c == "":
            normalized_sets.append(set())
        elif isinstance(c, (set, list, tuple)):
            normalized_sets.append({str(x).strip().lower() for x in c if str(x).strip()})
        else:
            normalized_sets.append({str(c).strip().lower()})

    for i in range(n):
        sim[i, i] = 1.0
        set_i = normalized_sets[i]
        if not set_i:
            continue
        for j in range(i + 1, n):
            set_j = normalized_sets[j]
            if not set_j:
                continue
            inter = len(set_i & set_j)
            if inter > 0:
                val = inter / len(set_i | set_j)
                sim[i, j] = val
                sim[j, i] = val
    return sim


def compute_text_similarity_matrix(texts: List[str]) -> np.ndarray:
    # Oblicza macierz podobieństwa cosinusowego [0, 1] na reprezentacji TF-IDF.
    if not texts:
        return np.empty((0, 0), dtype=float)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        sim = np.zeros((len(texts), len(texts)), dtype=float)
        np.fill_diagonal(sim, 1.0)
        return sim

    sim = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, 0.0, 1.0)


def build_component_similarity_matrices(movie_ids: List[int], features_by_movie: Dict[int, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    # Buduje słownik macierzy podobieństwa (N x N) w skali [0, 1] posortowany wg wariancji.
    all_feature_keys: Set[str] = set()
    for f_dict in features_by_movie.values():
        all_feature_keys.update(f_dict.keys())

    temp_matrices: List[Tuple[float, str, np.ndarray]] = []
    diag_mask = ~np.eye(len(movie_ids), dtype=bool)

    for feat_name in all_feature_keys:
        raw_values = [features_by_movie[m_id].get(feat_name) for m_id in movie_ids]

        if feat_name == "set_cast":
            safe_sets = [s if isinstance(s, set) else set() for s in raw_values]
            mat = compute_top_people_similarity_matrix(safe_sets)
        elif feat_name.startswith("num_"):
            mat = compute_numeric_similarity_matrix(raw_values)
        elif feat_name.startswith("set_"):
            safe_sets = [s if isinstance(s, set) else set() for s in raw_values]
            mat = compute_set_similarity_matrix(safe_sets)
        elif feat_name.startswith("cat_"):
            safe_cats = [c if c is not None else "" for c in raw_values]
            mat = compute_categorical_similarity_matrix(safe_cats)
        elif feat_name.startswith("text_"):
            safe_texts = [str(t) if t is not None else "" for t in raw_values]
            mat = compute_text_similarity_matrix(safe_texts)
        else:
            continue

        off_diag = mat[diag_mask]
        std_val = float(np.std(off_diag)) if len(off_diag) > 0 else 0.0
        if std_val > 1e-5:
            temp_matrices.append((std_val, feat_name, mat))

    temp_matrices.sort(key=lambda x: x[0], reverse=True)
    return {feat_name: mat for _, feat_name, mat in temp_matrices}