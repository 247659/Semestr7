#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rozwiązanie zadania predykcji ocen filmów metodą kNN (osobny klasyfikator dla każdego użytkownika).

Architektura dwuetapowa dla cech z TMDB:
1. Zapis wszystkich cech w postaci całkowicie surowej (1:1 odpowiedź JSON z TMDB API) do pliku cache.
2. Dopiero po wczytaniu surowych danych wykonywane są operacje czyszczenia, transformacji,
   obliczanie znormalizowanych miar podobieństwa [0, 1] oraz selekcja cech metodą Wrapper (SFS LOOCV).

Generuje pliki:
1. submission.csv
2. code.zip
"""

import os
import json
import math
import time
import zipfile
import argparse
from collections import Counter
from typing import Dict, List, Set, Tuple, Any, Optional

import requests
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================================================================
# 1. POBIERANIE I ZAPIS SUROWYCH DANYCH Z TMDB API (1:1 RAW JSON)
# ==============================================================================

class TMDBRawClient:
    """
    Pobiera pełne, nieprzetworzone dane o filmach z TMDB API (wraz z podzasobami
    credits oraz keywords) i zapisuje je w pliku JSON w postaci surowej (1:1).
    """

    BASE_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"

    def __init__(
        self,
        api_key: Optional[str] = None,
        bearer_token: Optional[str] = None,
        raw_cache_path: str = "tmdb_raw.json",
        language: str = "en-US",
    ):
        self.api_key = api_key or os.environ.get("TMDB_API_KEY", "")
        self.bearer_token = bearer_token or os.environ.get("TMDB_BEARER_TOKEN", "")
        self.raw_cache_path = raw_cache_path
        self.language = language
        self.session = requests.Session()

    def _fetch_raw_movie(self, tmdb_id: int) -> Dict[str, Any]:
        """Pobiera surowy obiekt JSON z TMDB API dla pojedynczego filmu."""
        url = self.BASE_URL.format(tmdb_id=tmdb_id)
        params = {
            "append_to_response": "credits,keywords",
            "language": self.language,
        }
        headers = {"accept": "application/json"}

        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        elif self.api_key:
            params["api_key"] = self.api_key
        else:
            raise RuntimeError(
                "Brak klucza TMDB API! Podaj --api-key, --bearer-token lub ustaw "
                "zmienną środowiskową TMDB_API_KEY / TMDB_BEARER_TOKEN."
            )

        for attempt in range(4):
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=15)
                if response.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"[OSTRZEŻENIE] TMDB ID {tmdb_id} zwrócił status {response.status_code}")
                    return {}
            except requests.RequestException as exc:
                print(f"[BŁĄD SIECI] Próba {attempt + 1} dla TMDB ID {tmdb_id}: {exc}")
                time.sleep(1.0)
        return {}

    def load_or_fetch_raw(self, movies_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Zwraca słownik {movie_id: surowy_json_tmdb}.
        Jeśli film nie znajduje się jeszcze w pliku cache, pobiera go z API
        i zapisuje w postaci całkowicie surowej.
        """
        raw_cache: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.raw_cache_path):
            print(f"[INFO] Wczytywanie surowych danych TMDB z pliku: {self.raw_cache_path}")
            with open(self.raw_cache_path, "r", encoding="utf-8") as f:
                raw_cache = json.load(f)

        updated = False
        total = len(movies_df)

        for idx, row in movies_df.iterrows():
            m_id_str = str(int(row["movie_id"]))
            tmdb_id = int(row["tmdb_id"])
            title = str(row["title"])

            if m_id_str not in raw_cache or not raw_cache[m_id_str]:
                if self.api_key or self.bearer_token:
                    print(f"[TMDB API] Pobieranie surowych danych {idx + 1}/{total}: {title} (TMDB ID: {tmdb_id})")
                    raw_json = self._fetch_raw_movie(tmdb_id)
                else:
                    raw_json = {"title": title, "id": tmdb_id}
                raw_cache[m_id_str] = raw_json
                updated = True

        if updated:
            with open(self.raw_cache_path, "w", encoding="utf-8") as f:
                json.dump(raw_cache, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Zapisano surowe dane filmów do pliku: {self.raw_cache_path}")

        return {int(k): v for k, v in raw_cache.items()}


# ==============================================================================
# 2. PRZETWARZANIE I TRANSFORMACJA SUROWYCH CECH W PAMIĘCI
# ==============================================================================

class RawFeatureProcessor:
    """
    Przekształca surowe obiekty JSON z TMDB w ustrukturyzowane cechy z podziałem na:
    - cechy liczbowe (`num_`)
    - cechy zbiorowe (`set_` - natywne obiekty `set` w Pythonie)
    - cechy kategoryczne (`cat_`)
    - cechy tekstowe (`text_`)
    """

    IGNORED_RAW_KEYS = {
        "id", "imdb_id", "backdrop_path", "poster_path", "homepage",
        "adult", "video", "softcore", "status", "title", "original_title"
    }

    LOG_TRANSFORM_KEYS = {"budget", "revenue", "popularity", "vote_count"}

    @classmethod
    def transform_single_movie(cls, raw: Dict[str, Any], fallback_title: str = "") -> Dict[str, Any]:
        if any(k.startswith(("num_", "set_", "cat_", "text_")) for k in raw.keys()):
            converted = dict(raw)
            for k, v in list(converted.items()):
                if k.startswith("set_") and isinstance(v, list):
                    converted[k] = set(v)
            return converted

        features: Dict[str, Any] = {}

        for key, val in raw.items():
            if val is None or key in cls.IGNORED_RAW_KEYS:
                continue

            # Cechy liczbowe
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                val_float = float(val)
                if key in cls.LOG_TRANSFORM_KEYS:
                    features[f"num_{key}_log"] = math.log1p(val_float) if val_float > 0 else np.nan
                else:
                    features[f"num_{key}"] = val_float if val_float > 0 else np.nan

            # Rok premiery
            elif key == "release_date" and isinstance(val, str):
                if len(val) >= 4 and val[:4].isdigit():
                    features["num_release_year"] = float(val[:4])
                else:
                    features["num_release_year"] = np.nan

            # Ekstrakcja cech obiektowych zgodnie ze specyfikacją
            elif key == "genres" and isinstance(val, list):
                extracted = {
                    str(item["name"]).strip().lower()
                    for item in val
                    if isinstance(item, dict) and item.get("name")
                }
                if extracted:
                    features["set_genres"] = extracted

            elif key == "production_companies" and isinstance(val, list):
                extracted = set()
                for item in val:
                    if isinstance(item, dict):
                        if item.get("name"):
                            extracted.add(str(item["name"]).strip().lower())
                        if item.get("origin_country"):
                            extracted.add(str(item["origin_country"]).strip().lower())
                if extracted:
                    features["set_production_companies"] = extracted

            elif key == "production_countries" and isinstance(val, list):
                extracted = {
                    str(item["iso_3166_1"]).strip().lower()
                    for item in val
                    if isinstance(item, dict) and item.get("iso_3166_1")
                }
                if extracted:
                    features["set_production_countries"] = extracted

            elif key == "spoken_languages" and isinstance(val, list):
                extracted = {
                    str(item["iso_639_1"]).strip().lower()
                    for item in val
                    if isinstance(item, dict) and item.get("iso_639_1")
                }
                if extracted:
                    features["set_spoken_languages"] = extracted

        # Obsługa credits: cast (Top 5) i crew (wyłącznie Director i jego name)
        credits_data = raw.get("credits")
        if isinstance(credits_data, dict):
            cast_raw = credits_data.get("cast") or []
            cast_set = {
                str(c["name"]).strip().lower()
                for c in cast_raw[:5]
                if isinstance(c, dict) and c.get("name")
            }
            if cast_set:
                features["set_cast"] = cast_set

            # Filtrowanie wyłącznie reżyserów (job == "Director")
            crew_raw = credits_data.get("crew") or []
            directors_set = {
                str(m["name"]).strip().lower()
                for m in crew_raw
                if isinstance(m, dict) and m.get("name") and m.get("job") == "Director"
            }
            if directors_set:
                features["set_director"] = directors_set

        # Słowa kluczowe: keywords (wyłącznie pole name)
        kw_data = raw.get("keywords")
        if isinstance(kw_data, dict):
            kw_list = kw_data.get("keywords") or []
            kw_set = {
                str(k["name"]).strip().lower()
                for k in kw_list
                if isinstance(k, dict) and k.get("name")
            }
            if kw_set:
                features["set_keywords"] = kw_set

        # Cechy kategoryczne: język oraz kolekcja (wyłącznie pole name kolekcji)
        orig_lang = raw.get("original_language")
        if orig_lang and isinstance(orig_lang, str):
            features["cat_original_language"] = orig_lang.strip().lower()

        collection = raw.get("belongs_to_collection")
        if isinstance(collection, dict) and collection.get("name"):
            features["cat_belongs_to_collection"] = str(collection["name"]).strip().lower()

        # Opisy tekstowe
        tagline = (raw.get("tagline") or "").strip()
        overview = (raw.get("overview") or "").strip()
        combined_text = f"{tagline} {overview}".strip()
        if combined_text:
            features["text_overview"] = combined_text
        elif fallback_title:
            features["text_overview"] = fallback_title

        return features

    @classmethod
    def transform_all(
        cls,
        raw_movies: Dict[int, Dict[str, Any]],
        movies_df: pd.DataFrame,
    ) -> Dict[int, Dict[str, Any]]:
        """Przekształca surowe dane wszystkich filmów w słownik cech gotowych do obliczeń."""
        processed: Dict[int, Dict[str, Any]] = {}
        title_map = dict(zip(movies_df["movie_id"].astype(int), movies_df["title"].astype(str)))

        for m_id, raw_json in raw_movies.items():
            fallback = title_map.get(m_id, "")
            processed[m_id] = cls.transform_single_movie(raw_json, fallback_title=fallback)

        return processed


# ==============================================================================
# 3. ZNORMALIZOWANE MIARY PODOBIEŃSTWA DLA RÓŻNYCH TYPÓW KOMPONENTÓW [0, 1]
# ==============================================================================

def compute_numeric_similarity_matrix(values: List[Any]) -> np.ndarray:
    """
    Znormalizowana miara podobieństwa [0, 1] dla cechy liczbowej:
    s(x, y) = 1 - |x - y| / (max - min).
    """
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
    """
    Współczynnik podobieństwa Jaccarda [0, 1] dla cech reprezentowanych jako zbiory:
    s(A, B) = |A ∩ B| / |A ∪ B|.
    """
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
    """
    Miara podobieństwa z nasyceniem dla głównych aktorów (set_cast) oraz twórców (set_crew):
    1 wspólna kluczowa osoba = 0.70, 2 lub więcej = 1.00.
    """
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
    """
    Znormalizowana miara podobieństwa [0, 1] dla cech kategorycznych (pojedynczych i wielowartościowych).
    """
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
    """Oblicza macierz podobieństwa cosinusowego [0, 1] na reprezentacji TF-IDF.

    Puste dokumenty lub dokumenty bez słów kluczowych mają zerowe podobieństwo
    z innymi, a na przekątnej zachowują 1.0.
    """
    if not texts:
        return np.empty((0, 0), dtype=float)

    # TfidfVectorizer automatycznie:
    # 1. Zamienia tekst na lowercase
    # 2. Tokenizuje (domyślny regex: r'(?u)\b\w\w+\b')
    # 3. Usuwa stop words (wbudowana lista 'english')
    # 4. Normalizuje wektory L2
    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Obsługa przypadku, gdy wszystkie słowa w całym zbiorze to stop words
        sim = np.zeros((len(texts), len(texts)), dtype=float)
        np.fill_diagonal(sim, 1.0)
        return sim

    # Obliczenie podobieństwa cosinusowego
    sim = cosine_similarity(tfidf_matrix)

    # Zabezpieczenie numeryczne i spójność przekątnej
    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, 0.0, 1.0)


def build_component_similarity_matrices(
    movie_ids: List[int],
    features_by_movie: Dict[int, Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """
    Buduje słownik macierzy podobieństwa (N x N) w skali [0, 1].
    Sortuje klucze według malejącej wariancji pozadiagonalnej (zamiast alfabetycznie),
    aby wyeliminować faworyzowanie prefiksu 'cat_' przy remisach.
    """
    all_feature_keys: Set[str] = set()
    for f_dict in features_by_movie.values():
        all_feature_keys.update(f_dict.keys())

    temp_matrices: List[Tuple[float, str, np.ndarray]] = []
    diag_mask = ~np.eye(len(movie_ids), dtype=bool)

    for feat_name in all_feature_keys:
        raw_values = [features_by_movie[m_id].get(feat_name) for m_id in movie_ids]

        if feat_name in ("set_cast"):
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
            # Jako kryterium porządkujące stosujemy zróżnicowanie informacyjne cechy
            temp_matrices.append((std_val, feat_name, mat))

    # Sortowanie od cech najbardziej zróżnicowanych informacyjnie
    temp_matrices.sort(key=lambda x: x[0], reverse=True)
    return {feat_name: mat for _, feat_name, mat in temp_matrices}


# ==============================================================================
# 4. INDYWIDUALNY KLASYFIKATOR kNN Z SELEKCJĄ CECH (WRAPPER LOOCV)
# ==============================================================================

class UserKNNClassifier:
    """
    Osobny klasyfikator kNN dla pojedynczego użytkownika.
    Ocena ustalana jest wyłącznie poprzez głosowanie odległościowe (Distance-Weighted Voting),
    w którym każdy z k najbliższych sąsiadów oddaje głos o wadze proporcjonalnej
    do bliskości (odwrotności odległości) na swoją klasę oceny {0, 1, 2, 3, 4, 5}.
    """

    SPARSE_BONUS_FEATURES = {"cat_belongs_to_collection"}

    def __init__(
        self,
        user_id: int,
        k_candidates: Tuple[int, ...] = (2, 3, 4, 5, 7, 9, 11, 15, 17, 21),
        max_selected_features: int = 8,
        eps: float = 1e-4,
    ):
        self.user_id = user_id
        self.k_candidates = k_candidates
        self.max_selected_features = max_selected_features
        self.eps = eps

        self.train_indices: np.ndarray = np.array([], dtype=int)
        self.train_ratings: np.ndarray = np.array([], dtype=int)
        self.selected_features: List[str] = []
        self.best_k: int = 7
        self.best_power: float = 1.0
        self.cv_loss: float = 0.0
        self.user_mode_rating: int = 3

    def _predict_from_neighbors(
        self,
        sim_matrix: np.ndarray,
        train_ratings: np.ndarray,
        k: int,
        power: float,
        fallback_rating: int,
    ) -> np.ndarray:
        """
        Klasyfikacja wyłącznie metodą głosowania odległościowego.
        Waga sąsiada: w = (sim)^power / (1 - sim + eps).
        """
        n_queries, n_train = sim_matrix.shape
        k_eff = min(k, n_train)

        # Wybór k najbliższych sąsiadów
        order = np.argsort(-sim_matrix, axis=1)[:, :k_eff]
        top_sims = np.take_along_axis(sim_matrix, order, axis=1)
        top_ratings = train_ratings[order]

        # Waga zależna od odległości: d = 1 - sim -> waga rośnie gwałtownie, gdy d dąży do 0
        distances = np.maximum(1.0 - top_sims, 0.0)
        weights = np.power(np.maximum(top_sims, 0.0), power) / (distances + self.eps)

        preds = np.empty(n_queries, dtype=int)

        for i in range(n_queries):
            w_row = weights[i]
            if np.sum(w_row) <= 1e-8:
                preds[i] = fallback_rating
                continue

            # Tablica sumy głosów na każdą z 6 klas ocen {0, 1, 2, 3, 4, 5}
            votes = np.zeros(6, dtype=float)
            for r_val, w_val in zip(top_ratings[i], w_row):
                votes[int(r_val)] += w_val

            # Wybór oceny z największą sumą głosów ważonych odległością
            preds[i] = int(np.argmax(votes))

        return preds

    def _evaluate_loocv(
        self,
        combined_sim_sub: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, int, float]:
        """
        Ocena jakości podzbioru cech w procedurze LOOCV wyłącznie za pomocą
        głosowania odległościowego. Dobiera optymalne parametry k oraz power.
        """
        S = combined_sim_sub.copy()
        np.fill_diagonal(S, -1.0)

        n = len(y)
        k_max = min(max(self.k_candidates), n - 1)
        order = np.argsort(-S, axis=1)[:, :k_max]
        sorted_sims = np.maximum(np.take_along_axis(S, order, axis=1), 0.0)
        sorted_y = y[order]

        sorted_dist = np.maximum(1.0 - sorted_sims, 0.0)

        best_loss = float("inf")
        best_k = self.k_candidates[0]
        best_p = 1.0

        for p in (1.0, 2.0):
            # Obliczenie wag odległościowych dla wszystkich sąsiadów
            w_all = np.power(sorted_sims, p) / (sorted_dist + self.eps)

            for k in self.k_candidates:
                if k >= n:
                    continue

                w_k = w_all[:, :k]
                y_k = sorted_y[:, :k]

                # Przeprowadzenie głosowania odległościowego dla wszystkich próbek LOOCV
                pred_vote = np.empty(n, dtype=int)
                for i in range(n):
                    if np.sum(w_k[i]) <= 1e-8:
                        pred_vote[i] = self.user_mode_rating
                        continue

                    votes = np.zeros(6, dtype=float)
                    for r_val, w_val in zip(y_k[i], w_k[i]):
                        votes[int(r_val)] += w_val
                    pred_vote[i] = int(np.argmax(votes))

                # Metryka błędu: kombinacja MAE i błędu klasyfikacji (0-1 loss)
                mae = np.mean(np.abs(pred_vote - y))
                zero_one = np.mean(pred_vote != y)
                loss = 0.7 * mae + 0.3 * zero_one

                if loss < best_loss:
                    best_loss = loss
                    best_k = k
                    best_p = p

        return best_loss, best_k, best_p

    def fit(
        self,
        train_movie_indices: np.ndarray,
        train_ratings: np.ndarray,
        sim_matrices: Dict[str, np.ndarray],
    ) -> "UserKNNClassifier":
        """
        Dopasowanie modelu użytkownika metodą Sequential Forward Selection (SFS)
        sterowaną wyłącznie błędem głosowania odległościowego w LOOCV.
        """
        self.train_indices = train_movie_indices
        self.train_ratings = train_ratings.astype(int)

        # Jako ocenę domyślną (fallback) przyjmujemy dominantę ocen użytkownika
        vals, counts = np.unique(self.train_ratings, return_counts=True)
        self.user_mode_rating = int(vals[np.argmax(counts)])

        n = len(self.train_indices)
        candidate_features = list(sim_matrices.keys())
        user_sub_matrices = {
            f: sim_matrices[f][np.ix_(self.train_indices, self.train_indices)]
            for f in candidate_features
        }

        feature_coverage: Dict[str, float] = {}
        for f, sub_mat in user_sub_matrices.items():
            off_diag = sub_mat.copy()
            np.fill_diagonal(off_diag, 0.0)
            active_rows = np.sum(np.max(off_diag, axis=1) > 1e-5)
            feature_coverage[f] = active_rows / max(1, n)

        selected: List[str] = []
        remaining: List[str] = list(candidate_features)
        current_sum_sim = np.zeros((n, n), dtype=float)

        best_selected = []
        best_overall_loss = float("inf")
        best_overall_k = 7
        best_overall_p = 1.0

        max_steps = min(self.max_selected_features, len(candidate_features))

        for step in range(max_steps):
            step_best_feat = None
            step_best_loss = float("inf")
            step_best_k = 7
            step_best_p = 1.0

            for feat in remaining:
                if step == 0 and (
                    feat in self.SPARSE_BONUS_FEATURES or feature_coverage[feat] < 0.25
                ):
                    continue

                cand_sim = (current_sum_sim + user_sub_matrices[feat]) / (len(selected) + 1)
                loss, k_opt, p_opt = self._evaluate_loocv(cand_sim, self.train_ratings)

                if loss < step_best_loss:
                    step_best_loss = loss
                    step_best_feat = feat
                    step_best_k = k_opt
                    step_best_p = p_opt

            if step_best_feat is not None and (
                step_best_loss <= best_overall_loss + 1e-6 or len(selected) < 2
            ):
                selected.append(step_best_feat)
                remaining.remove(step_best_feat)
                current_sum_sim += user_sub_matrices[step_best_feat]

                if step_best_loss < best_overall_loss:
                    best_overall_loss = step_best_loss
                    best_overall_k = step_best_k
                    best_overall_p = step_best_p
            else:
                break

        self.selected_features = selected
        self.best_k = best_overall_k
        self.best_power = best_overall_p
        self.cv_loss = best_overall_loss
        return self

    def predict(
        self,
        query_movie_indices: np.ndarray,
        sim_matrices: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Predykcja ocen nowych filmów wyłącznie przez głosowanie odległościowe."""
        n_queries = len(query_movie_indices)
        n_train = len(self.train_indices)

        combined_sim = np.zeros((n_queries, n_train), dtype=float)

        if self.selected_features:
            for feat in self.selected_features:
                combined_sim += sim_matrices[feat][np.ix_(query_movie_indices, self.train_indices)]
            combined_sim /= len(self.selected_features)

        return self._predict_from_neighbors(
            sim_matrix=combined_sim,
            train_ratings=self.train_ratings,
            k=self.best_k,
            power=self.best_power,
            fallback_rating=self.user_mode_rating,
        )


# ==============================================================================
# 5. GŁÓWNY POTOK WYKONAWCZY (GENEROWANIE submission.csv ORAZ code.zip)
# ==============================================================================

def create_code_zip(zip_filename: str, files_to_include: List[str]) -> None:
    """Tworzy archiwum code.zip zawierające kod źródłowy oraz zapisane surowe dane TMDB."""
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filepath in files_to_include:
            if os.path.exists(filepath):
                zf.write(filepath, arcname=os.path.basename(filepath))
    print(f"[INFO] Utworzono archiwum z kodem i surowymi zasobami: {zip_filename}")


def evaluate_validation_accuracy(
    train_df: pd.DataFrame,
    movie_id_to_idx: Dict[int, int],
    sim_matrices: Dict[str, np.ndarray],
    n_splits: int = 5,
    seed: int = 42,
) -> None:
    """
    Przeprowadza walidację krzyżową (K-Fold CV) dla każdego użytkownika osobno.
    Dla każdego foldu trenuje model (Filter + Wrapper SFS + dobór k) wyłącznie
    na części treningowej i mierzy Accuracy na odłożonej części testowej.
    """
    rng = np.random.default_rng(seed)
    all_true: List[int] = []
    all_pred: List[int] = []
    all_baseline_mode: List[int] = []

    t0 = time.time()
    print(f"\n[WALIDACJA] Uruchamianie {n_splits}-krotnej walidacji krzyżowej dla 358 użytkowników...")

    for user_id, user_group in train_df.groupby("user_id"):
        m_indices = np.array([movie_id_to_idx[int(m)] for m in user_group["movie_id"]], dtype=int)
        ratings = user_group["rating"].to_numpy(dtype=int)

        # Losowa permutacja 90 filmów użytkownika i podział na n_splits części
        perm = rng.permutation(len(ratings))
        folds = np.array_split(perm, n_splits)

        for fold_idx in range(n_splits):
            val_pos = folds[fold_idx]
            train_pos = np.concatenate([folds[i] for i in range(n_splits) if i != fold_idx])

            # Trening klasyfikatora tylko na części treningowej
            clf = UserKNNClassifier(user_id=int(user_id))
            clf.fit(m_indices[train_pos], ratings[train_pos], sim_matrices)

            # Predykcja na niewidzianej części walidacyjnej
            preds = clf.predict(m_indices[val_pos], sim_matrices)

            # Dominanta z części treningowej jako punkt odniesienia (baseline)
            vals, counts = np.unique(ratings[train_pos], return_counts=True)
            mode_val = int(vals[np.argmax(counts)])

            all_true.extend(ratings[val_pos].tolist())
            all_pred.extend(preds.tolist())
            all_baseline_mode.extend([mode_val] * len(val_pos))

    y_true = np.array(all_true, dtype=int)
    y_pred = np.array(all_pred, dtype=int)
    y_base = np.array(all_baseline_mode, dtype=int)

    acc_exact = np.mean(y_pred == y_true)
    acc_base = np.mean(y_base == y_true)
    acc_within_1 = np.mean(np.abs(y_pred - y_true) <= 1)
    mae = np.mean(np.abs(y_pred - y_true))

    print("=" * 65)
    print(f"WYNIKI WALIDACJI KRZYŻOWEJ ({n_splits}-Fold CV, łącznie {len(y_true)} ocen):")
    print(f" -> Accuracy dokładne (y_pred == y_true): {acc_exact * 100:.2f}% (Baseline mody: {acc_base * 100:.2f}%)")
    print(f" -> Accuracy +/- 1 gwiazdka (|err| <= 1): {acc_within_1 * 100:.2f}%")
    print(f" -> Średni błąd bezwzględny (MAE):        {mae:.4f}")
    print(f" -> Czas walidacji:                       {time.time() - t0:.2f} s")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predykcja ocen filmów metodą kNN (zapis surowych danych TMDB + przetwarzanie w pamięci)."
    )
    parser.add_argument("--movie-file", default="movie.csv", help="Ścieżka do pliku movie.csv")
    parser.add_argument("--train-file", default="train.csv", help="Ścieżka do pliku train.csv")
    parser.add_argument("--task-file", default="task.csv", help="Ścieżka do pliku task.csv")
    parser.add_argument("--submission-file", default="submission.csv", help="Wyjściowy plik submission.csv")
    parser.add_argument("--cache-file", default="tmdb_raw.json", help="Plik JSON z surowymi odpowiedziami TMDB")
    parser.add_argument("--zip-file", default="code.zip", help="Wyjściowe archiwum code.zip")
    parser.add_argument("--api-key", default=None, help="Klucz API v3 do bazy TMDB")
    parser.add_argument("--bearer-token", default=None, help="Token Bearer API v4 do bazy TMDB")
    args = parser.parse_args()

    movies_df = pd.read_csv(
        args.movie_file, sep=";", header=None, names=["movie_id", "tmdb_id", "title"]
    )
    train_df = pd.read_csv(
        args.train_file, sep=";", header=None, names=["id", "user_id", "movie_id", "rating"]
    )
    task_df = pd.read_csv(
        args.task_file, sep=";", header=None, names=["id", "user_id", "movie_id", "rating"],
        na_values=["NULL"]
    )

    print(
        f"[INFO] Wczytano: {len(movies_df)} filmów, {len(train_df)} ocen treningowych "
        f"({train_df['user_id'].nunique()} użytkowników), {len(task_df)} wierszy w task.csv."
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
    movie_id_to_idx = {m_id: i for i, m_id in enumerate(sorted_movie_ids)}

    sim_matrices = build_component_similarity_matrices(sorted_movie_ids, features_by_movie)
    print(f"[INFO] Wygenerowano macierze podobieństw dla {len(sim_matrices)} cech: {list(sim_matrices.keys())}")

    evaluate_validation_accuracy(train_df, movie_id_to_idx, sim_matrices, n_splits=5)

    t_start = time.time()
    train_grouped = train_df.groupby("user_id")
    task_grouped = task_df.groupby("user_id")

    predictions_map: Dict[int, int] = {}
    cv_losses: List[float] = []
    chosen_k_counter: Counter = Counter()
    chosen_feat_counter: Counter = Counter()

    for user_id, user_train in train_grouped:
        train_m_idx = np.array([movie_id_to_idx[int(m)] for m in user_train["movie_id"]], dtype=int)
        train_y = user_train["rating"].to_numpy(dtype=float)

        clf = UserKNNClassifier(user_id=int(user_id))
        clf.fit(train_m_idx, train_y, sim_matrices)

        cv_losses.append(clf.cv_loss)
        chosen_k_counter[clf.best_k] += 1
        for f in clf.selected_features:
            chosen_feat_counter[f] += 1

        if user_id in task_grouped.groups:
            user_task = task_grouped.get_group(user_id)
            task_m_idx = np.array([movie_id_to_idx[int(m)] for m in user_task["movie_id"]], dtype=int)
            user_preds = clf.predict(task_m_idx, sim_matrices)

            for row_id, pred_val in zip(user_task["id"], user_preds):
                predictions_map[int(row_id)] = int(pred_val)

    elapsed = time.time() - t_start
    print(f"[INFO] Wytrenowano 358 indywidualnych klasyfikatorów kNN w {elapsed:.2f} s.")
    print(f"[INFO] Średni błąd LOOCV (0.7*MAE + 0.3*Err01): {np.mean(cv_losses):.4f}")
    print(f"[INFO] Rozkład wybranych wartości k: {dict(chosen_k_counter.most_common())}")
    print(f"[INFO] Najczęściej wybierane cechy: {chosen_feat_counter.most_common(10)}")

    submission_df = task_df.copy()
    submission_df["rating"] = [
        predictions_map.get(int(r_id), 3) for r_id in submission_df["id"]
    ]
    submission_df["rating"] = submission_df["rating"].astype(int)
    submission_df.to_csv(args.submission_file, sep=";", header=False, index=False)
    print(f"[INFO] Zapisano wynikowy plik predykcji: {args.submission_file}")

    script_path = os.path.abspath(__file__) if "__file__" in globals() else "solution.py"
    create_code_zip(
        args.zip_file,
        files_to_include=[
            script_path,
            args.cache_file,
            args.movie_file,
        ],
    )


if __name__ == "__main__":
    main()