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
import re
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

            cached = raw_cache.get(m_id_str)
            # Wpis zawierający tylko title/id nie jest pełnymi metadanymi TMDB.
            useful_keys = {
                "genres", "release_date", "original_language", "overview",
                "production_companies", "production_countries", "runtime",
                "credits", "keywords", "budget", "revenue", "vote_average",
            }
            cache_valid = (
                isinstance(cached, dict)
                and bool(useful_keys.intersection(cached.keys()))
                and cached.get("id", tmdb_id) == tmdb_id
            )
            if not cache_valid:
                if not (self.api_key or self.bearer_token):
                    raise RuntimeError(
                        f"Brak kompletnych danych TMDB w cache dla filmu {title!r} "
                        f"(movie_id={m_id_str}, tmdb_id={tmdb_id}) i brak klucza API. "
                        "Ustaw TMDB_API_KEY/TMDB_BEARER_TOKEN albo podaj --api-key/--bearer-token."
                    )
                print(f"[TMDB API] Pobieranie surowych danych {idx + 1}/{total}: {title} (TMDB ID: {tmdb_id})")
                raw_json = self._fetch_raw_movie(tmdb_id)
                if not isinstance(raw_json, dict) or not raw_json:
                    # Zostawiamy jawne ostrzeżenie; fallback tytułu umożliwi dalsze działanie,
                    # ale nie udajemy, że pełne metadane zostały pobrane.
                    print(f"[OSTRZEŻENIE] Brak metadanych TMDB dla filmu {title!r}; zostanie użyty tytuł.")
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
                companies = set()
                company_countries = set()
                for item in val:
                    if isinstance(item, dict):
                        if item.get("name"):
                            companies.add(str(item["name"]).strip().lower())
                        if item.get("origin_country"):
                            company_countries.add(str(item["origin_country"]).strip().lower())
                if companies:
                    features["set_production_companies"] = companies
                if company_countries:
                    features["set_company_origin_countries"] = company_countries

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
# 4. INDYWIDUALNY kNN + WRAPPER SFS (LOOCV BEZ PRZECIEKU)
# ==============================================================================

DEFAULT_K_CANDIDATES = (2, 3, 4, 5, 7, 9, 11, 15, 17)
POWER_CANDIDATES = (1.0, 2.0)


def calculate_loss(y_true: np.ndarray, y_pred: np.ndarray, loss_name: str = "mae") -> float:
    """Jedyna używana funkcja straty: MAE (średni błąd bezwzględny)."""
    if loss_name != "mae":
        raise ValueError("Ta wersja rozwiązania obsługuje wyłącznie loss='mae'.")
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return float("inf")
    return float(np.mean(np.abs(y_pred - y_true)))


def rating_mode_without_current_label(y: np.ndarray) -> np.ndarray:
    """LOOCV fallback: dominanta dla każdego wiersza liczona bez jego własnej etykiety."""
    y = np.asarray(y, dtype=int)
    total_counts = np.bincount(y, minlength=6)
    fallbacks = np.empty(len(y), dtype=int)
    for i, current_label in enumerate(y):
        counts = total_counts.copy()
        counts[current_label] -= 1
        # Gdy po odłożeniu obserwacji nie ma żadnej oceny, stosujemy neutralne 3.
        fallbacks[i] = int(np.argmax(counts)) if counts.sum() else 3
    return fallbacks


class UserKNNClassifier:
    """Osobny kNN dla użytkownika; cechy oraz k/p dobierane wrapperem SFS i LOOCV."""

    def __init__(
        self,
        user_id: int,
        k_candidates: Tuple[int, ...] = DEFAULT_K_CANDIDATES,
        max_selected_features: int = 8,
        eps: float = 1e-4,
        aggregation: str = "weighted_mean",
        loss_name: str = "mae",
    ):
        if aggregation != "weighted_mean":
            raise ValueError(f"Nieznana agregacja: {aggregation}")
        if loss_name != "mae":
            raise ValueError(f"Nieznana funkcja straty: {loss_name}")
        self.user_id = int(user_id)
        self.k_candidates = tuple(sorted({int(k) for k in k_candidates if int(k) >= 1})) or (1,)
        self.max_selected_features = max(0, int(max_selected_features))
        self.eps = float(eps)
        self.aggregation = aggregation
        self.loss_name = loss_name

        self.train_indices = np.array([], dtype=int)
        self.train_ratings = np.array([], dtype=int)
        self.selected_features: List[str] = []
        self.best_k = 1
        self.best_power = 1.0
        self.cv_loss = float("nan")
        self.user_mode_rating = 3

    def _aggregate_neighbors(
        self,
        neighbor_ratings: np.ndarray,
        weights: np.ndarray,
        fallback_ratings: np.ndarray,
    ) -> np.ndarray:
        """Agregacja weighted_mean: średnia ocen ważona podobieństwem sąsiadów."""
        neighbor_ratings = np.asarray(neighbor_ratings, dtype=float)
        weights = np.asarray(weights, dtype=float)
        result = np.asarray(fallback_ratings, dtype=int).copy()
        weight_sum = weights.sum(axis=1)
        valid = weight_sum > 1e-12
        if np.any(valid):
            means = np.sum(neighbor_ratings[valid] * weights[valid], axis=1) / weight_sum[valid]
            # Oceny wyjściowe są całkowite w zakresie 0..5; .5 zaokrąglamy w górę.
            result[valid] = np.clip(np.floor(means + 0.5), 0, 5).astype(int)
        return result

    def _predict_from_neighbors(
        self,
        sim_matrix: np.ndarray,
        train_ratings: np.ndarray,
        k: int,
        power: float,
        fallback_rating: int,
    ) -> np.ndarray:
        """Predykcja z k najbliższych sąsiadów; fallback jest tylko dla braku wag."""
        n_queries, n_train = sim_matrix.shape
        if n_queries == 0:
            return np.array([], dtype=int)
        if n_train == 0:
            return np.full(n_queries, int(fallback_rating), dtype=int)

        k_eff = min(max(1, int(k)), n_train)
        order = np.argsort(-sim_matrix, axis=1)[:, :k_eff]
        top_sims = np.clip(np.take_along_axis(sim_matrix, order, axis=1), 0.0, 1.0)
        top_ratings = train_ratings[order]
        distances = np.maximum(1.0 - top_sims, 0.0)
        weights = np.power(top_sims, power) / (distances + self.eps)
        fallback = np.full(n_queries, int(fallback_rating), dtype=int)
        return self._aggregate_neighbors(top_ratings, weights, fallback)

    def _evaluate_loocv(self, combined_sim_sub: np.ndarray, y: np.ndarray) -> Tuple[float, int, float]:
        """LOOCV dla konkretnego podzbioru cech; fallback nie używa odłożonej etykiety."""
        y = np.asarray(y, dtype=int)
        n = len(y)
        if n < 2:
            return float("inf"), 1, 1.0

        similarities = np.asarray(combined_sim_sub, dtype=float).copy()
        np.fill_diagonal(similarities, -1.0)
        valid_ks = [k for k in self.k_candidates if k < n]
        if not valid_ks:
            valid_ks = [1]
        k_max = min(max(valid_ks), n - 1)
        order = np.argsort(-similarities, axis=1)[:, :k_max]
        sorted_sims = np.clip(np.take_along_axis(similarities, order, axis=1), 0.0, 1.0)
        sorted_y = y[order]

        # Kluczowa poprawka: dla i-tej obserwacji fallback jest dominantą y bez y[i].
        fold_fallbacks = rating_mode_without_current_label(y)
        best_loss = float("inf")
        best_k = valid_ks[0]
        best_power = POWER_CANDIDATES[0]

        for power in POWER_CANDIDATES:
            weights_all = np.power(sorted_sims, power) / (
                np.maximum(1.0 - sorted_sims, 0.0) + self.eps
            )
            for k in valid_ks:
                if k > weights_all.shape[1]:
                    continue
                weights_k = weights_all[:, :k]
                ratings_k = sorted_y[:, :k]
                predictions = self._aggregate_neighbors(ratings_k, weights_k, fold_fallbacks)
                loss = calculate_loss(y, predictions, self.loss_name)
                if loss < best_loss - 1e-12:
                    best_loss, best_k, best_power = loss, k, power

        return best_loss, int(best_k), float(best_power)

    def fit(
        self,
        train_movie_indices: np.ndarray,
        train_ratings: np.ndarray,
        sim_matrices: Dict[str, np.ndarray],
    ) -> "UserKNNClassifier":
        """Dopasowanie modelu użytkownika z SFS; zachowuje najlepszy znaleziony subset."""
        self.train_indices = np.asarray(train_movie_indices, dtype=int)
        self.train_ratings = np.asarray(train_ratings, dtype=int)
        if len(self.train_indices) != len(self.train_ratings):
            raise ValueError("Liczba indeksów filmów nie zgadza się z liczbą ocen.")
        if np.any((self.train_ratings < 0) | (self.train_ratings > 5)):
            raise ValueError("Oceny treningowe muszą należeć do {0, 1, 2, 3, 4, 5}.")

        if len(self.train_ratings):
            vals, counts = np.unique(self.train_ratings, return_counts=True)
            self.user_mode_rating = int(vals[np.argmax(counts)])
        else:
            self.user_mode_rating = 3

        n = len(self.train_indices)
        if n < 2:
            self.selected_features = []
            self.best_k = 1
            self.best_power = 1.0
            self.cv_loss = float("nan")
            return self

        if self.max_selected_features == 0 or not sim_matrices:
            # Nawet bez cech oceniaj fallback w LOOCV, każdorazowo bez etykiety
            # obserwacji walidacyjnej; nie raportuj błędu treningowego jako CV loss.
            self.selected_features = []
            self.cv_loss, self.best_k, self.best_power = self._evaluate_loocv(
                np.zeros((n, n), dtype=float), self.train_ratings
            )
            return self

        candidate_features = sorted(sim_matrices.keys())
        user_sub_matrices = {
            feature: np.asarray(sim_matrices[feature])[np.ix_(self.train_indices, self.train_indices)]
            for feature in candidate_features
        }

        selected_path: List[str] = []
        remaining = list(candidate_features)
        current_sum_sim = np.zeros((n, n), dtype=float)

        # Pusty subset to baseline LOOCV (predykcja fallbackiem liczonym bez danej etykiety).
        best_loss, best_k, best_power = self._evaluate_loocv(current_sum_sim, self.train_ratings)
        best_selected: List[str] = []

        # Rozważamy wszystkie rozmiary subsetu do limitu. selected_path to ścieżka zachłanna;
        # best_selected przechowuje osobno dokładnie tę wersję, dla której loss był najmniejszy.
        max_steps = min(self.max_selected_features, len(candidate_features))
        for _step in range(max_steps):
            step_best_feature = None
            step_best_loss = float("inf")
            step_best_k = 1
            step_best_power = 1.0

            for feature in remaining:
                candidate_sim = (current_sum_sim + user_sub_matrices[feature]) / (len(selected_path) + 1)
                loss, k_opt, power_opt = self._evaluate_loocv(candidate_sim, self.train_ratings)
                if loss < step_best_loss - 1e-12:
                    step_best_feature = feature
                    step_best_loss = loss
                    step_best_k = k_opt
                    step_best_power = power_opt

            if step_best_feature is None:
                break

            # Kontynuujemy ścieżkę SFS, by móc sprawdzić większe subsety; wynik końcowy
            # jest jednak zawsze najlepszym subsetem, niekoniecznie ostatnim.
            selected_path.append(step_best_feature)
            remaining.remove(step_best_feature)
            current_sum_sim += user_sub_matrices[step_best_feature]

            if step_best_loss < best_loss - 1e-12:
                best_loss = step_best_loss
                best_k = step_best_k
                best_power = step_best_power
                best_selected = selected_path.copy()

        self.selected_features = best_selected
        self.best_k = int(best_k)
        self.best_power = float(best_power)
        self.cv_loss = float(best_loss)
        return self

    def predict(
        self,
        query_movie_indices: np.ndarray,
        sim_matrices: Dict[str, np.ndarray],
    ) -> np.ndarray:
        query_movie_indices = np.asarray(query_movie_indices, dtype=int)
        n_queries = len(query_movie_indices)
        n_train = len(self.train_indices)
        if n_queries == 0:
            return np.array([], dtype=int)
        if n_train == 0:
            return np.full(n_queries, 3, dtype=int)

        combined_sim = np.zeros((n_queries, n_train), dtype=float)
        if self.selected_features:
            for feature in self.selected_features:
                combined_sim += sim_matrices[feature][np.ix_(query_movie_indices, self.train_indices)]
            combined_sim /= len(self.selected_features)

        return self._predict_from_neighbors(
            sim_matrix=combined_sim,
            train_ratings=self.train_ratings,
            k=self.best_k,
            power=self.best_power,
            fallback_rating=self.user_mode_rating,
        )


def create_code_zip(zip_filename: str, files_to_include: List[str]) -> None:
    """Tworzy archiwum z kodem i zasobami potrzebnymi do odtworzenia wyniku."""
    readme = """# Movie-rating kNN solution\n\n\
## Uruchomienie\n\
1. Zainstaluj zależności z requirements.txt.\n\
2. Ustaw TMDB_API_KEY lub TMDB_BEARER_TOKEN, jeśli cache TMDB nie zawiera wszystkich filmów.\n\
3. Uruchom: python solution.py --movie-file movie.csv --train-file train.csv --task-file task.csv\n\
Skrypt używa wyłącznie agregacji weighted_mean oraz funkcji straty MAE.\\
Dobiera k, power i podzbiór cech za pomocą wewnętrznego LOOCV i dopasowuje osobny model kNN dla każdego użytkownika.\\
Pliki wejściowe w tym archiwum są zasobami eksperymentu. Nie umieszczaj w archiwum kluczy API.\n"""
    requirements = "numpy\npandas\nrequests\nscikit-learn\n"
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        written = set()
        for filepath in files_to_include:
            if filepath and os.path.isfile(filepath):
                arcname = os.path.basename(filepath)
                if arcname not in written and arcname != os.path.basename(zip_filename):
                    zf.write(filepath, arcname=arcname)
                    written.add(arcname)
        zf.writestr("README.md", readme)
        zf.writestr("requirements.txt", requirements)
    print(f"[INFO] Utworzono archiwum z kodem i zasobami: {zip_filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predykcja ocen filmów kNN z agregacją weighted_mean i funkcją straty MAE."
    )
    parser.add_argument("--movie-file", default="movie.csv", help="Ścieżka do movie.csv (separator ;, bez nagłówka)")
    parser.add_argument("--train-file", default="train.csv", help="Ścieżka do train.csv (separator ;, bez nagłówka)")
    parser.add_argument("--task-file", default="task.csv", help="Ścieżka do task.csv (separator ;, bez nagłówka)")
    parser.add_argument("--submission-file", default="submission.csv", help="Wyjściowy plik submission.csv")
    parser.add_argument("--cache-file", default="tmdb_raw.json", help="Cache surowych danych TMDB")
    parser.add_argument("--zip-file", default="code.zip", help="Wyjściowe archiwum code.zip")
    parser.add_argument("--api-key", default=None, help="Klucz API v3 TMDB")
    parser.add_argument("--bearer-token", default=None, help="Token Bearer API v4 TMDB")
    parser.add_argument("--max-selected-features", type=int, default=8, help="Maksymalna liczba cech w SFS")
    args = parser.parse_args()
    if args.max_selected_features < 0:
        parser.error("--max-selected-features musi być >= 0")

    movies_df = pd.read_csv(
        args.movie_file, sep=";", header=None, names=["movie_id", "tmdb_id", "title"],
        dtype={"movie_id": "Int64", "tmdb_id": "Int64"},
    )
    train_df = pd.read_csv(
        args.train_file, sep=";", header=None, names=["id", "user_id", "movie_id", "rating"],
        na_values=["NULL", "null", ""],
    )
    task_df = pd.read_csv(
        args.task_file, sep=";", header=None, names=["id", "user_id", "movie_id", "rating"],
        na_values=["NULL", "null", ""],
    )

    required_movie_cols = ["movie_id", "tmdb_id", "title"]
    if movies_df[required_movie_cols].isna().any().any():
        raise ValueError("movie.csv zawiera brakujące movie_id, tmdb_id lub title.")
    if movies_df["movie_id"].duplicated().any():
        duplicates = movies_df.loc[movies_df["movie_id"].duplicated(), "movie_id"].tolist()[:10]
        raise ValueError(f"movie.csv zawiera powtórzone movie_id, np. {duplicates}")
    if train_df[["id", "user_id", "movie_id", "rating"]].isna().any().any():
        raise ValueError("train.csv zawiera brakujące wartości.")
    if task_df[["id", "user_id", "movie_id"]].isna().any().any():
        raise ValueError("task.csv zawiera brakujące id, user_id lub movie_id.")
    if task_df["id"].duplicated().any():
        raise ValueError("id wiersza w task.csv musi być unikalne.")

    movies_df["movie_id"] = movies_df["movie_id"].astype(int)
    movies_df["tmdb_id"] = movies_df["tmdb_id"].astype(int)
    for frame in (train_df, task_df):
        for col in ("id", "user_id", "movie_id"):
            frame[col] = frame[col].astype(int)

    train_df["rating"] = pd.to_numeric(train_df["rating"], errors="raise")
    if not np.all(np.isfinite(train_df["rating"].to_numpy(dtype=float))):
        raise ValueError("train.csv zawiera niepoprawne oceny.")
    if np.any(train_df["rating"].to_numpy(dtype=float) != np.floor(train_df["rating"].to_numpy(dtype=float))):
        raise ValueError("Oceny w train.csv muszą być liczbami całkowitymi.")
    train_df["rating"] = train_df["rating"].astype(int)
    if not train_df["rating"].between(0, 5).all():
        raise ValueError("Oceny w train.csv muszą należeć do {0, 1, 2, 3, 4, 5}.")

    known_task_ratings = task_df["rating"].notna()
    if known_task_ratings.any():
        known = pd.to_numeric(task_df.loc[known_task_ratings, "rating"], errors="raise")
        if not np.all(np.isfinite(known.to_numpy(dtype=float))) or not np.all(known == np.floor(known)):
            raise ValueError("Niepuste oceny w task.csv muszą być skończonymi liczbami całkowitymi.")
        if not known.between(0, 5).all():
            raise ValueError("Niepuste oceny w task.csv muszą należeć do {0, 1, 2, 3, 4, 5}.")
        task_df.loc[known_task_ratings, "rating"] = known.astype(int)

    known_movie_ids = set(movies_df["movie_id"].tolist())
    referenced_ids = set(train_df["movie_id"].tolist()) | set(task_df["movie_id"].tolist())
    missing_movies = sorted(referenced_ids - known_movie_ids)
    if missing_movies:
        raise ValueError(f"movie_id z train/task nie występują w movie.csv, np. {missing_movies[:10]}")

    missing_task_df = task_df.loc[task_df["rating"].isna()].copy()
    print(
        f"[INFO] Wczytano {len(movies_df)} filmów, {len(train_df)} ocen treningowych "
        f"({train_df['user_id'].nunique()} użytkowników), {len(task_df)} wierszy task, "
        f"w tym {len(missing_task_df)} ocen do uzupełnienia."
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
    print(
        f"[INFO] Używam stałej konfiguracji aggregation={best_aggregation}, loss={best_loss}."
    )

    # Fallback dla użytkownika bez historii: globalna dominanta jest jedyną informacją dostępną.
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
                max_selected_features=args.max_selected_features,
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
    submission_df.to_csv(args.submission_file, sep=";", header=False, index=False)

    print(f"[INFO] Zapisano submission: {args.submission_file}")
    print(f"[INFO] Finalny trening/predykcja: {time.time() - start:.2f} s")
    print(f"[INFO] Rozkład wybranego k: {dict(chosen_k_counter.most_common())}")
    print(f"[INFO] Najczęściej wybierane cechy: {chosen_feat_counter.most_common(10)}")
    if cv_losses:
        print(f"[INFO] Średni najlepszy wewnętrzny LOOCV loss: {float(np.mean(cv_losses)):.4f}")
    if cold_start_users:
        print(
            f"[OSTRZEŻENIE] {len(cold_start_users)} użytkowników nie ma ocen treningowych; "
            f"użyto globalnej dominanty {global_mode}: {sorted(cold_start_users)[:10]}"
        )

    script_path = os.path.abspath(__file__) if "__file__" in globals() else os.path.abspath("solution.py")
    create_code_zip(
        args.zip_file,
        files_to_include=[
            script_path,
            args.cache_file,
            args.movie_file,
            args.train_file,
            args.task_file,
        ],
    )


if __name__ == "__main__":
    main()
