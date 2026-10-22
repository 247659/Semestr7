#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rozwiązanie zadania predykcji ocen filmów metodą kNN (osobny klasyfikator dla każdego użytkownika).

Architektura dwuetapowa dla cech z TMDB:
1. Zapis wszystkich cech w postaci całkowicie surowej (1:1 odpowiedź JSON z TMDB API) do pliku cache.
2. Dopiero po wczytaniu surowych danych wykonywane są operacje czyszczenia, transformacji
   (logarytmowanie, obsługa braków danych, ekstrakcja zbiorów z credits/keywords, łączenie tekstów),
   obliczanie znormalizowanych miar podobieństwa [0, 1] oraz selekcja cech (Filter + Wrapper LOOCV).

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
            "append_to_response": "credits,keywords,genre",
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

    # Pola techniczne, identyfikatory i flagi o zerowej wariancji, które odrzucamy
    IGNORED_RAW_KEYS = {
        "id", "imdb_id", "backdrop_path", "poster_path", "homepage",
        "adult", "video", "softcore", "status", "title", "original_title"
    }

    # Cechy liczbowe o silnie skośnym rozkładzie wymagające transformacji log(1 + x)
    LOG_TRANSFORM_KEYS = {"budget", "revenue", "popularity", "vote_count"}

    @classmethod
    def transform_single_movie(cls, raw: Dict[str, Any], fallback_title: str = "") -> Dict[str, Any]:
        """
        Wykonuje docelowe operacje inżynierii cech na surowym słowniku JSON jednego filmu.
        """
        # Obsługa kompatybilności wstecznej, jeśli w cache zapisano już przetworzone klucze
        if any(k.startswith(("num_", "set_", "cat_", "text_")) for k in raw.keys()):
            converted = dict(raw)
            for k, v in list(converted.items()):
                if k.startswith("set_") and isinstance(v, list):
                    converted[k] = set(v)
            return converted

        features: Dict[str, Any] = {}

        # 1. Automatyczny przegląd głównych pól surowego JSON-a
        for key, val in raw.items():
            if val is None or key in cls.IGNORED_RAW_KEYS:
                continue

            # A) Wartości liczbowe (int / float)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                val_float = float(val)
                if key in cls.LOG_TRANSFORM_KEYS:
                    # Budżet lub przychód równy 0 w TMDB oznacza brak danych (NaN)
                    features[f"num_{key}_log"] = math.log1p(val_float) if val_float > 0 else np.nan
                else:
                    features[f"num_{key}"] = val_float if val_float > 0 else np.nan

            # B) Data premiery -> wyodrębnienie roku jako cechy liczbowej
            elif key == "release_date" and isinstance(val, str):
                if len(val) >= 4 and val[:4].isdigit():
                    features["num_release_year"] = float(val[:4])
                else:
                    features["num_release_year"] = np.nan

            # C) Listy obiektów/słowników (np. genres, production_companies, countries, languages)
            # Niech z genres wyjmuje tylko name
            # z production companies tylko name i origin_country
            # z belongs to collection
            # z production countires iso_3166_1
            # z spoken language iso_639_1
            # z cast name, gender
            # z crew gender i name
            # z keywords name
            elif isinstance(val, list) and val and isinstance(val[0], dict):
                extracted_set: Set[str] = set()
                for item in val:
                    name = item.get("name") or item.get("english_name") or item.get("iso_3166_1")
                    if name:
                        extracted_set.add(str(name).strip().lower())
                if extracted_set:
                    features[f"set_{key}"] = extracted_set

            # D) Listy prostych wartości tekstowych (np. origin_country: ["US"])
            elif isinstance(val, list) and val and isinstance(val[0], str):
                extracted_set = {str(x).strip().lower() for x in val if x and str(x).strip()}
                if extracted_set:
                    features[f"set_{key}"] = extracted_set

        # 2. Przetwarzanie zagnieżdżonego obiektu `credits` (obsada i ekipa filmowa)
        credits_data = raw.get("credits")
        if isinstance(credits_data, dict):
            # Top 15 najważniejszych aktorów z obsady (według kolejności w czołówce)
            cast_raw = credits_data.get("cast") or []
            cast_set = {
                str(c["name"]).strip().lower()
                for c in cast_raw[:15]
                if isinstance(c, dict) and c.get("name")
            }
            if cast_set:
                features["set_cast"] = cast_set

            # Ekipa twórcza: reżyserzy, scenarzyści, kompozytorzy muzyki
            crew_raw = credits_data.get("crew") or []
            directors: Set[str] = set()
            writers: Set[str] = set()
            composers: Set[str] = set()

            for member in crew_raw:
                if not isinstance(member, dict) or not member.get("name"):
                    continue
                person_name = str(member["name"]).strip().lower()
                job = member.get("job") or ""
                dept = member.get("department") or ""

                if job == "Director":
                    directors.add(person_name)
                if dept == "Writing" or job in ("Screenplay", "Writer", "Story"):
                    writers.add(person_name)
                if job == "Original Music Composer":
                    composers.add(person_name)

            if directors:
                features["set_directors"] = directors
            if writers:
                features["set_writers"] = writers
            if composers:
                features["set_composers"] = composers

        # 3. Przetwarzanie zagnieżdżonego obiektu `keywords` (słowa kluczowe)
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

        # 4. Cechy kategoryczne (język oryginału oraz przynależność do kolekcji/serii)
        orig_lang = raw.get("original_language")
        if orig_lang and isinstance(orig_lang, str):
            features["cat_original_language"] = orig_lang.strip().lower()

        collection = raw.get("belongs_to_collection")
        if isinstance(collection, dict) and collection.get("id") is not None:
            features["cat_collection_id"] = str(collection["id"])

        # 5. Cechy tekstowe: połączenie tagline i overview w jeden opis pod TF-IDF
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
    Brakujące wartości (NaN) zastępowane są medianą z dostępnych obserwacji.
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

# a co jak mamy kilka kategorii, np. kilka gatunków dla filmu??
def compute_categorical_similarity_matrix(categories: List[str]) -> np.ndarray:
    """
    Miara podobieństwa [0, 1] dla cechy kategorycznej:
    1.0 w przypadku identycznych niepustych wartości, 0.0 w przeciwnym razie.
    """
    n = len(categories)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        cat_i = categories[i]
        if not cat_i:
            continue
        for j in range(i + 1, n):
            if categories[j] == cat_i:
                sim[i, j] = 1.0
                sim[j, i] = 1.0
    return sim


def compute_text_similarity_matrix(texts: List[str]) -> np.ndarray:
    """
    Podobieństwo cosinusowe [0, 1] na reprezentacji TF-IDF dla opisów tekstowych.
    """
    n = len(texts)
    stop_words = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "is", "are", "was", "were", "by", "at", "from", "as", "that", "this",
        "his", "her", "their", "its", "it", "be", "into", "who", "when", "after",
        "but", "has", "have", "had", "not", "they", "he", "she", "him", "them"
    }

    tokenized_docs: List[List[str]] = []
    df_counts: Counter = Counter()

    for txt in texts:
        tokens = [
            tok for tok in re.findall(r"[a-z0-9]{2,}", (txt or "").lower())
            if tok not in stop_words
        ]
        tokenized_docs.append(tokens)
        for unique_tok in set(tokens):
            df_counts[unique_tok] += 1

    vocab = {tok: idx for idx, tok in enumerate(df_counts.keys())}
    if not vocab:
        return np.eye(n, dtype=float)

    tfidf = np.zeros((n, len(vocab)), dtype=float)
    for i, tokens in enumerate(tokenized_docs):
        if not tokens:
            continue
        tf_counter = Counter(tokens)
        doc_len = len(tokens)
        for tok, count in tf_counter.items():
            j = vocab[tok]
            tf = count / doc_len
            idf = math.log((1.0 + n) / (1.0 + df_counts[tok])) + 1.0
            tfidf[i, j] = tf * idf

    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms_safe = np.where(norms > 0, norms, 1.0)
    normalized_tfidf = tfidf / norms_safe

    sim = normalized_tfidf @ normalized_tfidf.T
    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, 0.0, 1.0)


def build_component_similarity_matrices(
    movie_ids: List[int],
    features_by_movie: Dict[int, Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """
    Automatycznie dobiera funkcję podobieństwa na podstawie prefiksu cechy
    (num_, set_, cat_, text_) i wyznacza macierz podobieństwa (N x N) w skali [0, 1].
    """
    all_feature_keys: Set[str] = set()
    for f_dict in features_by_movie.values():
        all_feature_keys.update(f_dict.keys())

    sim_matrices: Dict[str, np.ndarray] = {}

    for feat_name in sorted(all_feature_keys):
        raw_values = [features_by_movie[m_id].get(feat_name) for m_id in movie_ids]

        if feat_name.startswith("num_"):
            mat = compute_numeric_similarity_matrix(raw_values)
        elif feat_name.startswith("set_"):
            safe_sets = [s if isinstance(s, set) else set() for s in raw_values]
            mat = compute_set_similarity_matrix(safe_sets)
        elif feat_name.startswith("cat_"):
            safe_cats = [str(c) if c is not None else "" for c in raw_values]
            mat = compute_categorical_similarity_matrix(safe_cats)
        elif feat_name.startswith("text_"):
            safe_texts = [str(t) if t is not None else "" for t in raw_values]
            mat = compute_text_similarity_matrix(safe_texts)
        else:
            continue

        # Zachowaj wyłącznie cechy posiadające niezerową wariancję poza przekątną
        off_diag = mat[~np.eye(len(movie_ids), dtype=bool)]
        if len(off_diag) > 0 and np.std(off_diag) > 1e-5:
            sim_matrices[feat_name] = mat

    return sim_matrices


# ==============================================================================
# 4. INDYWIDUALNY KLASYFIKATOR kNN Z SELEKCJĄ CECH (FILTER + WRAPPER LOOCV)
# ==============================================================================

class UserKNNClassifier:
    """
    Osobny klasyfikator kNN dla pojedynczego użytkownika.
    Dobiera optymalny zestaw cech (Filter + Wrapper SFS), ich wagi, hiperparametr k
    oraz metodę agregacji sąsiadów za pomocą walidacji krzyżowej Leave-One-Out (LOOCV).
    """

    def __init__(
        self,
        user_id: int,
        k_candidates: Tuple[int, ...] = (3, 5, 7, 9, 11, 15, 21, 27),
        max_selected_features: int = 8,
        filter_top_m: int = 15,
    ):
        self.user_id = user_id
        self.k_candidates = k_candidates
        self.max_selected_features = max_selected_features
        self.filter_top_m = filter_top_m

        self.train_indices: np.ndarray = np.array([], dtype=int)
        self.train_ratings: np.ndarray = np.array([], dtype=float)
        self.selected_features: List[str] = []
        self.feature_weights: Dict[str, float] = {}
        self.best_k: int = 7
        self.best_aggregation: str = "weighted_mean"
        self.best_power: float = 1.0
        self.cv_loss: float = 0.0
        self.user_mean_rating: float = 3.0

    @staticmethod
    def _predict_from_neighbors(
        sim_matrix: np.ndarray,
        train_ratings: np.ndarray,
        k: int,
        aggregation: str,
        power: float,
        fallback_rating: float,
    ) -> np.ndarray:
        """Wyznacza predykcję oceny {0, 1, 2, 3, 4, 5} na podstawie k najbliższych sąsiadów."""
        n_queries, n_train = sim_matrix.shape
        k_eff = min(k, n_train)

        order = np.argsort(-sim_matrix, axis=1)[:, :k_eff]
        top_sims = np.take_along_axis(sim_matrix, order, axis=1)
        top_ratings = train_ratings[order]

        weights = np.power(np.maximum(top_sims, 0.0), power)
        weight_sums = np.sum(weights, axis=1)

        preds = np.empty(n_queries, dtype=int)

        if aggregation == "weighted_mean":
            valid = weight_sums > 1e-8
            cont_pred = np.full(n_queries, fallback_rating, dtype=float)
            cont_pred[valid] = np.sum(weights[valid] * top_ratings[valid], axis=1) / weight_sums[valid]
            preds = np.clip(np.rint(cont_pred), 0, 5).astype(int)

        elif aggregation == "weighted_voting":
            for i in range(n_queries):
                if weight_sums[i] <= 1e-8:
                    preds[i] = int(np.clip(round(fallback_rating), 0, 5))
                    continue
                class_scores = np.zeros(6, dtype=float)
                for r_val, w_val in zip(top_ratings[i], weights[i]):
                    class_scores[int(r_val)] += w_val
                preds[i] = int(np.argmax(class_scores))

        return preds

    def _evaluate_loocv(
        self,
        combined_sim_sub: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, int, str, float]:
        """Walidacja Leave-One-Out (LOOCV) dla oceny jakości podzbioru cech i doboru k."""
        S = combined_sim_sub.copy()
        np.fill_diagonal(S, -1.0)

        n = len(y)
        k_max = min(max(self.k_candidates), n - 1)
        order = np.argsort(-S, axis=1)[:, :k_max]
        sorted_sims = np.maximum(np.take_along_axis(S, order, axis=1), 0.0)
        sorted_y = y[order]

        best_loss = float("inf")
        best_k = self.k_candidates[0]
        best_agg = "weighted_mean"
        best_p = 1.0

        for p in (1.0, 2.0):
            w_all = np.power(sorted_sims, p)
            for k in self.k_candidates:
                if k >= n:
                    continue
                w = w_all[:, :k]
                y_k = sorted_y[:, :k]
                w_sum = np.sum(w, axis=1)
                valid = w_sum > 1e-8

                cont = np.full(n, self.user_mean_rating, dtype=float)
                cont[valid] = np.sum(w[valid] * y_k[valid], axis=1) / w_sum[valid]
                pred_mean = np.clip(np.rint(cont), 0, 5)

                mae = np.mean(np.abs(pred_mean - y))
                zero_one = np.mean(pred_mean != y)
                loss = 0.7 * mae + 0.3 * zero_one

                if loss < best_loss:
                    best_loss = loss
                    best_k = k
                    best_agg = "weighted_mean"
                    best_p = p

        # Sprawdzenie ważonego głosowania dla najlepszego k oraz p
        w_best = np.power(sorted_sims[:, :best_k], best_p)
        y_best = sorted_y[:, :best_k]
        vote_scores = np.zeros((n, 6), dtype=float)
        for c in range(6):
            vote_scores[:, c] = np.sum(w_best * (y_best == c), axis=1)
        pred_vote = np.argmax(vote_scores, axis=1)
        no_weight = np.sum(w_best, axis=1) <= 1e-8
        pred_vote[no_weight] = int(np.clip(round(self.user_mean_rating), 0, 5))

        loss_vote = 0.7 * np.mean(np.abs(pred_vote - y)) + 0.3 * np.mean(pred_vote != y)
        if loss_vote < best_loss:
            best_loss = loss_vote
            best_agg = "weighted_voting"

        return best_loss, best_k, best_agg, best_p

    def _filter_features(
        self,
        sim_matrices: Dict[str, np.ndarray],
        idx: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """
        Selekcja typu Filter: ocenia korelację podobieństwa filmów względem danej cechy
        z bliskością ocen wystawionych przez użytkownika: 1 - |y_i - y_j| / 5.
        """
        n = len(y)
        triu_i, triu_j = np.triu_indices(n, k=1)
        rating_sim = 1.0 - (np.abs(y[triu_i] - y[triu_j]) / 5.0)
        rating_sim_centered = rating_sim - np.mean(rating_sim)
        denom_y = np.linalg.norm(rating_sim_centered)

        scores: List[Tuple[str, float]] = []
        for feat_name, full_mat in sim_matrices.items():
            sub_mat = full_mat[np.ix_(idx, idx)]
            feat_vec = sub_mat[triu_i, triu_j]
            feat_centered = feat_vec - np.mean(feat_vec)
            denom_x = np.linalg.norm(feat_centered)

            if denom_x > 1e-8 and denom_y > 1e-8:
                corr = float(np.dot(feat_centered, rating_sim_centered) / (denom_x * denom_y))
            else:
                corr = 0.0
            scores.append((feat_name, corr))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def fit(
        self,
        train_movie_indices: np.ndarray,
        train_ratings: np.ndarray,
        sim_matrices: Dict[str, np.ndarray],
    ) -> "UserKNNClassifier":
        """Trenuje indywidualny model użytkownika (Filter + Wrapper SFS + strojenie k)."""
        self.train_indices = train_movie_indices
        self.train_ratings = train_ratings.astype(float)
        self.user_mean_rating = float(np.mean(self.train_ratings))

        # Etap 1: Selekcja cech metodą Filter
        ranked_features = self._filter_features(
            sim_matrices, self.train_indices, self.train_ratings
        )
        candidate_features = [f for f, _ in ranked_features[: self.filter_top_m]]

        user_sub_matrices = {
            f: sim_matrices[f][np.ix_(self.train_indices, self.train_indices)]
            for f in candidate_features
        }
        filter_dict = dict(ranked_features)

        # Etap 2: Selekcja cech metodą Wrapper (Sequential Forward Selection)
        selected: List[str] = []
        remaining: List[str] = list(candidate_features)
        current_weighted_sum = np.zeros(
            (len(self.train_indices), len(self.train_indices)), dtype=float
        )
        current_total_weight = 0.0

        best_overall_loss = float("inf")
        best_overall_k = 7
        best_overall_agg = "weighted_mean"
        best_overall_p = 1.0

        for _ in range(min(self.max_selected_features, len(candidate_features))):
            step_best_feat = None
            step_best_loss = float("inf")
            step_best_k = 7
            step_best_agg = "weighted_mean"
            step_best_p = 1.0
            step_best_weight = 1.0

            for feat in remaining:
                w_f = max(0.5, 1.0 + 2.0 * max(0.0, filter_dict.get(feat, 0.0)))
                cand_sim = (current_weighted_sum + w_f * user_sub_matrices[feat]) / (
                    current_total_weight + w_f
                )
                loss, k_opt, agg_opt, p_opt = self._evaluate_loocv(
                    cand_sim, self.train_ratings
                )
                if loss < step_best_loss:
                    step_best_loss = loss
                    step_best_feat = feat
                    step_best_k = k_opt
                    step_best_agg = agg_opt
                    step_best_p = p_opt
                    step_best_weight = w_f

            if step_best_feat is not None and (
                step_best_loss <= best_overall_loss + 1e-6 or len(selected) < 2
            ):
                selected.append(step_best_feat)
                remaining.remove(step_best_feat)
                self.feature_weights[step_best_feat] = step_best_weight
                current_weighted_sum += step_best_weight * user_sub_matrices[step_best_feat]
                current_total_weight += step_best_weight

                if step_best_loss < best_overall_loss:
                    best_overall_loss = step_best_loss
                    best_overall_k = step_best_k
                    best_overall_agg = step_best_agg
                    best_overall_p = step_best_p
            else:
                break

        self.selected_features = selected
        self.best_k = best_overall_k
        self.best_aggregation = best_overall_agg
        self.best_power = best_overall_p
        self.cv_loss = best_overall_loss
        return self

    def predict(
        self,
        query_movie_indices: np.ndarray,
        sim_matrices: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Generuje predykcje ocen dla nowych filmów rozważanego użytkownika."""
        n_queries = len(query_movie_indices)
        n_train = len(self.train_indices)

        combined_sim = np.zeros((n_queries, n_train), dtype=float)
        total_w = 0.0

        for feat in self.selected_features:
            w = self.feature_weights.get(feat, 1.0)
            combined_sim += w * sim_matrices[feat][np.ix_(query_movie_indices, self.train_indices)]
            total_w += w

        if total_w > 0:
            combined_sim /= total_w

        return self._predict_from_neighbors(
            sim_matrix=combined_sim,
            train_ratings=self.train_ratings,
            k=self.best_k,
            aggregation=self.best_aggregation,
            power=self.best_power,
            fallback_rating=self.user_mean_rating,
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

    # 1. Wczytanie plików CSV
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

    # 2. Pobranie lub wczytanie SUROWYCH danych z TMDB API (zapis 1:1 w pliku JSON)
    raw_client = TMDBRawClient(
        api_key=args.api_key,
        bearer_token=args.bearer_token,
        raw_cache_path=args.cache_file,
    )
    raw_movies_data = raw_client.load_or_fetch_raw(movies_df)

    # 3. Wykonanie wymaganych operacji transformacji i ekstrakcji cech dopiero w pamięci
    print("[INFO] Przetwarzanie surowych danych TMDB i ekstrakcja cech...")
    features_by_movie = RawFeatureProcessor.transform_all(raw_movies_data, movies_df)

    # 4. Wyznaczenie znormalizowanych macierzy podobieństw [0, 1] dla każdego komponentu
    sorted_movie_ids = sorted(movies_df["movie_id"].astype(int).tolist())
    movie_id_to_idx = {m_id: i for i, m_id in enumerate(sorted_movie_ids)}

    sim_matrices = build_component_similarity_matrices(sorted_movie_ids, features_by_movie)
    print(f"[INFO] Wygenerowano macierze podobieństw dla {len(sim_matrices)} cech: {list(sim_matrices.keys())}")

    evaluate_validation_accuracy(train_df, movie_id_to_idx, sim_matrices, n_splits=5)

    # 5. Trening indywidualnych klasyfikatorów kNN i predykcja dla każdego użytkownika
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

    # 6. Zapis predykcji do pliku submission.csv (format identyczny z task.csv)
    submission_df = task_df.copy()
    submission_df["rating"] = [
        predictions_map.get(int(r_id), 3) for r_id in submission_df["id"]
    ]
    submission_df["rating"] = submission_df["rating"].astype(int)
    submission_df.to_csv(args.submission_file, sep=";", header=False, index=False)
    print(f"[INFO] Zapisano wynikowy plik predykcji: {args.submission_file}")

    # 7. Utworzenie archiwum code.zip z kodem oraz surowym plikiem cech
    script_path = os.path.abspath(__file__) if "__file__" in globals() else "solution.py"
    create_code_zip(
        args.zip_file,
        files_to_include=[
            script_path,
            args.cache_file,
            args.movie_file,
        ],
    )

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

if __name__ == "__main__":

    main()