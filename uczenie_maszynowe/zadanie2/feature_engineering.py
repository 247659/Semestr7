"""
Moduł inżynierii cech (Feature Engineering) przekształcający surowe metadane
filmów z TMDB (tmdb_raw.json) oraz movie.csv w macierz cech numerycznych i binarnych.

Zgodnie z zatwierdzonym projektem (Etap 1):
- Grupa A: Podstawowe cechy numeryczne (budget, revenue, runtime, popularity,
  vote_average, vote_count, release_year, movie_age, roi_ratio, is_budget_known, is_revenue_known).
- Grupa B: Gatunki filmowe (multi-hot encoding wszystkich gatunków + genres_count).
- Grupa C: Informacje o produkcji (is_english, najczęstsze języki i kraje z progiem >= 5,
  liczba firm/krajów/języków, najczęstsze wytwórnie z progiem >= 5).
- Grupa D: Obsada i ekipa (cast_count, crew_count, top5_cast_max_popularity,
  top5_cast_mean_popularity, director_popularity, female_cast_ratio + multi-hot dla
  najczęstszych reżyserów >= 3, aktorów >= 4, scenarzystów >= 3).
- Grupa E: Statystyki opisu (overview_len_words, has_tagline) bez pełnego TF-IDF.
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Any


class MovieFeatureEngineer:
    def __init__(
        self,
        reference_year: int = 2026,
        min_lang_freq: int = 5,
        min_country_freq: int = 5,
        min_company_freq: int = 5,
        min_director_freq: int = 3,
        min_actor_freq: int = 4,
        min_writer_freq: int = 3,
        min_collection_freq: int = 2
    ):
        self.reference_year = reference_year
        self.min_lang_freq = min_lang_freq
        self.min_country_freq = min_country_freq
        self.min_company_freq = min_company_freq
        self.min_director_freq = min_director_freq
        self.min_actor_freq = min_actor_freq
        self.min_writer_freq = min_writer_freq
        self.min_collection_freq = min_collection_freq

        self.feature_names_: List[str] = []
        self.is_binary_feature_: np.ndarray = np.array([], dtype=bool)
        self.medians_: Dict[str, float] = {}

    @staticmethod
    def _extract_movie_attributes(rec: Dict[str, Any]) -> Dict[str, Any]:
        """Pomocnicza ekstrakcja pól z pojedynczego słownika JSON filmu."""
        if not rec:
            rec = {}

        budget = float(rec.get("budget") or 0.0)
        revenue = float(rec.get("revenue") or 0.0)
        runtime = float(rec.get("runtime") or 0.0)
        popularity = float(rec.get("popularity") or 0.0)
        vote_average = float(rec.get("vote_average") or 0.0)
        vote_count = float(rec.get("vote_count") or 0.0)

        rel_date = str(rec.get("release_date") or "")
        release_year = np.nan
        if len(rel_date) >= 4 and rel_date[:4].isdigit():
            release_year = float(int(rel_date[:4]))

        # Gatunki
        genres_raw = rec.get("genres") or []
        genres = [
            str(g.get("name")).strip()
            for g in genres_raw
            if isinstance(g, dict) and g.get("name")
        ]

        # Kolekcje
        collection_raw = rec.get("belongs_to_collection") or {}
        if isinstance(collection_raw, dict):
            collection_name = str(collection_raw.get("name") or "").strip()
        else:
            collection_name = ""

        # Język oryginalny i języki mówione
        orig_lang = str(rec.get("original_language") or "unknown").strip().lower()
        spoken_raw = rec.get("spoken_languages") or []
        spoken_raw = [
            str(s.get("iso_639_1")).strip()
            for s in spoken_raw
            if isinstance(s, dict) and s.get("iso_639_1")
        ]

        # Kraje produkcji
        countries_raw = rec.get("production_countries") or []
        countries = [
            str(c.get("iso_3166_1")).strip()
            for c in countries_raw
            if isinstance(c, dict) and c.get("iso_3166_1")
        ]
        if not countries and isinstance(rec.get("origin_country"), list):
            countries = [str(c).strip() for c in rec.get("origin_country") if c]

        # Firmy produkcyjne
        companies_raw = rec.get("production_companies") or []
        companies = [
            str(c.get("name")).strip()
            for c in companies_raw
            if isinstance(c, dict) and c.get("name")
        ]

        # Obsada i ekipa
        credits = rec.get("credits") or {}
        cast_raw = credits.get("cast") if isinstance(credits, dict) else []
        crew_raw = credits.get("crew") if isinstance(credits, dict) else []
        if not isinstance(cast_raw, list):
            cast_raw = []
        if not isinstance(crew_raw, list):
            crew_raw = []

        # Sortowanie obsady wg pola order
        cast_sorted = sorted(
            [c for c in cast_raw if isinstance(c, dict)],
            key=lambda x: x.get("order", 999),
        )
        top10_cast = cast_sorted[:10]
        top5_cast = cast_sorted[:5]

        top10_actor_names = [
            str(c.get("name")).strip() for c in top10_cast if c.get("name")
        ]

        # Odsetek kobiet (gender == 1 w TMDB) w top 10 obsady
        known_gender_cast = [
            c for c in top10_cast if c.get("gender") in (1, 2)
        ]
        if known_gender_cast:
            female_ratio = sum(1 for c in known_gender_cast if c.get("gender") == 1) / len(
                known_gender_cast
            )
        else:
            female_ratio = 0.0

        # Reżyserzy i scenarzyści
        directors = []
        writers = []
        for member in crew_raw:
            if not isinstance(member, dict):
                continue
            job = str(member.get("job") or "").strip().lower()
            dept = str(member.get("department") or "").strip().lower()
            name = str(member.get("name") or "").strip()
            if not name:
                continue
            if job == "director":
                directors.append(name)
            if dept == "writing" or job in ("screenplay", "writer", "story", "author"):
                writers.append(name)

        # Tekst: overview i tagline
        overview = str(rec.get("overview") or "").strip()
        tagline = str(rec.get("tagline") or "").strip()

        return {
            "budget": budget,
            "revenue": revenue,
            "runtime": runtime,
            "popularity": popularity,
            "vote_average": vote_average,
            "vote_count": vote_count,
            "release_year": release_year,
            "genres": list(dict.fromkeys(genres)),
            "collection_name": collection_name,
            "belongs_to_collection": 1.0 if collection_name else 0.0,
            "orig_lang": orig_lang,
            "spoken_languages_count": float(len(spoken_raw)),
            "countries": list(dict.fromkeys(countries)),
            "companies": list(dict.fromkeys(companies)),
            "cast_count": float(len(cast_raw)),
            "crew_count": float(len(crew_raw)),
            "female_cast_ratio": float(female_ratio),
            "top10_actors": list(dict.fromkeys(top10_actor_names)),
            "directors": list(dict.fromkeys(directors)),
            "writers": list(dict.fromkeys(writers)),
        }

    def fit_transform(
        self, movies_df: pd.DataFrame, tmdb_records: Dict[int, Dict[str, Any]]
    ) -> Tuple[np.ndarray, Dict[int, int], List[str], np.ndarray]:
        """
        Buduje macierz cech X_movies o wymiarach (M, P), gdzie M = liczba filmów (200).
        Zwraca:
        - X_movies: np.ndarray (M, P)
        - movie_id_to_row: dict {movie_id -> indeks wiersza 0..M-1}
        - feature_names: lista nazw cech o długości P
        - is_binary_feature: maska bool o długości P wskazująca cechy binarne (0/1)
        """
        movie_ids = sorted(movies_df["id"].astype(int).unique())
        movie_id_to_row = {m_id: idx for idx, m_id in enumerate(movie_ids)}

        parsed_list = [
            self._extract_movie_attributes(tmdb_records.get(m_id, {}))
            for m_id in movie_ids
        ]

        # Wyznaczenie median dla cech ciągłych (imputacja zer/braków bez data leakage)
        valid_budgets = [p["budget"] for p in parsed_list if p["budget"] > 0]
        valid_revenues = [p["revenue"] for p in parsed_list if p["revenue"] > 0]
        valid_runtimes = [p["runtime"] for p in parsed_list if p["runtime"] > 0]
        valid_pops = [p["popularity"] for p in parsed_list if p["popularity"] > 0]
        valid_vavg = [p["vote_average"] for p in parsed_list if p["vote_average"] > 0]
        valid_vcnt = [p["vote_count"] for p in parsed_list if p["vote_count"] > 0]
        valid_years = [
            p["release_year"] for p in parsed_list if not np.isnan(p["release_year"])
        ]

        self.medians_ = {
            "budget": float(np.median(valid_budgets)) if valid_budgets else 15_000_000.0,
            "revenue": float(np.median(valid_revenues)) if valid_revenues else 50_000_000.0,
            "runtime": float(np.median(valid_runtimes)) if valid_runtimes else 120.0,
            "popularity": float(np.median(valid_pops)) if valid_pops else 30.0,
            "vote_average": float(np.median(valid_vavg)) if valid_vavg else 7.9,
            "vote_count": float(np.median(valid_vcnt)) if valid_vcnt else 5000.0,
            "release_year": float(np.median(valid_years)) if valid_years else 1990.0,
        }

        # Zliczanie częstości kategorii w zbiorze 200 filmów
        genre_counter = Counter()
        collection_counter = Counter()
        lang_counter = Counter()
        country_counter = Counter()
        company_counter = Counter()
        director_counter = Counter()
        actor_counter = Counter()
        writer_counter = Counter()

        for p in parsed_list:
            genre_counter.update(p["genres"])
            collection_counter.update(p["collection_name"])
            lang_counter[p["orig_lang"]] += 1   # TODO DLACZEGO TU JEST +=1
            country_counter.update(p["countries"])
            company_counter.update(p["companies"])
            director_counter.update(p["directors"])
            actor_counter.update(p["top10_actors"])
            writer_counter.update(p["writers"])

        selected_genres = sorted(genre_counter.keys())
        selected_collections = sorted(
            [name for name, count in collection_counter.items() if count >= self.min_collection_freq]
        )
        selected_langs = sorted(
            [k for k, v in lang_counter.items() if v >= self.min_lang_freq and k != "unknown"]
        )
        selected_countries = sorted(
            [k for k, v in country_counter.items() if v >= self.min_country_freq]
        )
        selected_companies = sorted(
            [k for k, v in company_counter.items() if v >= self.min_company_freq]
        )
        selected_directors = sorted(
            [k for k, v in director_counter.items() if v >= self.min_director_freq]
        )
        selected_actors = sorted(
            [k for k, v in actor_counter.items() if v >= self.min_actor_freq]
        )
        selected_writers = sorted(
            [k for k, v in writer_counter.items() if v >= self.min_writer_freq]
        )

        # Definicja nazw cech oraz flag binarności
        feature_names: List[str] = []
        is_binary: List[bool] = []

        # Grupa A: Podstawowe numeryczne + flagi
        num_a_features = [
            ("budget_log", False),
            ("revenue_log", False),
            ("runtime", False),
            ("popularity", False),
            ("vote_average", False),
            ("vote_count_log", False),
            ("release_year", False),
        ]
        # Grupa C: Liczniki produkcyjne + is_english
        num_c_features = [
            ("is_english", True),
            ("prod_companies_count", False),
            ("prod_countries_count", False),
            ("spoken_languages_count", False),
        ]
        # Grupa D: Agregaty obsady i ekipy
        num_d_features = [
            ("cast_count", False),
            ("crew_count", False),
            ("female_cast_ratio", False),
            ("belongs_to_collection", True),
        ]

        base_spec = (
            num_a_features
            + num_c_features
            + num_d_features
        )
        for name, b_flag in base_spec:
            feature_names.append(name)
            is_binary.append(b_flag)

        # Multi-hot: Gatunki
        for g in selected_genres:
            feature_names.append(f"genre_{g.replace(' ', '_')}")
            is_binary.append(True)

        # Multi-hot: Kolekcje filmowe
        for collection in selected_collections:
            clean_collection = "".join(
                ch if ch.isalnum() else "_"
                for ch in collection
            ).strip("_")

            feature_names.append(f"collection_{clean_collection}")
            is_binary.append(True)

        # Multi-hot: Języki
        for lg in selected_langs:
            feature_names.append(f"lang_{lg}")
            is_binary.append(True)

        # Multi-hot: Kraje
        for ct in selected_countries:
            feature_names.append(f"country_{ct.replace(' ', '_')}")
            is_binary.append(True)

        # Multi-hot: Wytwórnie
        for cp in selected_companies:
            clean_cp = "".join(ch if ch.isalnum() else "_" for ch in cp).strip("_")
            feature_names.append(f"company_{clean_cp[:25]}")
            is_binary.append(True)

        # Multi-hot: Reżyserzy
        for dr in selected_directors:
            clean_dr = "".join(ch if ch.isalnum() else "_" for ch in dr).strip("_")
            feature_names.append(f"director_{clean_dr}")
            is_binary.append(True)

        # Multi-hot: Aktorzy
        for ac in selected_actors:
            clean_ac = "".join(ch if ch.isalnum() else "_" for ch in ac).strip("_")
            feature_names.append(f"actor_{clean_ac}")
            is_binary.append(True)

        # Multi-hot: Scenarzyści
        for wr in selected_writers:
            clean_wr = "".join(ch if ch.isalnum() else "_" for ch in wr).strip("_")
            feature_names.append(f"writer_{clean_wr}")
            is_binary.append(True)

        self.feature_names_ = feature_names
        self.is_binary_feature_ = np.array(is_binary, dtype=bool)

        # Budowa macierzy X_movies
        M = len(movie_ids)
        P = len(feature_names)
        X_movies = np.zeros((M, P), dtype=np.float64)

        for i, p in enumerate(parsed_list):

            b_val = p["budget"] if p["budget"] > 0 else self.medians_["budget"]
            r_val = p["revenue"] if p["revenue"] > 0 else self.medians_["revenue"]
            rt_val = p["runtime"] if p["runtime"] > 0 else self.medians_["runtime"]
            pop_val = (
                p["popularity"] if p["popularity"] > 0 else self.medians_["popularity"]
            )
            va_val = (
                p["vote_average"]
                if p["vote_average"] > 0
                else self.medians_["vote_average"]
            )
            vc_val = (
                p["vote_count"] if p["vote_count"] > 0 else self.medians_["vote_count"]
            )
            yr_val = (
                p["release_year"]
                if not np.isnan(p["release_year"])
                else self.medians_["release_year"]
            )

            budget_log = float(np.log1p(b_val))
            revenue_log = float(np.log1p(r_val))
            vote_count_log = float(np.log1p(vc_val))

            row_vals: List[float] = [
                budget_log,
                revenue_log,
                rt_val,
                pop_val,
                va_val,
                vote_count_log,
                yr_val,
                1.0 if p["orig_lang"] == "en" else 0.0,
                float(len(p["companies"])),
                float(len(p["countries"])),
                p["spoken_languages_count"],
                p["cast_count"],
                p["crew_count"],
                p["female_cast_ratio"],
                p["belongs_to_collection"],
            ]

            g_set = set(p["genres"])
            for g in selected_genres:
                row_vals.append(1.0 if g in g_set else 0.0)

            collection_set = {p["collection_name"]} if p["collection_name"] else set()
            for collection in selected_collections:
                row_vals.append(1.0 if collection in collection_set else 0.0)

            for lg in selected_langs:
                row_vals.append(1.0 if p["orig_lang"] == lg else 0.0)

            c_set = set(p["countries"])
            for ct in selected_countries:
                row_vals.append(1.0 if ct in c_set else 0.0)

            comp_set = set(p["companies"])
            for cp in selected_companies:
                row_vals.append(1.0 if cp in comp_set else 0.0)

            dir_set = set(p["directors"])
            for dr in selected_directors:
                row_vals.append(1.0 if dr in dir_set else 0.0)

            act_set = set(p["top10_actors"])
            for ac in selected_actors:
                row_vals.append(1.0 if ac in act_set else 0.0)

            wrt_set = set(p["writers"])
            for wr in selected_writers:
                row_vals.append(1.0 if wr in wrt_set else 0.0)

            X_movies[i, :] = row_vals

        return X_movies, movie_id_to_row, self.feature_names_, self.is_binary_feature_
