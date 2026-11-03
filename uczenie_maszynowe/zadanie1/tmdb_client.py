import os
import json
import time
from typing import Dict, Any, Optional

import requests
import pandas as pd


class TMDBRawClient:

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

        # Zwraca słownik {movie_id: surowy_json_tmdb}.
        # Jeśli film nie znajduje się jeszcze w pliku cache, pobiera go z API

        raw_cache: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.raw_cache_path):
            print(f"[INFO] Wczytywanie surowych danych TMDB z pliku: {self.raw_cache_path}")
            with open(self.raw_cache_path, "r", encoding="utf-8") as f:
                raw_cache = json.load(f)

        updated = False
        total = len(movies_df)
        useful_keys = {
            "genres", "release_date", "original_language", "overview",
            "production_companies", "production_countries", "runtime",
            "credits", "keywords", "budget", "revenue", "vote_average",
        }

        for idx, row in movies_df.iterrows():
            m_id_str = str(int(row["movie_id"]))
            tmdb_id = int(row["tmdb_id"])
            title = str(row["title"])

            cached = raw_cache.get(m_id_str)
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
                    print(f"[OSTRZEŻENIE] Brak metadanych TMDB dla filmu {title!r}; zostanie użyty tytuł.")
                    raw_json = {"title": title, "id": tmdb_id}
                raw_cache[m_id_str] = raw_json
                updated = True

        if updated:
            with open(self.raw_cache_path, "w", encoding="utf-8") as f:
                json.dump(raw_cache, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Zapisano surowe dane filmów do pliku: {self.raw_cache_path}")

        return {int(k): v for k, v in raw_cache.items()}