import math
from typing import Dict, Any

import numpy as np
import pandas as pd


class RawFeatureProcessor:
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

            if isinstance(val, (int, float)) and not isinstance(val, bool):
                val_float = float(val)
                if key in cls.LOG_TRANSFORM_KEYS:
                    features[f"num_{key}_log"] = math.log1p(val_float) if val_float > 0 else np.nan
                else:
                    features[f"num_{key}"] = val_float if val_float > 0 else np.nan

            elif key == "release_date" and isinstance(val, str):
                if len(val) >= 4 and val[:4].isdigit():
                    features["num_release_year"] = float(val[:4])
                else:
                    features["num_release_year"] = np.nan

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

            crew_raw = credits_data.get("crew") or []
            directors_set = {
                str(m["name"]).strip().lower()
                for m in crew_raw
                if isinstance(m, dict) and m.get("name") and m.get("job") == "Director"
            }
            if directors_set:
                features["set_director"] = directors_set

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

        orig_lang = raw.get("original_language")
        if orig_lang and isinstance(orig_lang, str):
            features["cat_original_language"] = orig_lang.strip().lower()

        collection = raw.get("belongs_to_collection")
        if isinstance(collection, dict) and collection.get("name"):
            features["cat_belongs_to_collection"] = str(collection["name"]).strip().lower()

        tagline = (raw.get("tagline") or "").strip()
        overview = (raw.get("overview") or "").strip()
        combined_text = f"{tagline} {overview}".strip()
        if combined_text:
            features["text_overview"] = combined_text
        elif fallback_title:
            features["text_overview"] = fallback_title

        return features

    @classmethod
    def transform_all(cls, raw_movies: Dict[int, Dict[str, Any]], movies_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        # Przekształca surowe dane wszystkich filmów w słownik cech gotowych do obliczeń.
        processed: Dict[int, Dict[str, Any]] = {}
        title_map = dict(zip(movies_df["movie_id"].astype(int), movies_df["title"].astype(str)))

        for m_id, raw_json in raw_movies.items():
            fallback = title_map.get(m_id, "")
            processed[m_id] = cls.transform_single_movie(raw_json, fallback_title=fallback)

        return processed