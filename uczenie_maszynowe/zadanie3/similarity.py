"""
Moduł obliczania podobieństwa pomiędzy użytkownikami (Person Similarity)
w oparciu o ich dotychczasowe oceny filmów.

Zgodnie ze Wskazówką 1 i 2 (Hints 1 & 2):
1. Cechy filmów nie są wykorzystywane — bazujemy wyłącznie na ocenach (0..5).
2. Podobieństwo dwóch osób jest obliczane WYŁĄCZNIE na podstawie filmów,
   które obie te osoby widziały (wspólnie ocenione filmy: I_uv = I_u n I_v).

Obsługiwane miary podobieństwa:
- 'jaccard_adjcos': hybryda Adjusted Cosine na wspólnych filmach pomnożona przez
  indeks Jaccarda wspólnie obejrzanych filmów |I_u n I_v| / |I_u u I_v|.
- 'jaccard_pearson': hybryda korelacji Pearsona na wspólnych filmach pomnożona przez
  indeks Jaccarda wspólnie obejrzanych filmów |I_u n I_v| / |I_u u I_v|.
- 'jaccard_agreement': ułamek filmów ze zgodnymi ocenami (|r_u - r_v| <= 1) na przekroju
  wspólnie obejrzanych filmów względem sumy zbiorów obejrzanych filmów |I_u u I_v|.
"""

import numpy as np


def _compute_jaccard_seen_matrix(mask_f: np.ndarray, common: np.ndarray) -> np.ndarray:
    """
    Oblicza klasyczną macierz podobieństwa Jaccarda zbiorów obejrzanych filmów:
    J(u, v) = |I_u n I_v| / |I_u u I_v|.
    """
    watched_counts = mask_f.sum(axis=1, keepdims=True)
    union = watched_counts + watched_counts.T - common
    jaccard_seen = np.zeros_like(common, dtype=np.float64)
    np.divide(common, union, out=jaccard_seen, where=union > 0)
    return jaccard_seen


def compute_user_similarity_matrix(
    R: np.ndarray,
    metric: str = "jaccard_pearson",
    shrinkage: float = 10.0,
    min_common: int = 2,
) -> np.ndarray:
    """
    Wektoryzowane obliczenie macierzy podobieństwa S o wymiarach (U, U).
    Każdy element S[u, v] jest wyznaczany wyłącznie na podstawie filmów,
    które ocenili zarówno użytkownik u, jak i użytkownik v (Hint 2).
    """
    mask = ~np.isnan(R)  # (U, M) - True tam, gdzie użytkownik widział film
    mask_f = mask.astype(np.float64)
    common = mask_f @ mask_f.T  # common[u, v] = liczba wspólnie obejrzanych filmów
    R_zero = np.where(mask, R, 0.0)

    if metric in ("jaccard_pearson"):
        safe_common = np.maximum(common, 1.0)
        sum_u_common = R_zero @ mask_f.T
        sum_v_common = mask_f @ R_zero.T
        mean_u_common = sum_u_common / safe_common
        mean_v_common = sum_v_common / safe_common

        cross = R_zero @ R_zero.T
        cov = cross - common * mean_u_common * mean_v_common

        R_sq = R_zero ** 2
        sum_sq_u_common = R_sq @ mask_f.T
        sum_sq_v_common = mask_f @ R_sq.T
        var_u = np.maximum(0.0, sum_sq_u_common - common * (mean_u_common ** 2))
        var_v = np.maximum(0.0, sum_sq_v_common - common * (mean_v_common ** 2))
        denom = np.sqrt(var_u * var_v)

        sim = np.zeros_like(common, dtype=np.float64)
        np.divide(cov, denom, out=sim, where=denom > 1e-12)

        jaccard_seen = _compute_jaccard_seen_matrix(mask_f, common)
        sim = sim * jaccard_seen

    elif metric in ("jaccard_adjcos"):
        user_counts = np.maximum(mask_f.sum(axis=1, keepdims=True), 1.0)
        user_means = np.nansum(R, axis=1, keepdims=True) / user_counts
        R_centered = np.where(mask, R - user_means, 0.0)

        num = R_centered @ R_centered.T
        R_sq = R_centered ** 2
        sum_sq_u = R_sq @ mask_f.T
        sum_sq_v = mask_f @ R_sq.T
        denom = np.sqrt(sum_sq_u * sum_sq_v)

        sim = np.zeros_like(common, dtype=np.float64)
        np.divide(num, denom, out=sim, where=denom > 1e-12)

        if metric == "jaccard_adjcos":
            jaccard_seen = _compute_jaccard_seen_matrix(mask_f, common)
            sim = sim * jaccard_seen

    elif metric == "jaccard_agreement":
        # Zgodność ocen Jaccarda: |{m in I_uv : |r_u - r_v| <= 1}| / |I_u u I_v|
        watched_counts = mask_f.sum(axis=1, keepdims=True)
        union = watched_counts + watched_counts.T - common
        U = R.shape[0]
        sim = np.zeros((U, U), dtype=np.float64)
        for u in range(U):
            both_seen = mask[u][None, :] & mask
            agree_mask = (np.abs(R_zero[u][None, :] - R_zero) <= 1.0) & both_seen
            agree_counts = agree_mask.sum(axis=1).astype(np.float64)
            np.divide(agree_counts, union[u], out=sim[u], where=union[u] > 0)

    else:
        raise ValueError(f"Nieznana miara podobieństwa: {metric}")

    # Significance weighting (tylko gdy shrinkage > 0 i nie użyliśmy już wagi Jaccarda)
    if shrinkage > 0.0 and not metric.startswith("jaccard_"):
        sim = sim * (common / (common + shrinkage))

    # Warunek minimalnej liczby wspólnie obejrzanych filmów (Hint 2)
    sim = np.where(common >= max(1, int(min_common)), sim, 0.0)
    np.fill_diagonal(sim, 0.0)
    return np.clip(sim, -1.0, 1.0)