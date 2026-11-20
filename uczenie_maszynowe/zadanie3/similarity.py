import numpy as np


def _compute_jaccard_seen_matrix(mask_f: np.ndarray, common: np.ndarray) -> np.ndarray:
    watched_counts = mask_f.sum(axis=1, keepdims=True)
    union = watched_counts + watched_counts.T - common
    jaccard_seen = np.zeros_like(common, dtype=np.float64)
    np.divide(common, union, out=jaccard_seen, where=union > 0)
    return jaccard_seen


def compute_user_similarity_matrix(
    R: np.ndarray,
    metric: str = "jaccard_pearson",
    min_common: int = 1,
) -> np.ndarray:
    mask = ~np.isnan(R)
    mask_f = mask.astype(np.float64)
    common = mask_f @ mask_f.T
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

    sim = np.where(common >= max(1, int(min_common)), sim, 0.0)
    np.fill_diagonal(sim, 0.0)
    return np.clip(sim, -1.0, 1.0)