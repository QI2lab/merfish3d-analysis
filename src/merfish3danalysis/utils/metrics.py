"""Same-gene spatial matching for simulated transcript coordinates."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree


def calculate_F1_with_radius(
    qi2lab_coords: ArrayLike,
    qi2lab_gene_ids: ArrayLike,
    gt_coords: ArrayLike,
    gt_gene_ids: ArrayLike,
    radius: float,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy closest-first matching within `radius`, with strict same-gene and one-to-one constraints.

    Returns F1 metrics and (TP coords, FP coords, FN coords).

    Parameters
    ----------
    qi2lab_coords : ArrayLike
        Decoded transcript coordinates.
    qi2lab_gene_ids : ArrayLike
        Decoded transcript gene identifiers.
    gt_coords : ArrayLike
        Ground-truth transcript coordinates.
    gt_gene_ids : ArrayLike
        Ground-truth gene identifiers.
    radius : float
        Maximum match distance.

    Returns
    -------
    tuple[dict, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Metrics, true-positive coordinates, false-positive coordinates, and
        false-negative coordinates.
    """
    # Canonicalize inputs
    qi2lab_coords = np.asarray(qi2lab_coords, float)
    gt_coords = np.asarray(gt_coords, float)
    # Force gene IDs to a common comparable dtype (strings) to avoid category/int/object mismatches
    qi2lab_gene_ids = np.asarray(qi2lab_gene_ids).astype(str)
    gt_gene_ids = np.asarray(gt_gene_ids).astype(str)

    Nq = qi2lab_coords.shape[0]
    Ng = gt_coords.shape[0]

    # Trivial cases
    if Nq == 0 and Ng == 0:
        return (
            {
                "F1 Score": 1.0,
                "Precision": 1.0,
                "Recall": 1.0,
                "True Positives": 0,
                "False Positives": 0,
                "False Negatives": 0,
            },
            np.empty((0, 3), float),
            np.empty((0, 3), float),
            np.empty((0, 3), float),
        )
    if Nq == 0:
        return (
            {
                "F1 Score": 0.0,
                "Precision": 0.0,
                "Recall": 0.0 if Ng > 0 else 1.0,
                "True Positives": 0,
                "False Positives": 0,
                "False Negatives": int(Ng),
            },
            np.empty((0, 3), float),
            np.empty((0, 3), float),
            gt_coords.copy(),
        )
    if Ng == 0:
        return (
            {
                "F1 Score": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "True Positives": 0,
                "False Positives": int(Nq),
                "False Negatives": 0,
            },
            np.empty((0, 3), float),
            qi2lab_coords.copy(),
            np.empty((0, 3), float),
        )

    # Build candidate pairs within radius, per shared gene (strict same-gene pooling)
    pair_q_idx_all: list[np.ndarray] = []
    pair_g_idx_all: list[np.ndarray] = []
    pair_dist_all: list[np.ndarray] = []

    common_genes = np.intersect1d(np.unique(qi2lab_gene_ids), np.unique(gt_gene_ids))
    for gene in common_genes:
        q_idx = np.flatnonzero(qi2lab_gene_ids == gene)
        g_idx = np.flatnonzero(gt_gene_ids == gene)

        q_pts = qi2lab_coords[q_idx]
        g_pts = gt_coords[g_idx]

        q_tree = cKDTree(q_pts)
        g_tree = cKDTree(g_pts)
        dist_coo = q_tree.sparse_distance_matrix(
            g_tree, max_distance=radius, output_type="coo_matrix"
        )

        if dist_coo.nnz == 0:
            continue

        # Local (per-gene) -> global indices
        pair_q_idx_all.append(q_idx[dist_coo.row])
        pair_g_idx_all.append(g_idx[dist_coo.col])
        pair_dist_all.append(dist_coo.data)

    if not pair_q_idx_all:
        q_used = np.zeros(Nq, dtype=bool)
        g_used = np.zeros(Ng, dtype=bool)
        tp_coords = np.empty((0, 3), float)
        fp_coords = qi2lab_coords.copy()
        fn_coords = gt_coords.copy()
        tp = 0
    else:
        pair_q_idx = np.concatenate(pair_q_idx_all)
        pair_g_idx = np.concatenate(pair_g_idx_all)
        pair_dist = np.concatenate(pair_dist_all)

        # Sort by distance ascending; stable for deterministic ties
        order = np.argsort(pair_dist, kind="stable")
        pair_q_idx = pair_q_idx[order]
        pair_g_idx = pair_g_idx[order]

        # Greedy one-to-one matching of the same-gene candidate pairs.
        q_used = np.zeros(Nq, dtype=bool)
        g_used = np.zeros(Ng, dtype=bool)
        matched_q = []

        for qi, gi in zip(pair_q_idx, pair_g_idx, strict=False):
            if q_used[qi] or g_used[gi]:
                continue
            q_used[qi] = True
            g_used[gi] = True
            matched_q.append(qi)

        matched_q = np.asarray(matched_q, dtype=int)

        tp = int(matched_q.size)
        tp_coords = qi2lab_coords[matched_q]
        fp_coords = qi2lab_coords[~q_used]
        fn_coords = gt_coords[~g_used]

    # Counts & metrics
    fp = int((~q_used).sum())
    fn = int((~g_used).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    results = {
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
    }
    return results, tp_coords, fp_coords, fn_coords
