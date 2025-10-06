"""Geometry and manifold evaluation metrics.

This module implements a subset of the geometric diagnostics described in
`docs/idea.md` to evaluate learned feature embeddings for the NB-TCM-CHM dataset.

Implemented metrics (minimal viable versions):
 - trustworthiness
 - continuity
 - geodesic distortion (relative error between feature-space distances and a reference geodesic estimate)
 - local intrinsic dimension (TwoNN & MLE variants)
 - curvature proxy (local PCA residual ratio)
 - inter-class margin (minimum inter-class distance averaged)
 - neighborhood overlap across sources

Design notes:
We treat a set of baseline features (e.g. produced by an initial frozen backbone
or the raw pixel transform features) as the *reference space* to approximate
original geodesic relations using a k-NN graph and shortest-path distances.
Computed geodesic distances are *sparse* (only among neighbors or via graph
shortest paths) to stay memory efficient.

The functions here are intentionally lightweight and avoid heavy dependencies.
If later higher fidelity is needed, one can swap in e.g. UMAP / KeOps / faiss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# k-NN graph + (approximate) geodesic distance utilities
# ---------------------------------------------------------------------------
def build_knn_graph(features: np.ndarray, k: int = 10, metric: str = "euclidean") -> Tuple[csr_matrix, np.ndarray]:
    """Build a symmetric k-NN graph (undirected) and return adjacency matrix.

    Args:
        features: (N, D) array of reference-space features.
        k: number of neighbors.
        metric: distance metric for sklearn NearestNeighbors.

    Returns:
        adjacency: CSR sparse matrix with edge weights = distances.
        knn_indices: (N, k) neighbor indices.
    """
    n, d = features.shape
    k_eff = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric).fit(features)
    distances, indices = nbrs.kneighbors(features, return_distance=True)
    # discard self index (first column assumed self)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    rows = np.repeat(np.arange(n), k_eff)
    cols = indices.reshape(-1)
    data = distances.reshape(-1)
    # Build directed then symmetrize by taking min distance if duplicate
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Symmetrize (keep minimum weight)
    adjacency = adjacency.minimum(adjacency.T)
    return adjacency, indices


def approximate_geodesic_distances(adjacency: csr_matrix, max_nodes: int = 5000) -> np.ndarray:
    """Compute all-pairs shortest paths (geodesic approximation) for a graph.

    For large N this becomes heavy (O(N^3)) if dense; we rely on sparse
    Dijkstra via SciPy. To keep runtime bounded we optionally cap to the first
    `max_nodes` samples (sufficient for diagnostics on smaller NB-TCM-CHM).

    Args:
        adjacency: CSR adjacency with non-zero positive distances.
        max_nodes: limit for computing full distance matrix.

    Returns:
        dist_matrix: (M, M) array of shortest path distances (M<=N).
    """
    n = adjacency.shape[0]
    m = min(n, max_nodes)
    sub_adj = adjacency[:m, :m]
    # Use Dijkstra (positive weights)
    dist = shortest_path(sub_adj, directed=False, unweighted=False)
    return dist


# ---------------------------------------------------------------------------
# Trustworthiness & Continuity
# ---------------------------------------------------------------------------
def _rank_neighbors(X: np.ndarray, k: int) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="euclidean").fit(X)
    _, indices = nbrs.kneighbors(X)
    return indices[:, 1:]


def trustworthiness(original: np.ndarray, embedded: np.ndarray, k: int = 10) -> float:
    """Compute trustworthiness (sklearn-like) without importing sklearn.manifold.

    Measures fraction of embedded k-neighbors that are also neighbors in the
    original space; penalizes intrusions.
    """
    k = min(k, len(original) - 1)
    orig_nn = _rank_neighbors(original, k)
    emb_nn = _rank_neighbors(embedded, k)
    n = len(original)
    ranks = {i: {nbr: r for r, nbr in enumerate(orig_nn[i])} for i in range(n)}
    t_sum = 0.0
    for i in range(n):
        for nbr in emb_nn[i]:
            if nbr not in ranks[i]:
                # intrusion: rank assumed > k, approximate penalty as k+1
                # more precise version would precompute full ranks; heuristic suffice
                t_sum += k + 1
    normalizer = n * k * (2 * n - 3 * k - 1)
    return 1.0 - (2.0 / normalizer) * t_sum


def continuity(original: np.ndarray, embedded: np.ndarray, k: int = 10) -> float:
    """Continuity: reciprocal of trustworthiness with roles swapped.
    Penalizes missing neighbors (extrusions) from embedded representation.
    """
    return trustworthiness(embedded, original, k)


# ---------------------------------------------------------------------------
# Geodesic distortion
# ---------------------------------------------------------------------------
def geodesic_distortion(embedded: np.ndarray, geo_ref: np.ndarray) -> float:
    """Mean relative error between Euclidean distances in embedded space and
    reference geodesic distances (upper triangle, excluding diagonal).
    """
    m = geo_ref.shape[0]
    emb_d = _pairwise_dists(embedded[:m])
    mask = ~np.isinf(geo_ref)
    triu = np.triu(mask, 1)
    denom = geo_ref[triu]
    num = np.abs(emb_d[triu] - geo_ref[triu])
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(num / denom))


def _pairwise_dists(X: np.ndarray) -> np.ndarray:
    # (a - b)^2 = a^2 + b^2 - 2ab
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = sq + sq.T - 2 * (X @ X.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)


# ---------------------------------------------------------------------------
# Local intrinsic dimension estimators
# ---------------------------------------------------------------------------
def twonn_intrinsic_dim(features: np.ndarray, k: int = 3) -> float:
    """TwoNN intrinsic dimension estimator (Facco et al.).
    For each point we compute the ratio r = d2/d1 among its first 2 neighbors.
    ID ≈ 1 / ( (1/N) * Σ log r ).
    """
    nbrs = NearestNeighbors(n_neighbors=3).fit(features)
    dists, _ = nbrs.kneighbors(features)
    d1 = dists[:, 1]
    d2 = dists[:, 2]
    r = d2 / np.maximum(d1, 1e-12)
    r = r[r > 1.0]  # filter numerical issues
    if len(r) == 0:
        return 0.0
    return float(1.0 / np.mean(np.log(r)))


def local_pca_residual(features: np.ndarray, k: int = 15, d: Optional[int] = None) -> float:
    """Curvature proxy via local PCA residual energy ratio.

    For each point: compute k-NN neighborhood, do PCA on centered neighbors,
    retain either (d) (if given) leading components (or TwoNN global estimate
    rounded), measure residual variance / total variance. Average across points.
    Higher residual => higher curvature / noise.
    """
    n = len(features)
    k_eff = min(k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(features)
    _, indices = nbrs.kneighbors(features)
    if d is None:
        d_est = max(1, int(round(twonn_intrinsic_dim(features))))
    else:
        d_est = d
    residuals = []
    for i in range(n):
        neigh = features[indices[i, 1:]]  # drop self
        C = np.cov(neigh.T)
        eigvals = np.sort(np.linalg.eigvalsh(C))[::-1]
        if eigvals.sum() <= 0:
            continue
        keep = eigvals[:d_est]
        res = 1.0 - keep.sum() / eigvals.sum()
        residuals.append(res)
    if not residuals:
        return 0.0
    return float(np.mean(residuals))


# ---------------------------------------------------------------------------
# Inter-class margin & cross-source neighborhood overlap
# ---------------------------------------------------------------------------
def inter_class_margin(embedded: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean minimum distance to another class for each sample."""
    dmat = _pairwise_dists(embedded)
    n = len(labels)
    margins = []
    for i in range(n):
        other = dmat[i][labels != labels[i]]
        if len(other) == 0:
            continue
        margins.append(np.min(other))
    if not margins:
        return 0.0
    return float(np.mean(margins))


def neighborhood_overlap(source_a: np.ndarray, source_b: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray, k: int = 10) -> float:
    """Cross-source neighborhood overlap: For each sample in B, compute the
    fraction of its k nearest neighbors in A that have the same class label.
    Returns average fraction.
    """
    if len(source_a) == 0 or len(source_b) == 0:
        return 0.0
    k = min(k, len(source_a))
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(source_a)
    _, idx = nbrs.kneighbors(source_b)
    overlaps = []
    for i in range(len(source_b)):
        same = (labels_a[idx[i]] == labels_b[i]).mean()
        overlaps.append(same)
    return float(np.mean(overlaps))


# ---------------------------------------------------------------------------
# Aggregate evaluation helper
# ---------------------------------------------------------------------------
@dataclass
class GeometryReport:
    trustworthiness: float
    continuity: float
    geodesic_distortion: float
    twonn_id: float
    curvature_residual: float
    inter_class_margin: float
    cross_source_overlap: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        d = {
            "trustworthiness": self.trustworthiness,
            "continuity": self.continuity,
            "geodesic_distortion": self.geodesic_distortion,
            "twonn_id": self.twonn_id,
            "curvature_residual": self.curvature_residual,
            "inter_class_margin": self.inter_class_margin,
        }
        if self.cross_source_overlap is not None:
            d["cross_source_overlap"] = self.cross_source_overlap
        return d


def evaluate_geometry(
    reference_features: np.ndarray,
    embedded_features: np.ndarray,
    labels: np.ndarray,
    source_flags: Optional[np.ndarray] = None,
    k: int = 10,
    max_geo_nodes: int = 3000,
) -> GeometryReport:
    """Compute a GeometryReport given reference-space features and learned embedding.

    Args:
        reference_features: Original/reference features (N, D0)
        embedded_features: Learned features (N, D1)
        labels: Class labels (N,)
        source_flags: Optional binary array (N,) where 0 denotes source A (web) and 1 denotes source B (phone)
        k: neighborhood size for k-NN based metrics
        max_geo_nodes: cap for all-pairs geodesic computation
    """
    adjacency, _ = build_knn_graph(reference_features, k=k)
    geo = approximate_geodesic_distances(adjacency, max_nodes=max_geo_nodes)
    gdist = geodesic_distortion(embedded_features, geo)
    t = trustworthiness(reference_features, embedded_features, k=k)
    c = continuity(reference_features, embedded_features, k=k)
    id_est = twonn_intrinsic_dim(embedded_features)
    curv = local_pca_residual(embedded_features, k=max(5, k))
    margin = inter_class_margin(embedded_features, labels)
    overlap = None
    if source_flags is not None:
        a_mask = source_flags == 0
        b_mask = source_flags == 1
        if a_mask.any() and b_mask.any():
            overlap = neighborhood_overlap(
                embedded_features[a_mask],
                embedded_features[b_mask],
                labels[a_mask],
                labels[b_mask],
                k=min(k, np.sum(a_mask)),
            )
    return GeometryReport(
        trustworthiness=t,
        continuity=c,
        geodesic_distortion=gdist,
        twonn_id=id_est,
        curvature_residual=curv,
        inter_class_margin=margin,
        cross_source_overlap=overlap,
    )


__all__ = [
    "build_knn_graph",
    "approximate_geodesic_distances",
    "trustworthiness",
    "continuity",
    "geodesic_distortion",
    "twonn_intrinsic_dim",
    "local_pca_residual",
    "inter_class_margin",
    "neighborhood_overlap",
    "evaluate_geometry",
    "GeometryReport",
]
