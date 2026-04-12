"""
Classical Multidimensional Scaling (Torgerson 1952 / Gower 1966).

Given a square symmetric distance matrix ``D`` (n x n), classical MDS returns
Euclidean coordinates ``X`` (n x k) such that the pairwise Euclidean distances
of ``X`` approximate ``D`` as closely as possible in a least-squares sense.
When ``D`` is itself generated from points in a Euclidean space of dimension
k*, classical MDS recovers those points exactly (up to rotation and reflection)
whenever k >= k*.

Algorithm
---------
1. Form the squared-distance matrix D^2.
2. Double-center it:   B = -1/2 * J * D^2 * J, where J = I - (1/n) * 11^T.
   B is the Gram (inner-product) matrix of the points once a common centroid
   is subtracted.
3. Eigendecompose B = V * Lambda * V^T. Because B is symmetric, eigenvalues are
   real; because the underlying distances are Euclidean, eigenvalues are >= 0.
   Negative eigenvalues indicate that D is *not* exactly Euclidean (e.g. GEO
   great-circle distances) and are dropped.
4. Retain the top-k positive eigenvalues. The embedding is
       X = V_k * diag(sqrt(lambda_k)),
   i.e. the k columns of V corresponding to the largest eigenvalues scaled by
   the square root of those eigenvalues.

Dimension selection
-------------------
The intrinsic dimensionality of a distance matrix under classical MDS is the
number of *non-trivial* positive eigenvalues of the double-centered matrix.
We pick the smallest k such that the first k positive eigenvalues explain at
least ``variance_threshold`` of the total positive eigenvalue mass, subject to
a hard cap ``max_dim``. Negative and (near-)zero eigenvalues are ignored.

References
----------
Torgerson, W. S. (1952). Multidimensional scaling: I. Theory and method.
    Psychometrika, 17(4), 401-419.
Gower, J. C. (1966). Some distance properties of latent root and vector methods
    used in multivariate analysis. Biometrika, 53(3/4), 325-338.
Borg, I., & Groenen, P. J. F. (2005). Modern Multidimensional Scaling: Theory
    and Applications. Springer, Chapter 12.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def double_center(D_squared: np.ndarray) -> np.ndarray:
    """Apply double-centering to a squared-distance matrix.

    B = -1/2 * J * D^2 * J  with  J = I - (1/n) * 11^T.

    Parameters
    ----------
    D_squared : (n, n) ndarray
        Matrix of squared pairwise distances.

    Returns
    -------
    B : (n, n) ndarray
        Gram matrix of the points whose pairwise squared distances match D^2
        (once their centroid is placed at the origin).
    """
    n = D_squared.shape[0]
    row_mean = D_squared.mean(axis=0, keepdims=True)
    col_mean = D_squared.mean(axis=1, keepdims=True)
    grand_mean = D_squared.mean()
    return -0.5 * (D_squared - row_mean - col_mean + grand_mean)


def classical_mds(
    distance_matrix: np.ndarray,
    max_dim: int = 100,
    variance_threshold: float = 0.999,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Embed a distance matrix into Euclidean space via classical MDS.

    Parameters
    ----------
    distance_matrix : (n, n) ndarray
        Symmetric pairwise distances (not squared). Must be non-negative with
        zero diagonal.
    max_dim : int
        Hard cap on the embedded dimensionality. If the natural MDS dimension
        exceeds this, the embedding is truncated to ``max_dim``.
    variance_threshold : float in (0, 1]
        Fraction of positive eigenvalue mass that the selected embedding must
        retain. The smallest k satisfying this (up to ``max_dim``) is chosen.
    eps : float
        Numerical floor for treating eigenvalues as zero.

    Returns
    -------
    X : (n, k) ndarray
        Embedded coordinates. k is at most ``max_dim`` and at most the number
        of positive eigenvalues of the double-centered matrix.
    eigenvalues : (k_full,) ndarray
        All positive eigenvalues of the double-centered matrix (sorted desc),
        before truncation. Useful for diagnostics.
    info : dict
        Embedding diagnostics:
            ``chosen_dim``            : k actually used
            ``natural_dim``           : number of eigenvalues > eps
            ``variance_retained``     : sum(lambda_k) / sum(positive_lambdas)
            ``negative_eigvalue_mass``: |sum of negative eigenvalues| (measure
                                       of how non-Euclidean the input is).
            ``strain``                : classical MDS strain metric
                                       sqrt(sum(lambda_dropped^2) /
                                            sum(positive_lambdas^2)).
    """
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"distance_matrix must be square; got shape {D.shape}")

    # Step 1: squared distances
    D2 = D * D

    # Step 2: double-centering -> Gram matrix B
    B = double_center(D2)
    # Force exact symmetry to avoid tiny asymmetries after arithmetic
    B = 0.5 * (B + B.T)

    # Step 3: eigendecomposition
    # np.linalg.eigh returns eigenvalues in ascending order.
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = eigvals > eps
    pos_eigvals = eigvals[positive]
    pos_eigvecs = eigvecs[:, positive]
    neg_mass = float(-eigvals[eigvals < 0].sum()) if np.any(eigvals < 0) else 0.0

    if pos_eigvals.size == 0:
        raise ValueError(
            "Double-centered matrix has no positive eigenvalues; "
            "input distances appear degenerate."
        )

    total_positive = pos_eigvals.sum()
    cumulative = np.cumsum(pos_eigvals) / total_positive
    # Smallest k such that cumulative[k-1] >= variance_threshold.
    k_nat = int(np.searchsorted(cumulative, variance_threshold) + 1)
    k_nat = min(k_nat, pos_eigvals.size)
    k = min(k_nat, max_dim, n - 1)

    # Step 4: embedding
    X = pos_eigvecs[:, :k] * np.sqrt(pos_eigvals[:k])[np.newaxis, :]

    dropped = pos_eigvals[k:]
    strain = float(np.sqrt((dropped ** 2).sum() / (pos_eigvals ** 2).sum()))
    retained = float(pos_eigvals[:k].sum() / total_positive)

    info = {
        "chosen_dim": k,
        "natural_dim": int(pos_eigvals.size),
        "variance_retained": retained,
        "negative_eigvalue_mass": neg_mass,
        "strain": strain,
    }
    return X.astype(np.float64), pos_eigvals, info


if __name__ == "__main__":
    # Self-test: embed a random 5D point cloud, verify MDS recovers pairwise distances.
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(30, 5))
    diff = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((diff * diff).sum(-1))
    X, eigs, info = classical_mds(D, max_dim=100, variance_threshold=1.0)
    diff2 = X[:, None, :] - X[None, :, :]
    D2 = np.sqrt((diff2 * diff2).sum(-1))
    err = float(np.max(np.abs(D - D2)))
    print("chosen dim  :", info["chosen_dim"])
    print("natural dim :", info["natural_dim"])
    print("variance    :", info["variance_retained"])
    print("max abs err :", err)
    assert err < 1e-9, "Classical MDS failed to recover Euclidean distances exactly"
    print("Self-test passed.")
