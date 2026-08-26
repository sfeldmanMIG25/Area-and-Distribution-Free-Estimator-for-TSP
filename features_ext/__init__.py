"""Candidate feature-group modules for GART 2.0 screening.

Each module in this package exposes the same two-function contract:

    feature_names() -> list[str]
    compute(coords: np.ndarray, mst_csr=None) -> dict[str, float]

so a screening driver can iterate over groups uniformly. Modules here are
candidates only; nothing in this package is wired into the production
feature extractor (``feature_creator_v3.compute_features_for_instance_v3``).
"""
