"""MST topology features beyond the aggregates GART 2.0 already carries.

Motivation
----------
The 30 existing features summarise the MST only through *aggregate* statistics
(edge moments/quantiles, leaf ratio, degree mean/std/max, weighted diameter and
weighted-diameter/total-weight).  None of them describe the *shape* of the tree:
whether it is a path, a union of paths, or a bushy branching structure, and
whether the path it contains runs straight or doubles back.

Degenerate instances (points pinned to a box face, points on a line segment,
polylines) produce MSTs that are long chains of degree-2 vertices whose
consecutive edges are nearly antiparallel.  Isotropic clouds do not.  The
features here measure exactly that, and they do it from tree topology plus edge
*directions*, so they fire on genuine collinear clouds, on polylines and on
boundary pile-ups alike, without reference to any particular generator.

The straightness feature is theoretically motivated: a prior analysis found
alpha ~ 1 + D / L for near-1D point sets, with D the separation of the two
extreme points and L the arc length joining them.  ``mst_topology_straightness``
is exactly that D / L measured on the MST's diameter path.  It is *not* the
existing normalised diameter, which divides the diameter by the MST's total
weight.

Properties
----------
* Dimension-agnostic: every quantity is a count, a ratio of counts, a ratio of
  Euclidean lengths, or a cosine.  Valid for d = 2 .. 100 unchanged.
* Cost: O(n) beyond the MST itself (four O(n) tree sweeps plus one pass over
  edges).  No convex hull, no solver, no distance matrix of its own.
* Scale invariant: all 12 features are ratios, counts/n, or cosines.  None
  carries a length unit.
* Rotation/translation/reflection invariant: only Euclidean distances and inner
  products of *difference* vectors are used.
* Deterministic: no randomness; every tie-break is a stable sort or an argmax
  over a fixed traversal order.
* No target leakage: reads coordinates and the MST only.

Conditioning on exactly-degenerate geometry
-------------------------------------------
On an exact lattice the Euclidean MST is massively non-unique: every spanning
tree of the unit-distance graph has the same total weight, but their diameter
paths differ enormously (a snake through an m x m grid has m^2 - 1 hops, a comb
has ~2m).  A path functional such as ``straightness`` is therefore genuinely
discontinuous there, and *no* tie-break rule can make it a continuous function
of the coordinates.

Measured on the unfixed module, a relative-1e-9 coordinate jitter moved
``straightness`` by up to 0.059 and ``deg2_straight_mean`` by up to 0.156 on
hexagonal lattices, and a pure row permutation -- which cannot change the point
set at all -- moved them by up to 0.882 on a square lattice.  The second is a
correctness bug outright.

The repair has three parts, each closing one channel through which noise was
reaching the answer:

1. ``canonical_coords`` orders the points by a per-axis CLUSTER index, so the
   vertex numbering is a function of the point set rather than of the caller's
   row order.  Clustering, not rounding: a fixed rounding grid has boundary
   discontinuities and a randomised rotated-lattice battery still flipped 35%
   of the time with one.
2. ``_canonical_mst`` re-derives the tree by Kruskal over every edge that could
   belong to ANY MST, with lengths bucketed at ``TIE_REL_TOL`` and ties broken
   on the canonical vertex pair.  It engages only when the length spectrum is
   actually degenerate, so a generic cloud pays nothing -- but that is not a
   rare path: it fires on 49 of the 78 real TSPLIB EUC_2D instances.
3. ``_farthest`` selects the diameter endpoint by a two-pass rule with the same
   tolerance.  Without it a stable tree still yielded an unstable answer,
   because on a lattice the candidate path lengths tie exactly and a 1e-9
   perturbation silently re-elects the endpoint.

Measured after the repair: zero flips in 300 randomised rotated-and-scaled
lattice trials at 1e-9 jitter (worst move 1.8e-08, pure float propagation), and
exactly 0.0 movement under row permutation.

What it deliberately does NOT claim: continuity in the coordinates for
perturbations LARGER than ``TIE_REL_TOL``.  Above that the perturbation genuinely
changes which spanning tree is minimal, and the value jumps.  That is correct
behaviour, not a residual bug: the quantity really is discontinuous there, and
the module's job is to be a well-defined function of the point set, not a
continuous one.

Lengths and direction vectors are taken from the ORIGINAL coordinates (merely
reordered), so on generic input, where the MST is unique with a healthy margin,
feature values are unchanged.  Measured on the 106,272-instance corpus the
repair moved ``straightness`` on 8 rows (0.008%, max 0.0216) and
``deg2_straight_mean`` on none.

Invariance note: the canonical ordering sorts on coordinate cluster indices, so
the tie-break among *exactly* tied spanning trees is not rotation invariant.
Everything else in the module is.  On non-degenerate input, where no tie
exists, full rotation invariance is unaffected.

Cost note: the repair is not free on tie-heavy real data -- it roughly doubles
this group's share of extraction time on TSPLIB EUC_2D.  See
paper_tooling/support_arms_cost.py for the decomposition.

Degenerate input
----------------
See ``_DEGENERATE`` below for the exact sentinel returned when n < 3 or when the
MST carries no positive-length edge (e.g. all points identical).  The output is
always finite.
"""
from __future__ import annotations

import numpy as np

__all__ = ["feature_names", "compute", "compute_pair", "canonical_coords",
           "PAIR_NAMES", "SNAP_REL_TOL"]

#: The two features this group contributes to the deployed 34-feature arm.
#: ``compute_pair`` returns exactly these and is bit-identical to ``compute``
#: on them, at a fraction of the cost.
PAIR_NAMES = ("mst_topology_straightness", "mst_topology_deg2_straight_mean")

#: Per-axis clustering tolerance for the canonical point ordering, as a
#: fraction of the bounding-box extent.  It has to sit far enough above ULP
#: noise to absorb it and far enough below the real point spacing not to merge
#: distinct coordinate values.  At 1e-5 the margin to the tightest spacing in
#: this corpus (pla85900, ~3e-3 of the bounding box) is still two orders of
#: magnitude.  Measured with paper_tooling/support_arms_stability.py.
SNAP_REL_TOL = 1e-5


# Cosine threshold above which a degree-2 MST vertex counts as "locally
# collinear": the two incident edges leave it at more than ~154 degrees.
_COLLINEAR_COS = 0.9


_FEATURE_NAMES = [
    "mst_topology_deg2_frac",
    "mst_topology_branch_frac",
    "mst_topology_spine_frac",
    "mst_topology_caterpillar_index",
    "mst_topology_straightness",
    "mst_topology_hop_diameter_norm",
    "mst_topology_chain_count_norm",
    "mst_topology_chain_len_mean_norm",
    "mst_topology_chain_len_max_norm",
    "mst_topology_deg2_straight_mean",
    "mst_topology_adj_cos_mean",
    "mst_topology_collinear_vertex_frac",
]

# Sentinel for n < 3 and for an MST with no positive-length edge.
#
#   deg2_frac / branch_frac      0.0  no interior vertex exists
#   spine_frac                   1.0  every vertex trivially lies on the
#                                     longest path (a point or a single edge)
#   caterpillar_index            1.0  ditto
#   straightness                 1.0  a single edge (or a point mass) is
#                                     perfectly straight; also used when the
#                                     diameter path has zero length
#   hop_diameter_norm            1.0  the whole tree is its own diameter
#   chain_*                      0.0  no degree-2 chain exists
#   deg2_straight_mean           0.0  no turn angle is observable -> neutral,
#                                     the midpoint of the [-1, 1] range
#   adj_cos_mean                 0.0  no adjacent edge pair -> neutral
#   collinear_vertex_frac        0.0  no vertex is observed to be collinear
_DEGENERATE = {
    "mst_topology_deg2_frac": 0.0,
    "mst_topology_branch_frac": 0.0,
    "mst_topology_spine_frac": 1.0,
    "mst_topology_caterpillar_index": 1.0,
    "mst_topology_straightness": 1.0,
    "mst_topology_hop_diameter_norm": 1.0,
    "mst_topology_chain_count_norm": 0.0,
    "mst_topology_chain_len_mean_norm": 0.0,
    "mst_topology_chain_len_max_norm": 0.0,
    "mst_topology_deg2_straight_mean": 0.0,
    "mst_topology_adj_cos_mean": 0.0,
    "mst_topology_collinear_vertex_frac": 0.0,
}


def feature_names() -> list:
    """Stable, ordered names of the features this module produces."""
    return list(_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


def canonical_coords(coords: np.ndarray, rel_tol: float | None = None):
    """Reorder a point cloud into a canonical, jitter-insensitive frame.

    Returns ``(canon, order)``: ``order`` is the permutation applied and
    ``canon`` is the ORIGINAL coordinates in that order.  Lengths and
    directions are always measured on ``canon``, so on generic input every
    feature value is bit-identical to the unfixed module.

    The ordering key is a per-axis CLUSTER index, not a rounded coordinate.
    Rounding to a fixed grid has boundary discontinuities: a coordinate landing
    near a half-step flips under arbitrarily small jitter, and a randomised
    rotated-lattice battery put that flip rate at 35% even with a dyadic step.
    Clustering has no fixed boundary -- it adapts to where the values actually
    are -- so a jitter far below the inter-cluster gap provably cannot change
    the key.  Points sharing a lattice column get the *same* integer, which is
    what keeps their ordering from being decided by ULP noise.
    """
    if rel_tol is None:
        rel_tol = SNAP_REL_TOL          # read at call time, not at def time
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 2:
        c = c.reshape(len(c), -1)
    if c.shape[0] == 0:
        return c, np.zeros(0, dtype=np.int64)

    extent = np.ptp(c, axis=0)
    scale = float(extent.max()) if extent.size else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    tol = scale * float(rel_tol)

    keys = np.empty(c.shape, dtype=np.int64)
    for j in range(c.shape[1]):
        keys[:, j] = _cluster_index(c[:, j], tol)
    # Lexicographic on rows: column 0 primary.  np.lexsort takes its keys with
    # the LAST one primary, hence the reversal.
    order = np.lexsort(keys.T[::-1])
    return c[order], order


def _cluster_index(x: np.ndarray, tol: float) -> np.ndarray:
    """Single-linkage cluster index of a 1-D array at absolute tolerance ``tol``.

    Values are grouped into maximal runs whose consecutive sorted gaps are all
    <= tol, and each value is replaced by its group's rank.  Exact, integral,
    and free of the boundary discontinuity a fixed rounding grid has.
    """
    o = np.argsort(x, kind="stable")
    xs = x[o]
    if xs.size <= 1:
        return np.zeros(x.shape, dtype=np.int64)
    grp = np.empty(xs.size, dtype=np.int64)
    grp[0] = 0
    np.cumsum(np.diff(xs) > tol, out=grp[1:])
    out = np.empty(x.shape, dtype=np.int64)
    out[o] = grp
    return out


# ---------------------------------------------------------------------------
# MST access
# ---------------------------------------------------------------------------


def _edges_from_mst(mst_csr, n: int):
    """Extract undirected endpoint arrays (u, v) from whatever the caller gave.

    Accepts a scipy sparse MST (any format), an ``mst_utils.MSTResult``, or a
    plain (m, 2) integer array of endpoints.  Weights are ignored: edge lengths
    are recomputed from ``coords`` so that the tree metric is exactly
    consistent with the direction vectors, in float64, and independent of the
    float32 the MST backends use internally.
    """
    endpoints = getattr(mst_csr, "endpoints", None)
    if endpoints is not None:  # mst_utils.MSTResult
        ep = np.asarray(endpoints, dtype=np.int64)
        return ep[:, 0], ep[:, 1]

    tocoo = getattr(mst_csr, "tocoo", None)
    if tocoo is not None:  # scipy sparse
        coo = tocoo()
        return coo.row.astype(np.int64), coo.col.astype(np.int64)

    arr = np.asarray(mst_csr)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64)
    raise TypeError("mst_csr must be a scipy sparse matrix, an MSTResult, or (m, 2) endpoints")


def _compute_mst_edges(coords: np.ndarray):
    """Compute an MST when the caller supplied none.

    Prefers the repo's shared ``mst_utils.compute_mst`` (memory-aware, exact,
    cache-aware) and falls back to a dense scipy MST if that import is not
    available, so the module also works standalone.
    """
    try:
        from mst_utils import compute_mst  # type: ignore

        res = compute_mst(coords)
        ep = np.asarray(res.endpoints, dtype=np.int64)
        return ep[:, 0], ep[:, 1]
    except Exception:
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial.distance import cdist

        dm = cdist(coords, coords)
        coo = minimum_spanning_tree(dm).tocoo()
        return coo.row.astype(np.int64), coo.col.astype(np.int64)


# ---------------------------------------------------------------------------
# Canonical MST on degenerate tie structure
# ---------------------------------------------------------------------------

#: Two candidate edges count as tied when their lengths differ by less than
#: this fraction of the longest MST edge.  Far above jitter, far below any real
#: length difference.
TIE_REL_TOL = 1e-7

#: An instance is treated as tie-degenerate when distinct MST edge lengths
#: number fewer than this fraction of the edges.  A generic cloud has almost
#: all lengths distinct and skips the repair entirely, at zero cost.
DEGENERACY_FRAC = 0.5

#: Cap on the candidate edge set.  Beyond this the ball query is the dominant
#: cost and the instance is not lattice-like anyway, so the repair is skipped.
MAX_CANDIDATE_EDGES = 400_000


def _canonical_mst(coords, u, v, w):
    """Recompute the MST with a canonical tie-break, if ties actually exist.

    Returns ``(u, v, repaired)``.

    A Euclidean MST is unique when all pairwise distances are distinct; on a
    lattice they are massively not, and which of the tied spanning trees a
    backend returns is decided by floating-point noise.  Every alternative MST
    edge is, by the cut property, no longer than the longest edge of any MST,
    so a ball query at that radius yields a candidate set that provably
    contains every edge of every MST.  Kruskal over that set with lengths
    bucketed at ``TIE_REL_TOL`` and ties broken on the canonical vertex pair
    then selects one specific MST as a function of the point set alone.

    Bucketing the lengths -- rather than comparing them raw -- is what makes
    this jitter-proof: a perturbation far below the gap between distinct length
    buckets cannot move an edge into a different bucket, so the sort order, and
    hence the tree, is unchanged.
    """
    m = u.size
    if m < 2:
        return u, v, False

    ws = np.sort(w)
    wmax = float(ws[-1])
    tol = TIE_REL_TOL * wmax
    n_distinct = 1 + int(np.count_nonzero(np.diff(ws) > tol))
    if n_distinct > DEGENERACY_FRAC * m:
        return u, v, False                      # generic: MST already unique

    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    # The radius must clear the tied edges by a real margin, not a hair. On a
    # lattice EVERY edge sits exactly at wmax, so a radius of wmax puts the
    # entire candidate mass on the query boundary and jitter then decides which
    # half is returned -- reintroducing the instability through the back door.
    # Ten tie-tolerances of slack is far above the jitter and still far below
    # the next distinct length bucket, so the extra edges sort strictly after
    # the tied ones and Kruskal never reaches them.
    pairs = tree.query_pairs(r=wmax * (1.0 + 10.0 * TIE_REL_TOL),
                             output_type="ndarray")
    if pairs.size == 0 or len(pairs) > MAX_CANDIDATE_EDGES:
        return u, v, False

    pu = pairs[:, 0].astype(np.int64)
    pv = pairs[:, 1].astype(np.int64)
    pw = np.linalg.norm(coords[pu] - coords[pv], axis=1)
    keep = pw > 0.0
    pu, pv, pw = pu[keep], pv[keep], pw[keep]
    if pu.size == 0:
        return u, v, False

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    n = coords.shape[0]
    bucket = _cluster_index(pw, tol)
    order = np.lexsort((pv, pu, bucket))        # length bucket, then vertex pair

    # Kruskal on ANY strictly increasing transform of the edge weights returns
    # the same tree, so replacing the weights by their rank in the desired
    # ordering implements the lexicographic tie-break exactly -- and lets scipy
    # run the union-find in C. A Python loop here cost more than the whole rest
    # of the extractor on the tie-heavy instances that need it (the repair
    # fires on 49 of the 78 real TSPLIB EUC_2D instances, so this path is the
    # common case, not the exception). Ranks start at 1: a zero entry in a
    # sparse matrix means "no edge".
    rank = np.empty(order.size, dtype=np.float64)
    rank[order] = np.arange(1, order.size + 1, dtype=np.float64)
    g = coo_matrix((rank, (pu, pv)), shape=(n, n))
    sel = minimum_spanning_tree(g).tocoo()
    out_u = sel.row.astype(np.int64)
    out_v = sel.col.astype(np.int64)
    k = out_u.size
    total = float(np.linalg.norm(coords[out_u] - coords[out_v], axis=1).sum())

    # Accept only a tree that is minimal UP TO THE TIE TOLERANCE. Demanding
    # exact minimality would reject the canonical tree on jittered input by
    # construction: bucketing deliberately treats sub-tolerance length
    # differences as ties, so the canonical pick can be heavier than the true
    # optimum by at most one tolerance per edge. A larger excess than that
    # means the candidate graph was genuinely incomplete, and a
    # canonical-but-worse tree is not a trade worth making.
    backend_total = float(w.sum())
    allowed = m * tol + 1e-9 * max(backend_total, 1.0)
    if k != m or (total - backend_total) > allowed:
        return u, v, False
    return out_u, out_v, True


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def _adjacency(n: int, u: np.ndarray, v: np.ndarray, w: np.ndarray):
    """CSR-style undirected adjacency: (indptr, neighbour, weight).

    Neighbour lists are sorted by (source, destination) rather than by source
    alone, so the DFS order below depends only on the canonical vertex
    numbering and not on the order the MST backend happened to emit its edges.
    """
    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])
    wgt = np.concatenate([w, w])
    order = np.lexsort((dst, src))
    deg = np.bincount(src, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    return indptr, dst[order], wgt[order], deg


def _sweep(start, indptr, nbr, wts, n):
    """Single DFS from ``start`` over one tree component.

    ``wts`` is a per-directed-edge weight list, or ``None`` for unit (hop)
    weights.  Returns (dist, parent, order) where ``order`` lists the visited
    vertices.  Plain Python lists are used because this is a scalar-indexed
    loop, where numpy element access is several times slower.
    """
    dist = [-1.0] * n
    parent = [-1] * n
    dist[start] = 0.0
    order = [start]
    stack = [start]
    while stack:
        x = stack.pop()
        dx = dist[x]
        for k in range(indptr[x], indptr[x + 1]):
            y = nbr[k]
            if dist[y] < 0.0:
                dist[y] = dx + (1.0 if wts is None else wts[k])
                parent[y] = x
                order.append(y)
                stack.append(y)
    return dist, parent, order


def _farthest(order, dist, eps=0.0):
    """Farthest visited vertex, ties broken on the lowest canonical index.

    Two passes, deliberately. A single running-best pass with a tolerance is
    not a total order -- "within eps of the current best" depends on the order
    of traversal -- so it can return different answers for inputs that differ
    by less than eps. Taking the maximum first and then the lowest index within
    eps of it is a total rule.

    ``eps`` absorbs exact ties that a perturbation has turned into
    sub-tolerance differences. On a lattice the candidate path lengths are
    integer multiples of the spacing, so the genuine gaps are enormous next to
    eps and nothing real is merged. Without this, a 1e-9 jitter silently
    re-elects the diameter endpoint even when the tree itself is stable, and
    the whole straightness value moves with it.
    """
    bd = dist[order[0]]
    for x in order:
        dx = dist[x]
        if dx > bd:
            bd = dx
    thresh = bd - eps
    best = -1
    for x in order:
        if dist[x] >= thresh and (best < 0 or x < best):
            best = x
    return best, dist[best]


def _forest_diameter(indptr, nbr, wts, n, roots, eps=0.0):
    """Double-sweep diameter over every component; exact on trees/forests.

    Returns (length, path_vertices) for the single longest path in the forest.
    Components whose diameters tie within ``eps`` are resolved by the lower
    root, which is canonical because the roots are canonical.
    """
    best_len = -1.0
    best_path = []
    for s in roots:
        d1, _, order1 = _sweep(s, indptr, nbr, wts, n)
        a, _ = _farthest(order1, d1, eps)
        d2, par2, order2 = _sweep(a, indptr, nbr, wts, n)
        b, blen = _farthest(order2, d2, eps)
        if blen > best_len + eps:
            best_len = blen
            path = []
            x = b
            while x != -1:
                path.append(x)
                x = par2[x]
            best_path = path
    return best_len, best_path


def _component_roots(indptr, nbr, n):
    """One representative vertex per connected component, in ascending order."""
    seen = np.zeros(n, dtype=bool)
    roots = []
    for s in range(n):
        if seen[s]:
            continue
        roots.append(s)
        _, _, order = _sweep(s, indptr, nbr, None, n)
        seen[order] = True
    return roots


# ---------------------------------------------------------------------------
# Direction alignment
# ---------------------------------------------------------------------------


def _alignment_stats(coords, indptr, nbr, deg, n):
    """Cosines between MST edges that share a vertex.

    For each vertex, the incident edges are turned into unit vectors pointing
    *away* from it.  A vertex sitting mid-chain on a straight run has two
    antiparallel outgoing vectors (cos = -1).

    Returns
    -------
    deg2_straight_mean : mean of ``-cos`` over degree-2 vertices, in [-1, 1];
        +1 means every chain vertex is perfectly straight-through.
    adj_cos_mean : mean of ``|cos|`` over every pair of edges sharing a vertex.
    collinear_frac : fraction of *all* n vertices that are degree-2 with
        ``-cos >= 0.9`` (locally one-dimensional).

    Zero-length edges have no direction; pairs involving one are excluded.  If
    no valid pair exists, the neutral sentinel 0.0 is returned.
    """
    # Directed edge vectors in the same order as the adjacency arrays.
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    vec = coords[nbr] - coords[src]
    norm = np.linalg.norm(vec, axis=1)
    valid = norm > 0.0
    unit = np.zeros_like(vec)
    np.divide(vec, norm[:, None], out=unit, where=valid[:, None])

    abs_cos_sum = 0.0
    abs_cos_cnt = 0
    straight_sum = 0.0
    straight_cnt = 0
    collinear_cnt = 0

    max_deg = int(deg.max()) if n else 0
    for k in range(2, max_deg + 1):
        vs = np.flatnonzero(deg == k)
        if vs.size == 0:
            continue
        # Each such vertex owns exactly k contiguous slots at indptr[v].
        slots = indptr[vs][:, None] + np.arange(k)[None, :]
        u = unit[slots]                      # (m, k, d)
        ok = valid[slots]                    # (m, k)
        cos = np.einsum("mid,mjd->mij", u, u)
        pair_ok = ok[:, :, None] & ok[:, None, :]
        iu, ju = np.triu_indices(k, 1)
        c = cos[:, iu, ju]
        p = pair_ok[:, iu, ju]
        if p.any():
            abs_cos_sum += float(np.abs(c[p]).sum())
            abs_cos_cnt += int(p.sum())
        if k == 2:
            s = -c[:, 0]
            m = p[:, 0]
            if m.any():
                straight_sum += float(s[m].sum())
                straight_cnt += int(m.sum())
                collinear_cnt += int((s[m] >= _COLLINEAR_COS).sum())

    deg2_straight_mean = straight_sum / straight_cnt if straight_cnt else 0.0
    adj_cos_mean = abs_cos_sum / abs_cos_cnt if abs_cos_cnt else 0.0
    collinear_frac = collinear_cnt / n if n else 0.0
    return deg2_straight_mean, adj_cos_mean, collinear_frac


# ---------------------------------------------------------------------------
# Degree-2 chains
# ---------------------------------------------------------------------------


def _chain_stats(u, v, deg, n):
    """Sizes of the maximal connected runs of degree-2 vertices.

    A "chain" is a connected component of the subgraph induced on the degree-2
    vertices.  A pure path has one chain of n - 2 vertices; a bushy tree has
    many chains of size 1; a union of s segments has roughly s long chains.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    idx = np.flatnonzero(deg == 2)
    m = idx.size
    if m == 0:
        return 0, 0.0, 0.0
    remap = np.full(n, -1, dtype=np.int64)
    remap[idx] = np.arange(m)
    keep = (deg[u] == 2) & (deg[v] == 2)
    ru = remap[u[keep]]
    rv = remap[v[keep]]
    data = np.ones(ru.size, dtype=np.int8)
    sub = csr_matrix((data, (ru, rv)), shape=(m, m))
    ncomp, labels = connected_components(sub, directed=False)
    sizes = np.bincount(labels, minlength=ncomp)
    return int(ncomp), float(sizes.mean()), float(sizes.max())


def _deg2_straight_mean(coords, indptr, nbr, deg) -> float:
    """``deg2_straight_mean`` alone, skipping the higher-degree cosine pairs.

    Bit-identical to the value ``_alignment_stats`` returns: that function's
    loop over degrees 3..max only feeds ``adj_cos_mean``, which the deployed
    pair does not use.
    """
    vs = np.flatnonzero(deg == 2)
    if vs.size == 0:
        return 0.0
    slots = indptr[vs][:, None] + np.arange(2)[None, :]
    src = np.repeat(np.arange(len(deg), dtype=np.int64), np.diff(indptr))
    vec = coords[nbr[slots]] - coords[src[slots]]
    norm = np.linalg.norm(vec, axis=2)
    ok = norm > 0.0
    unit = np.zeros_like(vec)
    np.divide(vec, norm[:, :, None], out=unit, where=ok[:, :, None])
    cos = np.einsum("md,md->m", unit[:, 0, :], unit[:, 1, :])
    m = ok[:, 0] & ok[:, 1]
    if not m.any():
        return 0.0
    return float((-cos[m]).sum() / int(m.sum()))


def compute_pair(coords: np.ndarray, mst_csr=None,
                 assume_canonical: bool = False) -> dict:
    """Just the two deployed features, at a fraction of the cost of ``compute``.

    ``compute`` builds the full 12-feature group: a second (hop-weighted)
    forest diameter, degree-2 chain components via a sparse connected-components
    call, and the all-degree cosine loop. None of that feeds
    ``straightness`` or ``deg2_straight_mean``, so the deployed arm should not
    pay for it. Values are asserted bit-identical to ``compute`` in
    paper_tooling/support_arms_verify.py.
    """
    st = _prepare(coords, mst_csr, assume_canonical)
    if st is None:
        return {k: _DEGENERATE[k] for k in PAIR_NAMES}
    coords, u, v, w, indptr, nbr, wts, deg, n = st

    path_eps = TIE_REL_TOL * float(w.sum())
    roots = _component_roots(indptr.tolist(), nbr.tolist(), n)
    arc_len, path = _forest_diameter(indptr.tolist(), nbr.tolist(), wts.tolist(),
                                     n, roots, path_eps)
    if arc_len > 0.0 and len(path) >= 2:
        gap = float(np.linalg.norm(coords[path[0]] - coords[path[-1]]))
        straightness = gap / arc_len
    else:
        straightness = 1.0
    out = {
        "mst_topology_straightness": float(min(max(straightness, 0.0), 1.0)),
        "mst_topology_deg2_straight_mean":
            float(_deg2_straight_mean(coords, indptr, nbr, deg)),
    }
    for k, val in out.items():
        if not np.isfinite(val):
            out[k] = _DEGENERATE[k]
    return out


def _prepare(coords, mst_csr, assume_canonical):
    """Shared front half of compute/compute_pair. None when degenerate."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        coords = coords.reshape(len(coords), -1)
    n = coords.shape[0]
    if n < 3:
        return None

    if assume_canonical:
        if mst_csr is None:
            raise ValueError("assume_canonical=True requires an mst_csr")
        u, v = _edges_from_mst(mst_csr, n)
    else:
        coords, _order = canonical_coords(coords)
        u, v = _compute_mst_edges(coords)

    if u.size:
        w = np.linalg.norm(coords[u] - coords[v], axis=1)
        keep = w > 0.0
        u, v, w = u[keep], v[keep], w[keep]
    else:
        w = np.zeros(0)
    if u.size == 0:
        return None

    u, v, repaired = _canonical_mst(coords, u, v, w)
    if repaired:
        w = np.linalg.norm(coords[u] - coords[v], axis=1)
        keep = w > 0.0
        u, v, w = u[keep], v[keep], w[keep]
        if u.size == 0:
            return None

    indptr, nbr, wts, deg = _adjacency(n, u, v, w)
    return coords, u, v, w, indptr, nbr, wts, deg, n


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute(coords: np.ndarray, mst_csr=None, assume_canonical: bool = False) -> dict:
    """Compute the MST-topology feature group.

    Parameters
    ----------
    coords : (n, d) float array
        Point coordinates in whatever frame the caller supplies.  Every feature
        here is invariant to translation, rotation, reflection and uniform
        scaling, so the frame does not matter.
    mst_csr : optional
        A scipy sparse Euclidean MST, an ``mst_utils.MSTResult``, or an (m, 2)
        array of endpoints.

        IGNORED unless ``assume_canonical`` is True.  A tree built on
        non-canonical coordinates is exactly the defect this module was fixed
        for, so silently trusting one would reintroduce it.
    assume_canonical : bool
        Set only when ``coords`` already came out of ``canonical_coords`` and
        ``mst_csr`` was built on the matching snapped array.  This is the cheap
        path: it lets a caller share one MST across feature groups instead of
        paying for a second one here.

    Returns
    -------
    dict
        One finite float per name in ``feature_names()``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        coords = coords.reshape(len(coords), -1)
    n = coords.shape[0]

    if n < 3:
        return dict(_DEGENERATE)

    if assume_canonical:
        if mst_csr is None:
            raise ValueError("assume_canonical=True requires an mst_csr")
        u, v = _edges_from_mst(mst_csr, n)
    else:
        # Reorder into the canonical frame, then build the tree.  Lengths and
        # directions below still read the original coordinates, merely
        # reordered, so generic instances keep bit-identical feature values.
        coords, _order = canonical_coords(coords)
        u, v = _compute_mst_edges(coords)

    if u.size:
        w = np.linalg.norm(coords[u] - coords[v], axis=1)
        keep = w > 0.0
        # Zero-length edges (duplicate points) carry no direction and no metric
        # information; scipy's MST already drops them. Dropping them here too
        # keeps the two paths through this function consistent.
        u, v, w = u[keep], v[keep], w[keep]
    else:
        w = np.zeros(0)

    if u.size == 0:
        return dict(_DEGENERATE)

    # Replace a tie-degenerate backend tree with the canonical one. A no-op,
    # and free, whenever the MST is already unique.
    u, v, repaired = _canonical_mst(coords, u, v, w)
    if repaired:
        w = np.linalg.norm(coords[u] - coords[v], axis=1)
        keep = w > 0.0
        u, v, w = u[keep], v[keep], w[keep]
        if u.size == 0:
            return dict(_DEGENERATE)

    indptr, nbr, wts, deg = _adjacency(n, u, v, w)

    # --- degree composition -------------------------------------------------
    deg2_frac = float(np.count_nonzero(deg == 2)) / n
    branch_frac = float(np.count_nonzero(deg >= 3)) / n

    # --- longest path (weighted diameter), straightness, caterpillar --------
    indptr_l = indptr.tolist()
    nbr_l = nbr.tolist()
    wts_l = wts.tolist()
    roots = _component_roots(indptr_l, nbr_l, n)

    # Any path length is bounded by the tree's total weight, so this scales the
    # tie tolerance to the quantity being compared.
    path_eps = TIE_REL_TOL * float(w.sum())
    arc_len, path = _forest_diameter(indptr_l, nbr_l, wts_l, n, roots, path_eps)
    spine_frac = len(path) / n
    if arc_len > 0.0 and len(path) >= 2:
        gap = float(np.linalg.norm(coords[path[0]] - coords[path[-1]]))
        straightness = gap / arc_len
    else:
        straightness = 1.0
    straightness = float(min(max(straightness, 0.0), 1.0))

    on_path = np.zeros(n, dtype=bool)
    on_path[np.asarray(path, dtype=np.int64)] = True
    near = on_path.copy()
    near[u[on_path[v]]] = True
    near[v[on_path[u]]] = True
    caterpillar_index = float(np.count_nonzero(near)) / n

    hop_diam, _ = _forest_diameter(indptr_l, nbr_l, None, n, roots)
    hop_diameter_norm = float(hop_diam) / (n - 1)

    # --- degree-2 chains ----------------------------------------------------
    n_chains, chain_mean, chain_max = _chain_stats(u, v, deg, n)
    chain_count_norm = n_chains / n
    chain_len_mean_norm = chain_mean / n
    chain_len_max_norm = chain_max / n

    # --- edge-direction alignment ------------------------------------------
    deg2_straight_mean, adj_cos_mean, collinear_frac = _alignment_stats(
        coords, indptr, nbr, deg, n
    )

    out = {
        "mst_topology_deg2_frac": float(deg2_frac),
        "mst_topology_branch_frac": float(branch_frac),
        "mst_topology_spine_frac": float(spine_frac),
        "mst_topology_caterpillar_index": float(caterpillar_index),
        "mst_topology_straightness": float(straightness),
        "mst_topology_hop_diameter_norm": float(hop_diameter_norm),
        "mst_topology_chain_count_norm": float(chain_count_norm),
        "mst_topology_chain_len_mean_norm": float(chain_len_mean_norm),
        "mst_topology_chain_len_max_norm": float(chain_len_max_norm),
        "mst_topology_deg2_straight_mean": float(deg2_straight_mean),
        "mst_topology_adj_cos_mean": float(adj_cos_mean),
        "mst_topology_collinear_vertex_frac": float(collinear_frac),
    }
    # Contract: never emit NaN/inf.
    for k, val in out.items():
        if not np.isfinite(val):
            out[k] = _DEGENERATE[k]
    return out
