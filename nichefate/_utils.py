import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree, Delaunay
from scipy.stats import rankdata, norm
from sklearn.neighbors import NearestNeighbors


def dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def lognorm(X, scale=1e4):
    """log1p of counts scaled to `scale` per cell (sparse in, sparse out)."""
    X = sp.csr_matrix(X, dtype=np.float32)
    tot = np.asarray(X.sum(1)).ravel()
    X = sp.diags(scale / np.maximum(tot, 1)) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def unit_coords(xy):
    """Coordinates in units of the median nearest-neighbour spacing."""
    xy = np.asarray(xy, float)
    d = cKDTree(xy).query(xy, 2)[0][:, 1]
    return xy / np.median(d)


def knn_adj(loc, k=10):
    """Row-normalised symmetric k-NN graph with self-loops."""
    idx = NearestNeighbors(n_neighbors=k + 1).fit(loc).kneighbors(loc, return_distance=False)
    n = len(loc)
    A = sp.csr_matrix((np.ones(idx.size), (np.repeat(np.arange(n), k + 1), idx.ravel())), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64)
    return sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A


def delaunay_adj(loc, self_loop=True):
    """Row-normalised Delaunay neighbour graph (direct contacts)."""
    jit = loc + np.random.default_rng(0).normal(0, 1e-3, loc.shape)
    ind, nbr = Delaunay(jit).vertex_neighbor_vertices
    rows = np.repeat(np.arange(len(loc)), np.diff(ind))
    A = sp.csr_matrix((np.ones(len(nbr)), (rows, nbr)), shape=(len(loc),) * 2)
    if self_loop:
        A = A + sp.eye(len(loc))
    return sp.diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A


def gauss_w(loc, sigma):
    """Gaussian weights to other cells within 3 sigma (no self)."""
    pairs = cKDTree(loc).query_pairs(3 * sigma, output_type="ndarray")
    d = np.linalg.norm(loc[pairs[:, 0]] - loc[pairs[:, 1]], axis=1)
    w = np.exp(-d ** 2 / (2 * sigma ** 2))
    n = len(loc)
    return sp.coo_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])), shape=(n, n)).tocsr()


def radius_mean(loc, M, r):
    """Mean of rows of M over all cells within radius r (incl. self)."""
    G = sp.csr_matrix(NearestNeighbors(radius=r).fit(loc).radius_neighbors_graph(loc, mode="connectivity"))
    G = G + sp.eye(len(loc)); G.data[:] = 1
    return sp.diags(1 / np.asarray(G.sum(1)).ravel()) @ M


def median_sqd(A, B, k=300_000, seed=0):
    r = np.random.default_rng(seed)
    i, j = r.integers(0, len(A), k), r.integers(0, len(B), k)
    return float(np.median(((A[i] - B[j]) ** 2).sum(1)))


def rank_normal(x):
    return norm.ppf((rankdata(x) - 0.5) / len(x))


def resid(Y, C):
    b, *_ = np.linalg.lstsq(C, Y, rcond=None)
    return Y - C @ b


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out
