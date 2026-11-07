"""Received ligand-receptor signal per cell (CytoSignal-style, in Python).

For a cell i and pair (L, R):  score_i = receptor_i x ligand around i
  receptor_i : receptor expression averaged over i and its Delaunay neighbours
  ligand     : secreted / ECM pairs -> Gaussian-weighted ligand expression of the other cells (sigma in cell spacings)
               contact pairs        -> mean ligand expression of the Delaunay neighbours
Multi-subunit ligands / receptors use the geometric mean of the subunits (CellChat convention).
Built-in tables (data/lr_mouse.csv, data/lr_human.csv): CellChatDB v1 + CellPhoneDB protein ligand-receptor and adhesion pairs."""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from ._utils import delaunay_adj, gauss_w, dense

_DATA = os.path.join(os.path.dirname(__file__), "data")


def load_lr(lr="mouse"):
    """'mouse' / 'human' (CellChat + CellPhoneDB) or a DataFrame with columns ligand, receptor[, pathway, contact]."""
    if isinstance(lr, pd.DataFrame):
        d = lr.copy()
        if "contact" not in d:
            d["contact"] = False
        if "pathway" not in d:
            d["pathway"] = d["ligand"].str.split("_").str[0].str.upper()
        return d
    if lr in ("mouse", "human"):
        return pd.read_csv(os.path.join(_DATA, f"lr_{lr}.csv"))
    raise ValueError("species must be 'mouse' or 'human', or pass lr=DataFrame with columns ligand, receptor[, pathway, contact]")


def _panel(lr, genes):
    g = set(genes)
    ok = [all(x in g for x in str(l).split("_") + str(r).split("_")) for l, r in zip(lr.ligand, lr.receptor)]
    return lr[np.array(ok, bool)].reset_index(drop=True)


def received(L, genes, loc, lr, sigma=2.0):
    """L: cells x genes log-normalised (sparse ok). Returns (cells x pairs array, pair table)."""
    lr = _panel(lr, genes)
    gi = {g: k for k, g in enumerate(genes)}
    Rimp = delaunay_adj(loc, self_loop=True)
    Wc = delaunay_adj(loc, self_loop=False)
    Wd = gauss_w(loc, sigma)
    L = sp.csc_matrix(L)
    col = lambda g: dense(L[:, gi[g]]).ravel()

    def gmean(cols):
        M = np.stack(cols, 1)
        return np.exp(np.log(np.maximum(M, 1e-12)).mean(1)) * (M.min(1) > 0)

    cacheR, cacheL = {}, {}
    S = np.zeros((L.shape[0], len(lr)), np.float32)
    for k, (l, r, c) in enumerate(zip(lr.ligand, lr.receptor, lr.contact)):
        if r not in cacheR:
            cacheR[r] = gmean([Rimp @ col(s) for s in r.split("_")])
        key = (l, bool(c))
        if key not in cacheL:
            W = Wc if c else Wd
            cacheL[key] = gmean([W @ col(s) for s in l.split("_")])
        S[:, k] = cacheR[r] * cacheL[key]
    return S, lr
