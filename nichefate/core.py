"""The four inference steps: compute_niche -> match -> fate -> drivers."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, norm

from . import _utils as U
from ._lr import load_lr, received, _panel
from ._gnn import gcn_embed
from ._ot import solve


# ----------------------------------------------------------------------------------------------------------- 1. niche
def compute_niche(adatas, species="mouse", lr=None, level="pathway", sigma=2.0, smooth=5.0, n_pcs=30, n_neighbors=10,
                  spatial_key="spatial", layer=None, signals=None, target_sum=1e4, n_sub=200_000, device=None, seed=0, verbose=True):
    """Describe every cell's niche, jointly for all sections (one AnnData per section / time point).

    Adds to each AnnData
      layers['nf_lognorm']  log-normalised expression
      obsm['nf_loc']        coordinates in units of the median cell spacing
      obsm['nf_signal']     received ligand-receptor signal (per pathway or per pair), smoothed over `smooth` spacings
      obsm['nf_niche']      graph-convolutional embedding of the cell's neighbourhood
      uns['nf_signal_names']
    species: 'mouse' or 'human' (built-in CellChat + CellPhoneDB ligand-receptor pairs).
    lr: optional custom DataFrame with columns ligand, receptor[, pathway, contact] (subunits joined by '_'); overrides species.
    level: 'pathway', 'receptor' (pairs sharing a receptor complex pooled) or 'pair'. sigma: ligand diffusion range, in cell spacings.
    signals: optional precomputed received signals (e.g. from CytoSignal), one DataFrame per section (index = obs_names,
    columns = signal names); used as obsm['nf_signal'] instead of the built-in scores.
    target_sum: counts per cell before log1p (normalisation of expression for the state PCA and the neighbourhood embedding).
    n_sub: cells sampled to fit the PCA that feeds the neighbourhood embedding."""
    adatas = list(adatas)
    genes = list(adatas[0].var_names)
    for ad in adatas[1:]:
        s = set(ad.var_names); genes = [g for g in genes if g in s]
    table = _panel(load_lr(lr if lr is not None else species), genes)
    if level == "pathway":
        key = table.pathway.astype(str).values
    elif level == "receptor":
        key = table.receptor.astype(str).values
    elif level == "pair":
        key = np.array([f"{l}-{r}" for l, r in zip(table.ligand, table.receptor)])
    else:
        raise ValueError("level must be 'pathway', 'receptor' or 'pair'")
    names = list(dict.fromkeys(key)) if level == "pair" else sorted(set(key))
    if len(table) == 0:
        raise ValueError(f"no ligand-receptor pair has all its genes in the panel for species={species!r}; check gene names "
                         "('Apoe' for mouse, 'APOE' for human) or pass lr=DataFrame")
    if verbose:
        print(f"{len(genes)} shared genes, {len(table)} ligand-receptor pairs, {len(names)} signals ({level})")
    Ls, Zin, adj = [], [], []
    for k, ad in enumerate(adatas):
        X = ad[:, genes].layers[layer] if layer else ad[:, genes].X
        L = U.lognorm(X, target_sum); loc = U.unit_coords(ad.obsm[spatial_key])
        if signals is None:
            S, _ = received(L, genes, loc, table, sigma)
            S = np.log1p(np.maximum(S, 0))
            if level != "pair":
                S = np.stack([S[:, key == p].mean(1) for p in names], 1)
            S = np.asarray(U.radius_mean(loc, S, smooth), np.float32); nm = names
        else:
            ext = signals[k].reindex(ad.obs_names); S = np.nan_to_num(ext.values.astype(np.float32)); nm = list(ext.columns.astype(str))
        ad.layers["nf_lognorm"] = U.lognorm(ad.layers[layer] if layer else ad.X, target_sum)
        ad.obsm["nf_loc"] = loc.astype(np.float32)
        ad.obsm["nf_signal"] = S
        ad.uns["nf_signal_names"] = np.array(nm)
        ad.uns["nf_lr"] = {"ligand": table.ligand.astype(str).values, "receptor": table.receptor.astype(str).values, "signal": key.astype(str)}
        Ls.append(L); adj.append(U.knn_adj(loc, n_neighbors))
    # joint PCA of all cells (shared genes) -> input of the neighbourhood embedding
    Lall = sp.vstack(Ls).tocsr(); mu = np.asarray(Lall.mean(0)).ravel()
    rs = np.random.default_rng(seed).choice(Lall.shape[0], min(n_sub, Lall.shape[0]), replace=False)
    pca = PCA(n_pcs, random_state=seed).fit(U.dense(Lall[rs]) - mu)
    Zin = [pca.transform(U.dense(L) - mu).astype(np.float32) for L in Ls]
    E = gcn_embed(Zin, adj, device=device, seed=seed)
    for ad, e in zip(adatas, E):
        ad.obsm["nf_niche"] = e
    if verbose:
        print("niche computed for", len(adatas), "sections:", [ad.n_obs for ad in adatas], "cells")


# ----------------------------------------------------------------------------------------------------------- 2. match
@dataclass
class NicheFateResult:
    adata1: object
    adata2: object
    groupby: str
    idx1: np.ndarray            # source cells (positions in adata1)
    idx2: np.ndarray            # target cells (positions in adata2)
    groups1: np.ndarray
    groups2: np.ndarray         # target groups (columns of cell_mass)
    cell_mass: np.ndarray       # source cell x target group: transported mass
    Z1: np.ndarray              # state PCA of the source cells (used as covariates by `drivers`)
    params: dict = field(default_factory=dict)
    adjusted: object = None     # set by `drivers`

    @property
    def transitions(self):
        """Source group x target group, rows sum to 1."""
        lab = self.adata1.obs[self.groupby].astype(str).values[self.idx1]
        u1 = np.unique(lab)
        T = np.stack([self.cell_mass[lab == g].sum(0) for g in u1])
        return pd.DataFrame(T / np.maximum(T.sum(1, keepdims=True), 1e-30), index=u1, columns=self.groups2)

    def __repr__(self):
        return (f"NicheFateResult: {len(self.idx1)} source cells -> {len(self.idx2)} target cells, "
                f"target groups {[str(g) for g in self.groups2]}")


def _select(labels, cells):
    """Boolean mask of labels in `cells`; a name that is not an exact label selects all labels starting with it."""
    labels = np.asarray(labels).astype(str); m = np.zeros(len(labels), bool)
    for c in np.atleast_1d(cells).astype(str):
        hit = labels == c
        m |= hit if hit.any() else np.char.startswith(labels, c)
    return m


def _mask(ad, groupby, cells):
    if cells is None:
        return np.arange(ad.n_obs)
    cells = np.asarray(cells)
    if cells.dtype == bool:
        return np.where(cells)[0]
    return np.where(_select(ad.obs[groupby].values, cells))[0]


def match(adata1, adata2, groupby, cells1=None, cells2=None, w_signal=0.5, w_niche=0.25, alpha=0.0,
          eps=0.005, tau=0.99, n_pcs=30, pca_sections=None, pca_cells=None, scale_signals=True, device=None, seed=0):
    """Match cells of an earlier section (adata1) to a later section (adata2).

    Cost between cells = own expression state + w_signal x received signals + w_niche x neighbourhood embedding
    (each block scaled to unit median distance); alpha > 0 adds tissue geometry (fused Gromov-Wasserstein).
    groupby: obs column with the states/cell types to summarise fates over.
    cells1 / cells2: groups (list; a prefix such as 'Fibro' selects all 'Fibro*' groups) or a boolean mask restricting
    source / target cells (e.g. one lineage).
    pca_sections: optional list of AnnData (e.g. all sections of the study); the state PCA is fitted on their cells of the
    matched groups (or of the boolean masks in `pca_cells`, one per section) instead of on the two matched sections only,
    so all pairs share one state space.
    scale_signals: standardise each received signal over the two sections (set False for signals already standardised)."""
    for ad in (adata1, adata2):
        if "nf_signal" not in ad.obsm:
            raise ValueError("run nf.compute_niche on both sections first")
    i1, i2 = _mask(adata1, groupby, cells1), _mask(adata2, groupby, cells2)
    L1, L2 = adata1.layers["nf_lognorm"][i1], adata2.layers["nf_lognorm"][i2]
    shared = adata1.var_names.isin(adata2.var_names)
    L2 = adata2[:, adata1.var_names[shared]].layers["nf_lognorm"][i2]; L1 = L1[:, shared]
    if pca_sections is None:
        Lc = sp.vstack([sp.csr_matrix(L1), sp.csr_matrix(L2)])
    else:
        g = adata1.var_names[shared]
        grp = np.unique(np.r_[adata1.obs[groupby].astype(str).values[i1], adata2.obs[groupby].astype(str).values[i2]])
        msk = pca_cells or [x.obs[groupby].astype(str).isin(grp).values for x in pca_sections]
        Lc = sp.vstack([sp.csr_matrix(x[:, g].layers["nf_lognorm"][np.asarray(m)]) for x, m in zip(pca_sections, msk)])
    mu = np.asarray(Lc.mean(0)).ravel()
    pca = PCA(n_pcs, random_state=seed).fit(U.dense(Lc) - mu)
    Z1, Z2 = pca.transform(U.dense(L1) - mu), pca.transform(U.dense(L2) - mu)
    A1, A2 = adata1.obsm["nf_signal"][i1].astype(float), adata2.obsm["nf_signal"][i2].astype(float)
    Aall = np.vstack([A1, A2]); keep = Aall.std(0) > 1e-9
    m, s = (Aall.mean(0), np.maximum(Aall.std(0), 1e-9)) if scale_signals else (0.0, 1.0)
    A1, A2 = ((A1 - m) / s)[:, keep], ((A2 - m) / s)[:, keep]
    E1, E2 = adata1.obsm["nf_niche"][i1].astype(float), adata2.obsm["nf_niche"][i2].astype(float)
    F1, F2 = [Z1 / np.sqrt(U.median_sqd(Z1, Z2))], [Z2 / np.sqrt(U.median_sqd(Z1, Z2))]
    if w_signal > 0 and keep.any():
        ma = U.median_sqd(A1, A2); F1.append(np.sqrt(w_signal) * A1 / np.sqrt(ma)); F2.append(np.sqrt(w_signal) * A2 / np.sqrt(ma))
    if w_niche > 0:
        me = U.median_sqd(E1, E2); F1.append(np.sqrt(w_niche) * E1 / np.sqrt(me)); F2.append(np.sqrt(w_niche) * E2 / np.sqrt(me))
    c1, c2 = adata1.obsm["nf_loc"][i1].astype(float), adata2.obsm["nf_loc"][i2].astype(float)
    if alpha > 0:
        c1 = c1 / np.sqrt(U.median_sqd(c1, c1, seed=1)); c2 = c2 / np.sqrt(U.median_sqd(c2, c2, seed=2))
    lab1 = adata1.obs[groupby].astype(str).values[i1]; lab2 = adata2.obs[groupby].astype(str).values[i2]
    r = solve(np.hstack(F1), np.hstack(F2), c1, c2, alpha, eps, tau, lab1, lab2, chunk=2048, inner_tol=1e-4,
              return_cell=True, device=device)
    return NicheFateResult(adata1, adata2, groupby, i1, i2, np.asarray(r["u1"]).astype(str), np.asarray(r["u2"]).astype(str),
                           r["cell_mass"], Z1.astype(np.float32),
                           dict(w_signal=w_signal, w_niche=w_niche, alpha=alpha, eps=eps, tau=tau, n_pcs=n_pcs))


# ----------------------------------------------------------------------------------------------------------- 3. fate
def _groups(result, g):
    if g is None:
        return list(result.groups2)
    g = [g] if isinstance(g, str) else list(g)
    out = []
    for x in g:
        hit = [u for u in result.groups2 if u == x] or [u for u in result.groups2 if u.startswith(x)]
        if not hit:
            raise ValueError(f"{x!r} is not a target group; groups are {list(result.groups2)}")
        out += hit
    return list(dict.fromkeys(out))


def fate(result, target=None, among=None, key_added=None):
    """Fate probability of every source cell.
    target=None -> DataFrame (source cells x target groups, rows sum to 1).
    target='DAM' (or a list; a prefix such as 'IAF' selects all 'IAF*' groups) -> P(target | among), returned as a Series
    and written to adata1.obs[key_added] (default 'nf_fate_<target>')."""
    CM = result.cell_mass; names = result.adata1.obs_names[result.idx1]
    if target is None:
        return pd.DataFrame(CM / np.maximum(CM.sum(1, keepdims=True), 1e-30), index=names, columns=result.groups2)
    col = {g: j for j, g in enumerate(result.groups2)}
    t, a = _groups(result, target), _groups(result, among)
    P = CM[:, [col[g] for g in t]].sum(1) / np.maximum(CM[:, [col[g] for g in set(a) | set(t)]].sum(1), 1e-30)
    key = key_added or "nf_fate_" + (target if isinstance(target, str) else "_".join(target))
    v = np.full(result.adata1.n_obs, np.nan); v[result.idx1] = P
    result.adata1.obs[key] = v
    return pd.Series(P, index=names, name=key)


# ----------------------------------------------------------------------------------------------------------- 4. drivers
def detection(adatas, celltype_key, min_cells=50):
    """Per signal: the best ligand-receptor pair's min(ligand, receptor) detection, where each is the highest fraction of
    cells with counts > 0 in any group of `celltype_key` with >= min_cells cells (multi-subunit = least detected subunit).
    A list of sections is pooled (fractions averaged over sections, groups present in all)."""
    adatas = adatas if isinstance(adatas, (list, tuple)) else [adatas]
    lr = adatas[0].uns["nf_lr"]
    genes = sorted({g for x in list(lr["ligand"]) + list(lr["receptor"]) for g in str(x).split("_")} & set(adatas[0].var_names))
    Ms = []
    for adata in adatas:
        ct = adata.obs[celltype_key].astype(str).values; u, n = np.unique(ct, return_counts=True)
        X = adata[:, genes].layers["nf_lognorm"]; X = (X > 0) if sp.issparse(X) else sp.csr_matrix(np.asarray(X) > 0)
        Ms.append(pd.DataFrame({c: np.asarray(X[ct == c].mean(0)).ravel() for c in u[n >= min_cells]}, index=genes))
    M = (sum(Ms) / len(Ms)).dropna(axis=1).values; gi = {g: i for i, g in enumerate(genes)}
    best = lambda c: (M[[gi[g] for g in c.split("_")]].min(0).max() if all(g in gi for g in c.split("_")) else 0.0)
    pair = np.array([min(best(l), best(r)) for l, r in zip(lr["ligand"], lr["receptor"])])
    return pd.Series(pair, index=np.asarray(lr["signal"]).astype(str)).groupby(level=0).max()


def _one(result, fate, cells, composition_key, covariate_keys, n_pcs, radius, block, n_perm, method, signal_groups, rng):
    ad = result.adata1
    f = np.asarray(fate.reindex(ad.obs_names[result.idx1]).values if isinstance(fate, pd.Series) else fate, float)
    sel = np.ones(len(result.idx1), bool)
    if cells is not None:
        c = np.asarray(cells)
        sel = c.copy() if c.dtype == bool else _select(ad.obs[result.groupby].values[result.idx1], c)
    sel &= np.isfinite(f)
    idx = result.idx1[sel]; y = f[sel]
    loc_all = ad.obsm["nf_loc"]; loc = loc_all[idx]
    nb = cKDTree(loc_all).query_ball_point(loc, radius)
    covs = np.c_[np.ones(len(idx)), np.log([len(j) for j in nb]), result.Z1[sel][:, :n_pcs]]
    if composition_key is not None:
        ct = ad.obs[composition_key].astype(str).values; u = np.unique(ct)
        covs = np.c_[covs, np.array([[np.mean(ct[j] == t) for t in u[1:]] for j in nb])]
    for k in covariate_keys or []:
        covs = np.c_[covs, pd.get_dummies(ad.obs[k].astype(str).values[idx]).values[:, 1:].astype(float)]
    A = ad.obsm["nf_signal"][idx].astype(float); names = np.asarray(ad.uns["nf_signal_names"]).astype(str)
    if signal_groups is not None:
        grp = pd.Series(signal_groups).reindex(names).values.astype(str); ug = sorted(set(grp) - {"nan"})
        A = np.stack([A[:, grp == g].mean(1) for g in ug], 1); names = np.array(ug)
    ok = A.std(0) > 1e-9; A, names = A[:, ok], names[ok]
    raw = np.array([spearmanr(a, y)[0] for a in A.T])
    Sn = np.apply_along_axis(U.rank_normal, 0, A)
    if method == "logistic":
        import statsmodels.api as sm
        z = np.empty(len(names)); r = np.empty(len(names)); C = covs[:, 1:]
        for k in range(len(names)):
            Xk = sm.add_constant(np.c_[(Sn[:, k] - Sn[:, k].mean()) / Sn[:, k].std(), C], has_constant="add")
            fit = sm.GLM(y, Xk, family=sm.families.Binomial()).fit(cov_type="HC0")
            r[k], z[k] = fit.params[1], fit.tvalues[1]
        p = 2 * norm.sf(np.abs(z)); adj = pd.DataFrame(U.resid(Sn, covs), index=ad.obs_names[idx], columns=names)
    else:
        Sr = U.resid(Sn, covs); adj = pd.DataFrame(Sr.copy(), index=ad.obs_names[idx], columns=names); Sr /= np.linalg.norm(Sr, axis=0)
        yr = U.resid(U.rank_normal(y), covs); yr /= np.linalg.norm(yr)
        r = yr @ Sr
        bid = np.unique(np.floor(loc / block).astype(int), axis=0, return_inverse=True)[1].ravel()
        order = np.argsort(bid, kind="stable"); groups = np.split(order, np.cumsum(np.bincount(bid))[:-1])
        null = np.empty((n_perm, len(names)))
        for k in range(n_perm):
            perm = np.concatenate([groups[g] for g in rng.permutation(len(groups))])
            yp = np.empty_like(yr); yp[order] = yr[perm]; null[k] = (yp / np.linalg.norm(yp)) @ Sr
        p = (1 + (np.abs(null) >= np.abs(r)).sum(0)) / (n_perm + 1)
        z = (r - null.mean(0)) / null.std(0)
    return pd.DataFrame(dict(r=r, z=z, p=p, raw=raw), index=names), len(idx), adj


def drivers(result, fate, cells=None, composition_key=None, covariate_keys=None, n_pcs=20, radius=5.0, block=10.0,
            n_perm=1000, min_detect=None, celltype_key=None, min_signif=None, signif_key="nf_signif", sign_check=True, method="partial",
            signal_groups=None, seed=0):
    """Rank received signals by their association with the fate, among the source cells.

    method='partial': partial rank correlation of each signal with the fate probability, adjusting for the cell's own state
    (n_pcs PCs), local cell density, the local composition of `composition_key` (within `radius` spacings) and any
    categorical obs columns in `covariate_keys` (e.g. tissue neighbourhood / layer). Null: the fate is permuted between
    spatial blocks (block x block spacings). method='logistic': fractional logistic regression of the fate on each signal
    with the same covariates (Wald z).
    result / fate may be lists (e.g. one per animal): z is combined across them (Stouffer), p = two-sided normal p.
    min_detect: keep signals whose best ligand-receptor pair has ligand and receptor detected in >= this fraction of some
    group of `celltype_key` (default composition_key or groupby); e.g. 0.10 (CellPhoneDB default), 0.05 for low-depth data.
    min_signif: keep signals that CytoSignal calls significantly received (per-cell test, see signal_significance) in >= this
    fraction of the tested source cells (averaged over results); needs adata1.obsm[signif_key] from signal_significance().
    Signals not tested by CytoSignal are dropped. Column 'signif' = that fraction.
    sign_check: column 'stable' = unadjusted Spearman direction agrees with the adjusted one in every result.
    signal_groups: optional mapping signal -> group (dict / Series), e.g. ligand-receptor pair -> pathway; signals of a group
    are averaged before testing.
    Returns signal, r, z, p, q (Benjamini-Hochberg over the kept signals) [, stable, signif, detect], sorted by z;
    Each result gets `.adjusted`: per-cell signals after removing the covariates (DataFrame, cells x signals)."""
    many = isinstance(result, (list, tuple))
    results, fates = (list(result), list(fate)) if many else ([result], [fate])
    rng = np.random.default_rng(seed); parts, n, adjs = [], 0, []
    for res, fa in zip(results, fates):
        d, k, a = _one(res, fa, cells, composition_key, covariate_keys, n_pcs, radius, block, n_perm, method, signal_groups, rng); parts.append(d); n += k; adjs.append(a)
    names = sorted(set.intersection(*[set(d.index) for d in parts]))
    Z = np.column_stack([d.loc[names, "z"] for d in parts])
    out = pd.DataFrame(dict(signal=names, r=np.mean([d.loc[names, "r"] for d in parts], 0)))
    if len(parts) == 1:
        out["z"], out["p"] = Z[:, 0], parts[0].loc[names, "p"].values
    else:
        out["z"] = Z.sum(1) / np.sqrt(Z.shape[1]); out["p"] = 2 * norm.sf(np.abs(out.z))
        for i in range(Z.shape[1]):
            out[f"z_{i + 1}"] = Z[:, i]
    if sign_check:
        out["stable"] = np.all([(np.sign(d.loc[names, c].values) == np.sign(out.z.values)) for d in parts for c in ("raw", "r")], 0)
    if min_signif is not None:
        fr = []
        for res, a in zip(results, adjs):
            if signif_key not in res.adata1.obsm:
                raise ValueError(f"min_signif needs adata1.obsm['{signif_key}']: run nichefate.signal_significance() on each section first")
            S = res.adata1.obsm[signif_key]
            S = S if isinstance(S, pd.DataFrame) else pd.DataFrame(S, index=res.adata1.obs_names)
            fr.append(S.reindex(index=a.index, columns=out.signal).astype(float).mean(0).values)
        out["signif"] = np.nanmean(np.vstack(fr), 0) if len(fr) > 1 else fr[0]
        out = out[out.signif >= min_signif]
    if min_detect is not None:
        key = celltype_key or composition_key or results[0].groupby
        det = detection([r.adata1 for r in results], key)
        out["detect"] = det.reindex(out.signal).fillna(0).values
        out = out[out.detect >= min_detect]
    out["q"] = U.bh(out.p.values)
    cols = ["signal", "r", "z", "p", "q"] + [c for c in out.columns if c not in ("signal", "r", "z", "p", "q")]
    out = out[cols].sort_values("z", ascending=False).reset_index(drop=True)
    out.attrs["n_cells"] = int(n)
    for res, a in zip(results, adjs):
        res.adjusted = a
    return out
