"""Per-cell significance of received ligand-receptor signals with the official CytoSignal test (R package `cytosignal`).

CytoSignal (Peng et al.) scores each cell's received signal from the ligand of its neighbours and its own receptor, and then calls,
per cell and pair, whether the score is significant (inferSignif: spatial permutation null, spatial FDR, p < 0.05). Niche-FATE uses
this as a filter on its driver test: a signal is only reported as a driver if CytoSignal finds it significantly received by enough
of the cells being tested (drivers(..., min_signif=...)). Requires R with the `cytosignal` package; CytoSignal runs with its defaults
except the three settings below, which depend on the platform."""
import glob
import os
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

_R = os.path.join(os.path.dirname(__file__), "data", "cytosignal_signif.R")


def _rscript(rscript):
    if rscript:
        return rscript
    p = shutil.which("Rscript")
    if p:
        return p
    for c in sorted(glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"), reverse=True):
        return c
    raise RuntimeError("Rscript not found; install R and the cytosignal package, or pass rscript='path/to/Rscript'")


def _pairs(adata, lr, signals):
    if lr is None:
        if "nf_lr" not in adata.uns:
            raise ValueError("no ligand-receptor table: run compute_niche first or pass lr=DataFrame(ligand, receptor, signal)")
        u = adata.uns["nf_lr"]; lr = pd.DataFrame(dict(ligand=u["ligand"], receptor=u["receptor"], signal=u["signal"]))
    lr = lr.copy()
    if "signal" not in lr:
        lr["signal"] = lr["pathway"] if "pathway" in lr else lr.ligand.astype(str) + "-" + lr.receptor.astype(str)
    lr = lr.astype({"ligand": str, "receptor": str, "signal": str})
    if signals is not None:
        lr = lr[lr.signal.isin([str(s) for s in signals])]
    return lr.drop_duplicates(["ligand", "receptor", "signal"]).reset_index(drop=True)


def signal_significance(adata, signals=None, lr=None, layer=None, spatial_key="spatial", um_per_unit=1.0, min_counts=300,
                        perm_size=1000, cells=None, groupby=None, rscript=None, workdir=None, key_added="nf_signif", verbose=True):
    """Run CytoSignal's per-cell significance test on one section for the ligand-receptor pairs behind `signals`.

    adata: one section with raw counts (adata.X, or `layer`) and coordinates in adata.obsm[spatial_key].
    signals: signal names to test (default: all of adata.uns['nf_signal_names'] that have pairs in the table).
    lr: optional DataFrame with columns ligand, receptor, signal (subunits joined by '_'); default adata.uns['nf_lr'] from compute_niche.
    um_per_unit: microns per coordinate unit (CytoSignal's inferEpsParams scale.factor; its diffusion range is defined in microns).
    min_counts: CytoSignal's removeLowQuality counts.thresh (its default 300; lower it for low-depth imaging panels, e.g. 10-20).
    perm_size: CytoSignal's inferIntrScore permutation size (default 1000).
    cells / groupby: optional cells (labels of `groupby`) for which the summary fraction is computed (default: all cells).
    Only the genes of the tested pairs are passed to CytoSignal; all other counts are pooled into one column so that the library
    size (used by CytoSignal's QC and normalisation) is unchanged.

    Adds adata.obsm[key_added]: DataFrame (cells x signals, bool), True if the cell significantly receives any pair of the signal;
    adata.uns[key_added + '_pairs']: per pair, number and fraction of significant cells.
    Returns per signal: n_pairs, frac (fraction of the selected cells significant for at least one pair), best_pair, best_pair_frac."""
    lr = _pairs(adata, lr, signals)
    if signals is None and "nf_signal_names" in adata.uns:
        lr = lr[lr.signal.isin(np.asarray(adata.uns["nf_signal_names"]).astype(str))].reset_index(drop=True)
    if len(lr) == 0:
        raise ValueError("no ligand-receptor pairs to test")
    genes = list(map(str, adata.var_names)); gi = {g: i for i, g in enumerate(genes)}
    ok = [all(q in gi for q in (l + "_" + r).split("_")) for l, r in zip(lr.ligand, lr.receptor)]
    lr = lr[np.array(ok, bool)].reset_index(drop=True)
    need = sorted({q for l, r in zip(lr.ligand, lr.receptor) for q in (l + "_" + r).split("_")})
    X = adata.layers[layer] if layer else adata.X
    X = sp.csr_matrix(X)
    if X.data.size and not np.allclose(X.data[:1000], np.round(X.data[:1000])):
        raise ValueError("CytoSignal needs raw counts: pass layer='counts' (or the layer holding them)")
    Xs = X[:, [gi[g] for g in need]]; rest = np.asarray(X.sum(1)).ravel() - np.asarray(Xs.sum(1)).ravel()
    Xk = sp.hstack([Xs, sp.csr_matrix(rest[:, None])]).tocsc()
    cellnames = [f"c{i}" for i in range(adata.n_obs)]
    tmp = tempfile.mkdtemp(dir=workdir)
    try:
        sio.mmwrite(os.path.join(tmp, "counts.mtx"), sp.csc_matrix(Xk.T.astype(float)))
        open(os.path.join(tmp, "genes.txt"), "w").write("\n".join(need + ["ZZPOOLEDREST"]) + "\n")
        open(os.path.join(tmp, "cells.txt"), "w").write("\n".join(cellnames) + "\n")
        np.savetxt(os.path.join(tmp, "loc.csv"), np.asarray(adata.obsm[spatial_key], float)[:, :2], delimiter=",")
        open(os.path.join(tmp, "lr.csv"), "w").write("\n".join(["ligand,receptor,type"] + [f"{l},{r},diffusion-dependent" for l, r in zip(lr.ligand, lr.receptor)]) + "\n")
        open(os.path.join(tmp, "params.txt"), "w").write(f"{um_per_unit}\n{perm_size}\n{min_counts}\n")
        if verbose:
            print(f"CytoSignal significance: {len(lr)} pairs, {len(set(lr.signal))} signals, {adata.n_obs} cells")
        r = subprocess.run([_rscript(rscript), _R, tmp], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("CytoSignal failed:\n" + r.stderr[-3000:])
        ids = {}
        for line in open(os.path.join(tmp, "ids.tsv")).read().strip().split("\n"):
            i, l, rr = line.split("\t"); ids[i] = (l, rr)
        sig = {(l, rr): set() for l, rr in zip(lr.ligand, lr.receptor)}
        for f in glob.glob(os.path.join(tmp, "signif_*.tsv")):
            for line in open(f).read().strip().split("\n")[1:]:
                if line.strip():
                    i, cs = (line.split("\t") + [""])[:2]
                    sig.setdefault(ids.get(i, (i, i)), set()).update(c for c in cs.split(",") if c)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    idx = {c: k for k, c in enumerate(cellnames)}
    P = np.zeros((adata.n_obs, len(lr)), bool)
    for k, (l, rr) in enumerate(zip(lr.ligand, lr.receptor)):
        P[[idx[c] for c in sig[(l, rr)] if c in idx], k] = True
    sel = np.ones(adata.n_obs, bool)
    if cells is not None:
        lab = adata.obs[groupby].astype(str).values
        sel = np.isin(lab, [cells] if isinstance(cells, str) else [str(c) for c in cells])
    lr["n_significant"] = P[sel].sum(0); lr["frac"] = P[sel].mean(0)
    names = list(dict.fromkeys(lr.signal))
    S = pd.DataFrame({s: P[:, (lr.signal == s).values].any(1) for s in names}, index=adata.obs_names)
    adata.obsm[key_added] = S
    adata.uns[key_added + "_pairs"] = lr
    out = []
    for s in names:
        q = lr[lr.signal == s].sort_values("frac", ascending=False)
        out.append(dict(signal=s, n_pairs=len(q), frac=S.loc[sel, s].mean(), best_pair=f"{q.ligand.iloc[0]} -> {q.receptor.iloc[0]}", best_pair_frac=q.frac.iloc[0]))
    return pd.DataFrame(out).set_index("signal")
