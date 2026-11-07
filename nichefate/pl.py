"""Plotting."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

_PAL = ["#3a6db5", "#e07b4f", "#5a9e6b", "#8e3c6d", "#c49a3a", "#6e9a9b", "#9b7cc4", "#c98a5e", "#8f8983", "#d36a8c"]


def _ax(ax, size=(4, 4)):
    if ax is None:
        _, ax = plt.subplots(figsize=size)
    return ax


def transitions(result, ax=None, cmap="Blues", annot=True, min_frac=0.0):
    """Heatmap of source group -> target group fractions (rows sum to 1)."""
    T = result.transitions
    T = T.loc[:, T.max(0) >= min_frac]
    ax = _ax(ax, (0.6 * T.shape[1] + 2, 0.45 * T.shape[0] + 1.5))
    im = ax.imshow(T.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(T.shape[1]), T.columns, rotation=45, ha="right"); ax.set_yticks(range(T.shape[0]), T.index)
    ax.set_xlabel("later state"); ax.set_ylabel("earlier state")
    if annot:
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                v = T.values[i, j]
                if v >= 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04, label="fraction")
    return ax


def sankey(results, groups, labels=None, colors=None, ax=None, gap=0.04):
    """Flows of the given groups across consecutive matchings (results[k] maps time k -> k+1; a list at step k is pooled,
    e.g. all animal pairs of one interval).
    First bar = observed share of each group among `groups`; flows = predicted transitions, chained over time."""
    ax = _ax(ax, (2.2 * (len(results) + 1), 3.5))
    colors = colors or {g: _PAL[i % len(_PAL)] for i, g in enumerate(groups)}
    labels = labels or [f"t{k}" for k in range(len(results) + 1)]
    W = 0.12
    for k, res in enumerate(results):
        rs = res if isinstance(res, (list, tuple)) else [res]
        T = sum(r.transitions.reindex(index=groups, columns=groups).fillna(0) for r in rs) / len(rs)
        if k == 0:
            lab = np.concatenate([r.adata1.obs[r.groupby].astype(str).values[r.idx1] for r in {id(r.adata1): r for r in rs}.values()])
            share = np.array([np.mean(lab == g) for g in groups]); share = share / max(share.sum(), 1e-30)
        else:
            share = inc                                    # chain: what arrived at time k flows on
        flow = share[:, None] * T.values; flow = flow / max(flow.sum(), 1e-30)
        x0, x1 = k, k + 1
        y = 0; ys = {}
        for i, g in enumerate(groups):
            ax.add_patch(Rectangle((x0 - W / 2, y), W, share[i], color=colors[g])); ys[g] = y; y += share[i] + gap
        inc = flow.sum(0); y = 0; yt = {}
        for j, g in enumerate(groups):
            ax.add_patch(Rectangle((x1 - W / 2, y), W, inc[j], color=colors[g])); yt[g] = y; y += inc[j] + gap
        out_off = dict(ys); in_off = dict(yt)
        for i, g in enumerate(groups):
            for j, h in enumerate(groups):
                v = flow[i, j]
                if v <= 1e-4:
                    continue
                a0, b0 = out_off[g], in_off[h]
                xs = np.linspace(x0 + W / 2, x1 - W / 2, 30); s = (1 - np.cos(np.pi * (xs - xs[0]) / (xs[-1] - xs[0]))) / 2
                top = a0 + (b0 - a0) * s; bot = top + v
                ax.add_patch(Polygon(np.r_[np.c_[xs, top], np.c_[xs[::-1], bot[::-1]]], color=colors[h], alpha=0.35, lw=0))
                out_off[g] += v; in_off[h] += v
    ax.set_xlim(-0.4, len(results) + 0.4); ax.set_ylim(-0.05, 1 + gap * len(groups) + 0.05); ax.invert_yaxis()
    ax.set_xticks(range(len(results) + 1), labels); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=colors[g]) for g in groups], labels=groups, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(groups), 4))
    return ax


def fate_map(adata, key, ax=None, cmap="RdBu_r", vmin=0, vmax=1, s=3, title=None):
    """Fate probability (adata.obs[key]) on the tissue; other cells in grey."""
    ax = _ax(ax, (5, 5)); xy = adata.obsm["nf_loc"]; v = adata.obs[key].values.astype(float); m = np.isfinite(v)
    ax.scatter(xy[~m, 0], xy[~m, 1], s=s * 0.4, c="#e6e3de", lw=0, rasterized=True)
    sc = ax.scatter(xy[m, 0], xy[m, 1], s=s, c=v[m], cmap=cmap, vmin=vmin, vmax=vmax, lw=0, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.04, label=key); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title or key)
    return ax


def signal_map(adata, signal, ax=None, cells=None, cmap="viridis", s=3):
    """Received signal (one column of obsm['nf_signal']) on the tissue; `cells` = boolean mask to colour (others grey)."""
    ax = _ax(ax, (5, 5)); xy = adata.obsm["nf_loc"]; names = list(np.asarray(adata.uns["nf_signal_names"]).astype(str))
    v = adata.obsm["nf_signal"][:, names.index(signal)]; m = np.ones(adata.n_obs, bool) if cells is None else np.asarray(cells)
    ax.scatter(xy[~m, 0], xy[~m, 1], s=s * 0.4, c="#e6e3de", lw=0, rasterized=True)
    sc = ax.scatter(xy[m, 0], xy[m, 1], s=s, c=v[m], cmap=cmap, lw=0, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.04, label=signal); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(f"{signal} received")
    return ax


def drivers(df, top=10, ax=None, alpha=0.05, by="q", stable=True, color="#3a6db5", neg_color="#c8553d"):
    """Driver signals as bars of z (right = pushes towards the fate, left = away from it), up to `top` per direction; stars by df[by].
    Only signals with df[by] < alpha (and, with stable=True, the 'stable' column) are shown; alpha=None shows the top ones regardless."""
    d = df if alpha is None else df[df[by] < alpha]
    if stable and alpha is not None and "stable" in d:
        d = d[d.stable]
    d = pd.concat([d[d.z > 0].head(top), d[d.z < 0].tail(top)]).sort_values("z")
    ax = _ax(ax, (4, 0.32 * max(len(d), 1) + 1))
    ax.barh(range(len(d)), d.z, color=[color if z > 0 else neg_color for z in d.z], height=0.7)
    ax.set_yticks(range(len(d)), d.signal)
    for i, (z, v) in enumerate(zip(d.z, d[by])):
        ax.text(z + (0.08 if z > 0 else -0.08), i, "***" if v < 1e-3 else "**" if v < 1e-2 else "*" if v < 0.05 else "", va="center", ha="left" if z > 0 else "right")
    lo, hi = min(d.z.min(), 0) if len(d) else 0, max(d.z.max(), 0) if len(d) else 1; pad = 0.18 * max(hi - lo, 1)
    ax.set_xlim(lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0))
    ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("association with fate (z)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def binned(x, y, n_bins=5, ax=None, color="#c8553d", xlabel=None, ylabel=None, n_boot=1000, seed=0):
    """Mean of y in quantile bins of x (points at the bin median of x; band = bootstrap 95% CI) + Spearman rho over all cells."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    b = np.clip(np.digitize(x, np.quantile(x, np.linspace(0, 1, n_bins + 1))[1:-1]), 0, n_bins - 1); rng = np.random.default_rng(seed)
    xm, mu, lo, hi = [], [], [], []
    for k in range(n_bins):
        v = y[b == k]; bs = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
        xm.append(np.median(x[b == k])); mu.append(v.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
    ax = _ax(ax, (3.2, 3))
    ax.fill_between(xm, lo, hi, color=color, alpha=0.2, lw=0); ax.plot(xm, mu, "-o", color=color, lw=2, ms=5)
    r = spearmanr(x, y); ax.text(0.97, 0.97, f"rho = {r.correlation:.2f}, p = {r.pvalue:.1e}\nn = {len(x):,}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.set_xlabel(xlabel or ""); ax.set_ylabel(ylabel or "")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def groups_map(adata, key, ax=None, s=3, colors=None, order=None):
    """Categorical map of adata.obs[key] (e.g. cell states) on the tissue."""
    ax = _ax(ax, (5, 5)); xy = adata.obsm["nf_loc"]; lab = adata.obs[key].astype(str).values
    order = order or sorted(set(lab)); colors = colors or {g: _PAL[i % len(_PAL)] for i, g in enumerate(order)}
    m = np.isin(lab, order)
    ax.scatter(xy[~m, 0], xy[~m, 1], s=s * 0.4, c="#e6e3de", lw=0, rasterized=True)
    for g in order:
        k = lab == g; ax.scatter(xy[k, 0], xy[k, 1], s=s, c=colors[g], lw=0, label=g, rasterized=True)
    ax.legend(frameon=False, markerscale=3, loc="upper left", bbox_to_anchor=(1, 1)); ax.set_aspect("equal"); ax.axis("off")
    return ax
