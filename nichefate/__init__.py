"""Niche-FATE: niche-aware cell fate inference from spatial transcriptomics time courses.

    import nichefate as nf
    nf.compute_niche([ad_t1, ad_t2])                 # received signals + neighbourhood embedding
    res = nf.match(ad_t1, ad_t2, groupby="state")    # match cells across time
    p = nf.fate(res, "DAM")                          # fate probability per cell
    d = nf.drivers(res, p)                           # which received signals drive the fate
    nf.signal_significance(ad_t1)                    # optional: CytoSignal per-cell significance (R), then
    d = nf.drivers(res, p, min_signif=0.05)          #   keep signals significantly received by >= 5% of the tested cells
"""
from .core import compute_niche, match, fate, drivers, detection, NicheFateResult
from ._csig import signal_significance
from . import pl

__all__ = ["compute_niche", "match", "fate", "drivers", "detection", "signal_significance", "NicheFateResult", "pl"]
