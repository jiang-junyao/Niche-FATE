"""Niche-FATE: niche-aware cell fate inference from spatial transcriptomics time courses.

    import nichefate as nf
    nf.compute_niche([ad_t1, ad_t2])                 # received signals + neighbourhood embedding
    res = nf.match(ad_t1, ad_t2, groupby="state")    # match cells across time
    p = nf.fate(res, "DAM")                          # fate probability per cell
    d = nf.drivers(res, p)                           # which received signals drive the fate
"""
from .core import compute_niche, match, fate, drivers, detection, NicheFateResult
from . import pl

__all__ = ["compute_niche", "match", "fate", "drivers", "detection", "NicheFateResult", "pl"]
