# Niche-FATE

Niche-aware cell fate inference from spatial transcriptomics time courses.

- **The niche decides the match.** Cells are matched across time points by optimal transport on their own expression, the ligand–receptor signals they receive and a graph embedding of their neighbourhood, so cells that look alike but sit in different niches can get different fates.
- **From niche to associated signals.** Each earlier cell gets a fate probability; received signals are then ranked by their association with that fate, adjusted for the cell's own state and local composition, with a spatial permutation test. An association does not by itself show that a signal causes the fate.
- **Whole sections.** The transport never stores the cell × cell matrix, so sections of ~100,000 cells each run on one GPU.

## Install

```
git clone https://github.com/jiang-junyao/Niche-FATE.git
cd Niche-FATE
pip install -e .
```

Requires Python ≥ 3.9 and PyTorch (GPU optional).

## Quick start

```python
import nichefate as nf

nf.compute_niche([ad_t1, ad_t2], species="mouse")                     # received signals + niche embedding
res = nf.match(ad_t1, ad_t2, groupby="state")                         # match cells across time
p = nf.fate(res, "DAM", among=["homeostatic", "DAM"])                 # fate probability of each earlier cell
d = nf.drivers(res, p, cells=["homeostatic"], composition_key="cell_type", min_detect=0.10)   # signals associated with the fate

nf.pl.sankey([res], ["homeostatic", "DAM"])
nf.pl.fate_map(ad_t1, "nf_fate_DAM")
nf.pl.drivers(d)
```

Inputs: one AnnData per section with raw counts in `.X`, coordinates in `.obsm["spatial"]` and cell states in `.obs`.

### Optional: keep only signals that CytoSignal calls significant

`nf.signal_significance` runs the official [CytoSignal](https://github.com/welch-lab/cytosignal) per-cell significance test (spatial FDR, p < 0.05) for the ligand–receptor pairs behind each signal. `drivers(..., min_signif=...)` then reports only signals that are significantly received by at least that fraction of the tested cells. Needs R with the `cytosignal` package.

```python
nf.signal_significance(ad_t1, um_per_unit=1.0, min_counts=20)        # coordinates in µm; low-depth panel -> lower QC threshold
d = nf.drivers(res, p, cells=["homeostatic"], composition_key="cell_type", min_signif=0.05)
```

## Examples

Data are in [`tutorial_data/`](tutorial_data).

| Tutorial | Data | Question |
|---|---|---|
| [1 · Midbrain development](tutorials/1_midbrain_development.ipynb) | Stereo-seq, mouse midbrain E12.5–E16.5 | which radial glia become glioblasts or neuroblasts |
| [2 · Alzheimer's disease](tutorials/2_alzheimers_microglia.ipynb) | Xenium, TgCRND8 vs wild type; STARmap PLUS, TauPS2APP vs control | homeostatic microglia becoming DAM and DAM reverting, the signals associated with conversion, and the relation to amyloid plaques |
| [3 · Colitis regeneration](tutorials/3_colitis_regeneration.ipynb) | MERFISH, DSS colitis, 15 mice | fibroblasts becoming inflammation-associated fibroblasts and the signals associated with this |

## Contact

Junyao Jiang · jyjiang@dandrite.au.dk
