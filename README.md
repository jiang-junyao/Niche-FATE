# Niche-FATE

Niche-aware cell fate inference from spatial transcriptomics time courses.

- **The niche decides the match.** Cells are matched across time points by optimal transport on their own expression, the ligand–receptor signals they receive and a graph embedding of their neighbourhood, so cells that look alike but sit in different niches can get different fates.
- **From niche to driver signals.** Each earlier cell gets a fate probability; received signals are then ranked by their association with that fate, adjusted for the cell's own state and local composition, with a spatial permutation test.
- **Whole sections.** The transport never stores the cell × cell matrix, so sections of ~100,000 cells each run on one GPU.

## Install

```
git clone https://github.com/<user>/Niche-FATE.git
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
d = nf.drivers(res, p, cells=["homeostatic"], composition_key="cell_type", min_detect=0.10)   # driver signals

nf.pl.sankey([res], ["homeostatic", "DAM"])
nf.pl.fate_map(ad_t1, "nf_fate_DAM")
nf.pl.drivers(d)
```

Inputs: one AnnData per section with raw counts in `.X`, coordinates in `.obsm["spatial"]` and cell states in `.obs`.

## Examples

Data are in [`tutorial_data/`](tutorial_data).

| Tutorial | Data | Question |
|---|---|---|
| [1 · Midbrain development](tutorials/1_midbrain_development.ipynb) | Stereo-seq, mouse midbrain E12.5–E16.5 | which radial glia become glioblasts or neuroblasts |
| [2 · Alzheimer's disease](tutorials/2_alzheimers_microglia.ipynb) | Xenium, TgCRND8 vs wild type; STARmap PLUS, TauPS2APP vs control | homeostatic microglia becoming DAM, DAM reverting, and the signals involved |
| [3 · Colitis regeneration](tutorials/3_colitis_regeneration.ipynb) | MERFISH, DSS colitis, 15 mice | fibroblasts becoming inflammation-associated fibroblasts and the signals that induce them |

## Contact

Junyao Jiang · jyjiang@dandrite.au.dk
