# Annot

Lightweight Python helpers to annotate untargeted metabolomics feature tables for both LC-MS and GC-MS. The package ships with curated reference libraries (HMDB-derived LC formulas, an internal LC library with RT, and a GC library with RT anchors) and matching routines that return best hits plus candidate lists.

## Installation

```bash
pip install git+https://github.com/BM-Boris/annot.git .
```

Requires Python 3.8+, `pandas`, `numpy`, and `tqdm` (installed automatically).

## Quick start

```python
from annot import run_lc, run_gc

# LC-MS against HMDB-style masses
results_lc = run_lc(
    data="features_lc.tsv",
    sep="\t",
    mz_col="mz",
    ion_mode="pos",        # or "neg"
    lib="hmdb",            # or "annot" for RT-aware internal library
    save="annotations_lc.csv",
)

# GC-MS with RT anchoring
results_gc = run_gc(
    data="features_gc.tsv",
    sep="\t",
    mz_col="mz",
    rt_col="rt",           # retention time in seconds
    save="annotations_gc.csv",
)
```

## Input expectations

- LC-MS: tabular file (CSV/TSV) with at least an `mz` column; add `rt` when using the RT-aware `lib="annot"` mode.  
- GC-MS: tabular file with `mz` and `rt` columns (RT in seconds).
- Columns are selected by name via `mz_col` / `rt_col`; everything else is preserved and returned with annotations.

## LC-MS annotation

- `lib="hmdb"` matches feature m/z to monoisotopic masses and common adducts (pos: `[M+H]+`, `[M+Na]+`, `[M+NH4]+`, `[M+K]+`; neg: `[M-H]-`, `[M+Cl]-`). Restrict adducts with `adducts=["[M+H]+", ...]`.
- `lib="annot"` uses the internal LC library (`annot/data/lc_lib.csv`) with observed RTs; matches on both m/z and RT.
- RT shift handling: `shift="auto"` infers a shift using Glutamine (pos) or Glutamate (neg); provide a float (seconds) to override.
- Tolerances: `mz_diff` is a relative tolerance (default 5e-6), `time_diff` is relative RT tolerance (default 0.05).
- Output columns include: original columns, `annotation` (best hit), `adduct`, `ppm_error`, `formula`, `distance` (matching score), and `candidates` (all hits within tolerances).

## GC-MS annotation

- Uses the bundled GC library (`annot/data/gcms_lib.csv`) with m/z and RT anchors.
- `shift="auto"` estimates RT shift from 4,4'-DDE anchors (falls back to 0 if not found); provide a float to override.
- Peaks are filtered into groups within `time_range` (seconds) and require `ngroup` members (defaults: 2.0 sec, 3 peaks). Groups with confirmed `mz0` anchors are preferred.
- Tolerances: `mz_diff` (relative, default 5e-6) and `time_diff` (relative RT, default 0.05).
- Output columns: original m/z/rt, `annotation`, `notes` (e.g., anchor labels), and `candidates` sorted by distance.

## Libraries

The package includes:

- `annot/data/hmdb.csv` - HMDB-based LC masses/formulas.  
- `annot/data/lc_lib.csv` - internal LC library with RT/adduct info.  
- `annot/data/gcms_lib.csv` - GC library with RT anchors.  

Use `path` in `load_library_data` or `lib` in `run_lc` to point to custom files if needed.

## Tips

- Keep input RT units consistent (seconds throughout).  
- Lower `mz_diff`/`time_diff` to tighten matches; increase when instrument error is higher.  
- Save outputs via the `save` argument to capture the annotated table as CSV.
