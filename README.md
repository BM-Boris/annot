# Annot

Lightweight Python helpers to annotate untargeted metabolomics feature tables for LC-MS and GC-MS. The package ships with bundled reference libraries (HMDB- and T3DB-derived LC masses/formulas, an RT-aware LC library, and a GC library with RT anchors or your own library) and matching routines that return best hits plus candidate lists.

## Installation

```bash
pip install git+https://github.com/BM-Boris/annot.git
```

Requires Python 3.10+, `pandas`, `numpy`, and `tqdm` (installed automatically).

## Quick start

```python
from annot import run_lc, run_gc

# LC-MS against HMDB-style masses
results_lc = run_lc(
    data="features_lc.tsv",
    sep="\t",
    mz_col="mz",
    ion_mode="pos",        # or "neg"
    lib="hmdb",            # "t3db" for toxicants, or "annot" for RT-aware LC matching
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

- LC-MS: tabular file (CSV/TSV) with at least an `mz` column; add `rt` when using the RT-aware `lib="annot"` mode or a user library with populated RT.
- GC-MS: tabular file with `mz` and `rt` columns (RT in seconds).
- Columns are selected by name via `mz_col` / `rt_col`.
- Returned tables contain matched rows only. Extra input columns are not included in the output.

## LC-MS annotation

- `lib="hmdb"` and `lib="t3db"` match feature m/z to monoisotopic masses and common adducts (pos: `[M+H]+`, `[M+Na]+`, `[M+NH4]+`, `[M+K]+`; neg: `[M-H]-`, `[M-2H]2-`, `[M+Cl]-`). Restrict adducts with `adducts=["[M+H]+", ...]`.
- `lib="annot"` uses the internal LC library (`annot/data/lc_lib.csv`) with observed RTs; matches on both m/z and RT.
- RT shift handling: `shift="auto"` infers a shift using Glutamine; provide a float (seconds) to override.
- Tolerances: `mz_diff` is a relative tolerance (default 5e-6), `time_diff` is relative RT tolerance (default 0.05). RT matches must also be less than 45 seconds apart after shift correction.
- Output columns include: selected `mz`/`rt` columns, `annotation` (best hit), `adduct`, `ppm_error`, `formula`, `distance` (matching score), and `candidates` (all hits within tolerances).

## GC-MS annotation

- Uses the bundled GC library (`annot/data/gcms_lib.csv`) with m/z and RT anchors.
- `shift="auto"` estimates RT shift from p,p'-DDE anchors (falls back to 0 if not found); provide a float to override.
- Peaks are filtered into groups within `time_range` (seconds) and require `ngroup` members (defaults: 2.0 sec, 3 peaks). Groups containing `mz0` are preferred.
- Tolerances: `mz_diff` (relative, default 5e-6) and `time_diff` (relative RT, default 0.05).
- Output columns include selected `mz`/`rt` columns, `annotation`, `notes` (e.g., anchor labels), and `candidates` sorted by distance.

## Libraries

The package includes:

- `annot/data/hmdb.csv` - HMDB-based LC masses/formulas.  
- `annot/data/t3db.csv` - T3DB toxicant masses/formulas.
- `annot/data/lc_lib.csv` - internal LC library with RT/adduct info.  
- `annot/data/gcms_lib.csv` - GC library with RT anchors.  

Use `path` in `load_library_data` to load a custom file directly when needed.

## User libraries

Pass `library_path` to use a comma-separated CSV instead of a bundled library:

```python
results_lc = run_lc("features_lc.csv", sep=",", library_path="my_lc.csv")
results_gc = run_gc("features_gc.csv", sep=",", library_path="my_gc.csv")
```

- LC: required `mz`, `name`; optional `rt`, `adduct`, `formula`. Populated `rt` enables RT matching; absent or entirely blank `rt` means m/z-only matching. Partially missing RT is rejected.
- GC: required `mz`, `rt`, `name`, `ion`. Use ion labels such as `mz0`, `mz1`, `mz2`; existing grouping rules apply.
- Column order does not matter. `mz` is ion m/z, not neutral mass; `rt` is in seconds. Numeric values must be finite and positive; names and GC ion labels must be non-empty.
- For LC, `library_path` overrides `lib`. `ion_mode` and `adducts` do not filter or generate ions; optional `adduct` and `formula` values are copied to the output.
- `shift="auto"` uses zero for user libraries; set a numeric shift in seconds to correct RT drift.

## Tips

- Keep input RT units consistent (seconds throughout).  
- Lower `mz_diff`/`time_diff` to tighten matches; increase when instrument error is higher.  
- Save outputs via the `save` argument to capture the annotated table as CSV.
