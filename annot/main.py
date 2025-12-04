import pandas as pd
import numpy as np
from tqdm import tqdm
import pkg_resources


def load_library_data(
    library_type: str = "lc",
    path: str | None = None,
) -> pd.DataFrame:
    """
    Load the appropriate library file depending on the library_type.

    Parameters
    ----------
    library_type : {"gc", "lc", "annot"}
        - "gc": loads data/gcms_lib.csv (GC-MS library with RT).
        - "lc": loads hmdb.csv (LC-MS library with monoisotopic masses).
        - "annot": loads data/lc_lib.csv (internal LC-MS annotated library with RT).
    path : str or None, default None
        Optional explicit path to a library file. If provided, it overrides the
        default file selected by `library_type`.

    Returns
    -------
    pandas.DataFrame
        Loaded library table.
    """
    if library_type not in {"gc", "lc", "annot"}:
        raise ValueError("library_type must be 'gc', 'lc', or 'annot'")

    filename = path
    if filename is None:
        if library_type == "gc":
            filename = "data/gcms_lib.csv"
        elif library_type == "lc":
            filename = "data/hmdb.csv"
        else:
            filename = "data/lc_lib.csv"
        filename = pkg_resources.resource_filename(__name__, filename)

    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(filename, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last resort: errors='ignore'
    return pd.read_csv(
        filename,
        low_memory=False,
        encoding="utf-8",
        encoding_errors="ignore",
    )


def pre_process(features: pd.DataFrame, lib_gc: pd.DataFrame):
    """
    Preprocess feature and GC library tables for GC-MS matching.

    Parameters
    ----------
    features : pandas.DataFrame
        Feature table with m/z and RT in the first two columns.
    lib_gc : pandas.DataFrame
        GC-MS library table, first two columns are m/z and RT.

    Returns
    -------
    tuple of numpy.ndarray
        (feature_mz, feature_rt, library_mz, library_rt)
    """
    features.dropna(subset=[features.columns[0], features.columns[1]], inplace=True)

    mz_array = features.iloc[:, 0].to_numpy().astype("float64")
    time_array = features.iloc[:, 1].to_numpy().astype("float64")
    lib_mz_array = lib_gc.iloc[:, 0].to_numpy().astype("float64")
    lib_time_array = lib_gc.iloc[:, 1].to_numpy().astype("float64")

    return mz_array, time_array, lib_mz_array, lib_time_array


def run_lc(
    data: str,
    sep: str = "\t",
    mz_col: str = "mz",
    rt_col: str = "rt",
    save: str | None = None,
    mz_diff: float = 5e-6,
    time_diff: float = 0.05,
    ion_mode: str = "pos",
    adducts: list[str] | None = None,
    lib: str = "hmdb",
    shift: str | float = "auto",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Annotate LC-MS data by matching m/z features to either the HMDB-based library
    (mass + adduct matching) or the internal annotated LC library with RT.

    Parameters
    ----------
    data : str
        Path to the input feature table (CSV/TSV).
    sep : str, default "\\t"
        Column separator for the input file.
    mz_col : str, default "mz"
        Name of the m/z column in the input feature table.
    rt_col : str, default "rt"
        Name of the retention time column (used when lib="annot").
    save : str or None, default None
        If provided, save the annotated table to this path (CSV).
    mz_diff : float, default 5e-6
        Maximum allowed relative m/z difference for a match.
    time_diff : float, default 0.05
        Maximum allowed relative RT difference (used when lib="annot").
    ion_mode : {"pos", "neg"}, default "pos"
        Ionization mode:
          - "pos": positive ESI, uses [M+H]+, [M+Na]+, [M+NH4]+, [M+K]+ by default.
          - "neg": negative ESI, uses [M-H]- and [M+Cl]- by default.
    adducts : list of str or None, default None
        Subset of adducts to use. If None, all defaults for the given ion_mode are used.
    lib : {"hmdb", "annot"}, default "hmdb"
        Which LC library to use.
    shift : {"auto", float}, default "auto"
        RT shift (only for lib="annot"). If "auto", infer using Glutamine 

    Returns
    -------
    pandas.DataFrame
        Columns include:
          * mz_col
          * annotation (best hit)
          * adduct (best adduct)
          * ppm_error (for the best hit)
          * formula (chemical formula of the best hit, if available)
          * candidates: all matches within mz_diff, sorted by ppm or distance
    """
    lib = lib.lower()
    if lib not in {"hmdb", "annot"}:
        raise ValueError("lib must be 'hmdb' or 'annot'")

    if lib == "hmdb":
        print(f"[LC] Library: {lib.upper()}, ion_mode={ion_mode}, mz_diff={mz_diff}")
    else:
        print(
            f"[LC] Library: {lib.upper()}, ion_mode={ion_mode}, "
            f"mz_diff={mz_diff}, time_diff={time_diff}"
        )

    features = pd.read_csv(data, sep=sep)

    if lib == "hmdb":
        features = features[[mz_col]].astype("float64")
    else:
        if rt_col not in features.columns:
            raise ValueError(
                f"rt_col '{rt_col}' not found in feature table for annot lib"
            )
        features = features[[mz_col, rt_col]].astype("float64")

    mz_array = features[mz_col].to_numpy()

    lib_lc = load_library_data("lc" if lib == "hmdb" else "annot")

    if lib == "hmdb":
        mass_col = "monisotopic_molecular_weight"
        name_col = "name"
        formula_col = "chemical_formula"
        lib_lc[mass_col] = pd.to_numeric(
            lib_lc[mass_col].astype(str).str.split().str[0],
            errors="coerce",
        )
        lib_lc = lib_lc.dropna(subset=[mass_col])
        lib_mz_array = lib_lc[mass_col].to_numpy()
        adduct_labels = None
        lib_time_array = None
    else:
        mass_col = "Mz_value"
        name_col = "Metabolite"
        formula_col = "Formula"
        rt_ref_col = "RT_sec"
        lib_lc[mass_col] = pd.to_numeric(
            lib_lc["MzAdduct"].astype(str).str.split().str[0],
            errors="coerce",
        )
        lib_lc["Adduct"] = lib_lc["MzAdduct"].astype(str).str.extract(
            r"\(([^)]*)\)",
            expand=False,
        )
        method_key = "HILIC+" if ion_mode == "pos" else "C18-"
        lib_lc = lib_lc[
            lib_lc["Method"]
            .astype(str)
            .str.contains(method_key, case=False, na=False)
        ]
        lib_lc = lib_lc.dropna(subset=[mass_col, rt_ref_col])
        lib_mz_array = lib_lc[mass_col].to_numpy()
        adduct_labels = lib_lc["Adduct"].fillna("").to_numpy()
        lib_time_array = pd.to_numeric(
            lib_lc[rt_ref_col],
            errors="coerce",
        ).to_numpy()
        time_array = features[rt_col].to_numpy()

    lib_name_array = lib_lc[name_col].astype(str).to_numpy()
    lib_formula_array = lib_lc[formula_col].astype(str).to_numpy()

    # Configure adducts (HMDB mode)
    if lib == "hmdb":
        proton = 1.007276
        mass_Na = 22.989218
        mass_NH4 = 18.033823
        mass_K = 38.963158
        mass_Cl = 34.969402

        if ion_mode == "pos":
            default_adducts = {
                "[M+H]+": (proton, 1),
                "[M+Na]+": (mass_Na, 1),
                "[M+NH4]+": (mass_NH4, 1),
                "[M+K]+": (mass_K, 1),
            }
        elif ion_mode == "neg":
            default_adducts = {
                "[M-H]-": (-proton, 1),
                "[M+Cl]-": (mass_Cl, 1),
            }
        else:
            raise ValueError("ion_mode must be 'pos' or 'neg'")

        if adducts is not None:
            used_adducts = {k: v for k, v in default_adducts.items() if k in adducts}
            if not used_adducts:
                raise ValueError(
                    f"None of the requested adducts {adducts} are valid. "
                    f"Allowed adducts: {list(default_adducts.keys())}"
                )
        else:
            used_adducts = default_adducts

        # Precompute theoretical m/z for each adduct (оптимизация пункта 7)
        theo_mz_by_adduct: dict[str, np.ndarray] = {}
        for adduct_name, (mass_shift, charge) in used_adducts.items():
            theo_mz_by_adduct[adduct_name] = (lib_mz_array + mass_shift) / charge
    else:
        used_adducts = None
        theo_mz_by_adduct = {}

    # RT shift for annot library
    rt_shift = 0.0
    if lib == "annot":
        if shift == "auto":
            anchor_name = "glutamine" 
            mask_anchor = lib_lc[name_col].str.lower() == anchor_name
            if not mask_anchor.any():
                print(f"[LC] Anchor {anchor_name} not found in library, shift set to 0")
            else:
                anchor_rows = lib_lc[mask_anchor]
                # prefer adduct compatible with ion mode
                if ion_mode == "pos":
                    anchor_rows = anchor_rows[
                        anchor_rows["Adduct"].str.contains("+", regex=False)
                    ]
                else:
                    anchor_rows = anchor_rows[
                        anchor_rows["Adduct"].str.contains("-", regex=False)
                    ]
                if anchor_rows.empty:
                    anchor_rows = lib_lc[mask_anchor]
                anchor_row = anchor_rows.iloc[0]
                ref_mz = float(anchor_row[mass_col])
                ref_rt = float(anchor_row[rt_ref_col])
                mz_vals = features[mz_col].to_numpy()
                rel_mz_anchor = np.abs(mz_vals - ref_mz) / ref_mz
                rt_vals = features[rt_col].to_numpy()
                rel_rt_anchor = np.abs(rt_vals - ref_rt) / ref_rt
                hit_idx = np.where(rel_mz_anchor < mz_diff)[0]
                if hit_idx.size == 0:
                    print(
                        f"[LC] Anchor {anchor_name} not found in data within tolerances, shift set to 0"
                    )
                else:
                    k = int(hit_idx[np.argmin(rel_mz_anchor[hit_idx])])
                    rt_shift = float(rt_vals[k] - ref_rt)
                    print(
                        f"[LC] Auto shift from {anchor_name}: "
                        f"shift={rt_shift:.3f} sec (feature idx {k})"
                    )
        else:
            rt_shift = float(shift)
            print(f"[LC] Manual RT shift applied: {rt_shift} sec")

    annotations: list[str | None] = []
    adduct_hits: list[str | None] = []
    ppm_best: list[float | None] = []
    best_formulas: list[str | None] = []
    best_distances: list[float | None] = []
    candidates_all: list[list[dict] | None] = []

    for i_feat, mz in enumerate(tqdm(mz_array,disable=not show_progress)):
        best_distance = None
        best_ppm = None
        best_name = None
        best_adduct = None
        best_formula = None
        candidates_raw: list[dict] = []

        if lib == "hmdb":
            # Используем заранее посчитанные theo_mz_by_adduct (пункт 7)
            for adduct_name, theo_mz in theo_mz_by_adduct.items():
                rel_error = np.abs((theo_mz - mz) / theo_mz)

                matched_indices = np.where(rel_error < mz_diff)[0]
                if matched_indices.size == 0:
                    continue

                for idx in matched_indices:
                    ppm = float(rel_error[idx] * 1e6)
                    name = lib_name_array[idx]
                    formula = str(lib_formula_array[idx])
                    candidates_raw.append(
                        {
                            "name": name,
                            "adduct": adduct_name,
                            "ppm": ppm,
                            "formula": formula,
                        }
                    )

                    if best_ppm is None or ppm < best_ppm:
                        best_ppm = ppm
                        best_name = name
                        best_adduct = adduct_name
                        best_formula = formula
        else:
            rt_val = float(features.iloc[i_feat][rt_col] - rt_shift)
            rel_mz = np.abs(lib_mz_array - mz) / lib_mz_array
            rel_rt = np.abs(lib_time_array - rt_val) / lib_time_array
            matched_indices = np.where(
                (rel_mz < mz_diff) & (rel_rt < time_diff)
            )[0]

            if matched_indices.size > 0:
                distances = (lib_mz_array[matched_indices] - mz) ** 2 + 1e-7 * (
                    (lib_time_array[matched_indices] - rt_val) ** 2
                )
                for idx_pos, idx in enumerate(matched_indices):
                    ppm = float(rel_mz[idx] * 1e6)
                    name = lib_name_array[idx]
                    formula = lib_formula_array[idx]
                    adduct_label = (
                        adduct_labels[idx] if adduct_labels is not None else None
                    )
                    candidates_raw.append(
                        {
                            "name": name,
                            "adduct": adduct_label if adduct_label else None,
                            "ppm": ppm,
                            "dist": float(distances[idx_pos]),
                            "formula": formula,
                        }
                    )

                    if best_distance is None or distances[idx_pos] < best_distance:
                        best_distance = distances[idx_pos]
                        best_ppm = ppm
                        best_name = name
                        best_adduct = adduct_label if adduct_label else None
                        best_formula = formula

        if best_name is None:
            annotations.append(None)
            adduct_hits.append(None)
            ppm_best.append(None)
            best_formulas.append(None)
            best_distances.append(None)
            candidates_all.append(None)
        else:
            annotations.append(best_name)
            adduct_hits.append(best_adduct)
            ppm_best.append(best_ppm)
            best_formulas.append(best_formula)
            if lib == "hmdb":
                best_distances.append(best_ppm)
            else:
                best_distances.append(best_distance)
            if lib == "annot":
                candidates_all.append(sorted(candidates_raw, key=lambda c: c["dist"]))
            else:
                candidates_all.append(sorted(candidates_raw, key=lambda c: c["ppm"]))

    features["annotation"] = annotations
    features["adduct"] = adduct_hits
    features["ppm_error"] = ppm_best
    features["formula"] = best_formulas
    features["distance"] = best_distances
    features["candidates"] = candidates_all

    out = features.dropna(subset=["annotation"]).reset_index(drop=True)

    total_candidates = sum(
        len(c) for c in out["candidates"] if isinstance(c, list)
    )
    print(
        f"[LC] Annotated {len(out)} features; unique metabolites: "
        f"{out['annotation'].nunique()}; total candidate entries: {total_candidates}"
    )

    if save is not None:
        out.to_csv(save, index=False)

    return out


def run_gc(
    data: str,
    sep: str = "\t",
    mz_col: str = "mz",
    rt_col: str = "rt",
    save: str | None = None,
    shift: str | float = "auto",
    time_diff: float = 0.05,
    mz_diff: float = 5e-6,
    time_range: float = 2.0,
    ngroup: int = 3,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Annotate GC-MS data by matching features to a reference library with RT anchoring.

    Parameters
    ----------
    data : str
        Path to the input feature table (CSV/TSV).
    sep : str, default "\\t"
        Column separator for the input file.
    mz_col : str, default "mz"
        Name of the m/z column in the input feature table.
    rt_col : str, default "rt"
        Name of the retention time column.
    save : str or None, default None
        If provided, save the annotated table to this path (CSV).
    shift : {"auto", float}, default "auto"
        If "auto", infer RT shift from 4,4'-DDE. If a number, use that value (in seconds).
    time_diff : float, default 0.05
        Maximum allowed relative RT difference for a match.
    mz_diff : float, default 5e-6
        Maximum allowed relative m/z difference for a match.
    time_range : float, default 2.0
        RT window (in seconds) within which peaks are grouped.
    ngroup : int, default 3
        Minimum group size required to keep a group.

    Returns
    -------
    pandas.DataFrame
        Original m/z and RT plus annotation/notes and candidates (all hits within
        tolerances, sorted by distance), filtered to the best groups per chemical.
    """
    features = pd.read_csv(data, sep=sep)
    features = features[[mz_col, rt_col]].astype("float64")

    print(
        f"[GC] mz_diff={mz_diff}, time_diff={time_diff}, "
        f"time_range={time_range}, ngroup={ngroup}, shift={shift}"
    )

    lib_gc = load_library_data("gc")
    mz_array, time_array, lib_mz_array, lib_time_array = pre_process(
        features,
        lib_gc,
    )

    annotations = np.empty(len(mz_array), dtype=object)
    notes = np.empty(len(mz_array), dtype=object)
    candidates_all: list[list[dict] | None] = []

    lib_time_array = lib_time_array * 60

    if shift == "auto":
        tol = 5e-6  # relative m/z tolerance
        mz = np.asarray(mz_array)
        rt = np.asarray(time_array)

        row1 = lib_gc[
            (lib_gc.Name == "4,4'-DDE ") & (lib_gc.note == "mz1")
        ].iloc[0]
        mz1, rt1_ref = row1.mz, row1.time * 60

        dmz1 = np.abs((mz - mz1) / mz1)
        cand_idx = np.where(dmz1 <= tol)[0]

        if cand_idx.size == 0:
            print("No 4,4'-DDE mz1 found in data, using shift = 0")
            shift = 0
        else:
            cand_shifts = rt[cand_idx] - rt1_ref
            base_i = int(np.argmin(dmz1[cand_idx]))
            shift = cand_shifts[base_i]
            confirmed = None
            confirmed_i = base_i

            def anchor_shift(mz_a, rt_a):
                dmz = np.abs((mz - mz_a) / mz_a)
                idx = np.where(dmz <= tol)[0]
                if idx.size == 0:
                    return None
                j = idx[int(np.argmin(dmz[idx]))]
                return rt[j] - rt_a

            def match_anchor(a_shift):
                if a_shift is None:
                    return None
                diffs = np.abs(cand_shifts - a_shift)
                hits = np.where(diffs <= time_range)[0]
                return int(hits[0]) if hits.size > 0 else None

            row_c13 = lib_gc[lib_gc.Name == "C13_pp_DDE "].iloc[0]
            c13_shift = anchor_shift(row_c13.mz, row_c13.time * 60)
            if c13_shift is None:
                print("C13_pp_DDE not found in data")
            else:
                k = match_anchor(c13_shift)
                if k is not None:
                    shift = cand_shifts[k]
                    confirmed = "C13"
                    confirmed_i = k
                    print(
                        "Shift chosen by 4,4'-DDE mz1 candidate confirmed by C13_pp_DDE"
                    )
                else:
                    print(
                        "C13_pp_DDE found in data, but no 4,4'-DDE mz1 candidate "
                        "matches its RT shift"
                    )

            row0 = lib_gc[
                (lib_gc.Name == "4,4'-DDE ") & (lib_gc.note == "mz0")
            ].iloc[0]
            mz0_shift = anchor_shift(row0.mz, row0.time * 60)
            if mz0_shift is None:
                print("4,4'-DDE mz0 not found in data")
            else:
                if confirmed == "C13":
                    if abs(mz0_shift - cand_shifts[confirmed_i]) <= time_range:
                        print("Shift also consistent with 4,4'-DDE mz0")
                    else:
                        print(
                            "4,4'-DDE mz0 shift not consistent with C13-based shift"
                        )
                else:
                    k = match_anchor(mz0_shift)
                    if k is not None:
                        shift = cand_shifts[k]
                        confirmed = "mz0"
                        confirmed_i = k
                        print(
                            "Shift chosen by 4,4'-DDE mz1 candidate confirmed by "
                            "4,4'-DDE mz0"
                        )
                    else:
                        print(
                            "4,4'-DDE mz0 found in data, but no 4,4'-DDE mz1 candidate "
                            "matches its RT shift"
                        )

            if confirmed is None:
                if cand_idx.size > 1:
                    print(
                        "Multiple 4,4'-DDE mz1 candidates; using best m/z match "
                        "(no confirmation)"
                    )
                else:
                    print(
                        "Single 4,4'-DDE mz1 candidate; using best m/z match"
                    )

            print("Shift = {} seconds based on 4,4'-DDE ".format(shift))

    time_array = time_array - float(shift)

    for i in tqdm(range(len(mz_array)),disable=not show_progress):
        dmz = mz_array[i] - lib_mz_array
        dt = time_array[i] - lib_time_array
        distances = dmz**2 + 1e-7 * (dt**2)

        rel_mz = np.abs(dmz) / lib_mz_array
        rel_time = np.abs(dt) / lib_time_array
        matched_indices = np.where(
            (rel_mz < mz_diff) & (rel_time < time_diff)
        )[0]

        if matched_indices.size == 0:
            annotations[i] = None
            notes[i] = None
            candidates_all.append(None)
            continue

        match_distances = distances[matched_indices]
        best_idx_pos = int(np.argmin(match_distances))
        best_idx = matched_indices[best_idx_pos]

        annotations[i] = lib_gc.iloc[best_idx]["Name"]
        notes[i] = lib_gc.iloc[best_idx]["note"]

        candidates: list[dict] = []
        for pos, idx in enumerate(matched_indices):
            candidates.append(
                {
                    "name": lib_gc.iloc[idx]["Name"],
                    "note": lib_gc.iloc[idx]["note"],
                    "ppm": float(rel_mz[idx] * 1e6),
                    "dist": float(match_distances[pos]),
                }
            )

        candidates_all.append(sorted(candidates, key=lambda c: c["dist"]))

    # Унификация имени колонки: 'annotation' вместо 'annotations' (пункт 9)
    features["annotation"] = annotations
    features["notes"] = notes
    features["candidates"] = candidates_all

    feat_nonnull = features.dropna(subset=["annotation"])
    feat_nonnull = feat_nonnull.sort_values(by=["annotation", rt_col])

    data_gc = feat_nonnull.reset_index().drop("index", axis=1)
    best_group_indices: list[int] = []

    for chemical in data_gc["annotation"].unique():
        chem_data = data_gc[data_gc["annotation"] == chemical]
        groups: list[tuple[list[int], bool]] = []

        i = 0
        while i < len(chem_data):
            group_indices = [chem_data.index[i]]
            has_m0 = "mz0" in chem_data.iloc[i]["notes"]

            j = i + 1
            while (
                j < len(chem_data)
                and abs(chem_data.iloc[i][rt_col] - chem_data.iloc[j][rt_col])
                <= time_range
            ):
                group_indices.append(chem_data.index[j])
                if "mz0" in chem_data.iloc[j]["notes"]:
                    has_m0 = True
                j += 1

            if len(group_indices) >= ngroup:
                groups.append((group_indices, has_m0))

            i = j

        if groups:
            if any(has_m0 for _, has_m0 in groups):
                groups = [grp for grp in groups if grp[1]]

            max_size = max(len(grp[0]) for grp in groups)
            groups = [grp for grp in groups if len(grp[0]) == max_size]

            for grp, _ in groups:
                best_group_indices.extend(grp)

    best_group_indices = sorted(set(best_group_indices))

    data_gc = (
        data_gc.loc[best_group_indices]
        .sort_values(by=["annotation", rt_col])
        .reset_index()
        .drop("index", axis=1)
    )

    total_candidates = sum(
        len(c) for c in data_gc["candidates"] if isinstance(c, list)
    )
    print(
        f"[GC] Annotated {len(data_gc)} features; unique chemicals: "
        f"{data_gc['annotation'].nunique()}; total candidate entries: {total_candidates}"
    )

    if save is not None:
        data_gc.to_csv(save, index=False)

    return data_gc
