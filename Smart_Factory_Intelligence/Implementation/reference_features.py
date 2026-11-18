"""
Feature engineering of the BASE paper (Zdravevski et al., Sci Rep 2026, 16:20006), implemented with pandas/numpy
(the paper used Apache Spark). Three aligned Bosch files are needed: train_numeric.csv, train_date.csv,
train_categorical.csv (rows must be in the same order, as in the Kaggle files).

Numeric/date aggregation (per part): total measurements, stations visited, first/last visited station, per-station
visited indicator, per-station min/max measured value, total cycle time (max-min date), min/max measurement time and
the station at which they occur.
Nominal compression: supervised Weight of Evidence  WoE = ln(%non-events / %events) per category (+0.5 smoothing),
summarised as max/min/mean/std at 4 levels (feature, line, station, path) = 16 features.
Training-row WoE is computed OUT-OF-FOLD (5 folds) to avoid target leakage; test rows use all training rows.

Documented interpretation choices (the paper does not specify them exactly) are marked 'INTERP'.
"""
import re, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

COL_RE = re.compile(r"L(\d+)_S(\d+)_[FD](\d+)")
STATS = ["max", "min", "mean", "std"]


def parse_columns(cols):
    """Return DataFrame(col, line, station) parsed from Bosch column names."""
    rows = []
    for c in cols:
        m = COL_RE.fullmatch(c)
        if m:
            rows.append((c, int(m.group(1)), int(m.group(2))))
    return pd.DataFrame(rows, columns=["col", "line", "station"])


def _read(path, nrows=None, chunksize=100_000, **kw):
    return pd.read_csv(path, nrows=nrows, chunksize=chunksize, **kw)


def numeric_and_date_features(numeric_path, date_path, nrows=None):
    """Pass A: station-level numeric aggregates + cycle-time aggregates. Returns (features, y, ids, visited, stations)."""
    header_n = pd.read_csv(numeric_path, nrows=0).columns
    header_d = pd.read_csv(date_path, nrows=0).columns
    num_cols = parse_columns([c for c in header_n if c not in ("Id", "Response")])
    dat_cols = parse_columns([c for c in header_d if c != "Id"])
    stations = sorted(num_cols.station.unique())
    st_idx = {s: num_cols.loc[num_cols.station == s, "col"].tolist() for s in stations}
    d_station = dat_cols.station.values
    parts, ys, ids, vis_all = [], [], [], []
    for cn, cd in zip(_read(numeric_path, nrows, dtype=np.float32), _read(date_path, nrows, dtype=np.float32)):
        if not np.array_equal(cn["Id"].values, cd["Id"].values):
            raise ValueError("train_numeric.csv and train_date.csv rows are not aligned by Id.")
        ids.append(cn["Id"].values); ys.append(cn["Response"].values.astype(int))
        f = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            vis = np.zeros((len(cn), len(stations)), dtype=bool)
            for j, s in enumerate(stations):
                sub = cn[st_idx[s]].values
                v = ~np.isnan(sub).all(axis=1); vis[:, j] = v
                f[f"S{s}_visited"] = v.astype(np.int8)
                f[f"S{s}_min"] = np.nanmin(sub, axis=1); f[f"S{s}_max"] = np.nanmax(sub, axis=1)
            f["n_measurements"] = cn[num_cols.col.tolist()].notna().values.sum(axis=1)
            f["n_stations"] = vis.sum(axis=1)
            st_arr = np.array(stations)
            has = vis.any(axis=1)
            f["first_station"] = np.where(has, st_arr[vis.argmax(axis=1)], -1)
            f["last_station"] = np.where(has, st_arr[len(stations) - 1 - vis[:, ::-1].argmax(axis=1)], -1)
            D = cd[dat_cols.col.tolist()].values
            tmin, tmax = np.nanmin(D, axis=1), np.nanmax(D, axis=1)
            f["min_time"], f["max_time"], f["total_cycle_time"] = tmin, tmax, tmax - tmin
            allnan = np.isnan(D).all(axis=1)
            f["station_at_min_time"] = np.where(allnan, -1, d_station[np.nan_to_num(D, nan=np.inf).argmin(axis=1)])
            f["station_at_max_time"] = np.where(allnan, -1, d_station[np.nan_to_num(D, nan=-np.inf).argmax(axis=1)])
        parts.append(pd.DataFrame(f)); vis_all.append(vis)
    feats = pd.concat(parts, ignore_index=True)
    return feats, np.concatenate(ys), np.concatenate(ids), np.concatenate(vis_all), stations


def _long_chunk(chunk, y):
    """Stack a categorical chunk to (r, col, val); r = global row position."""
    vals = chunk.drop(columns="Id")
    long = vals.stack().reset_index()
    long.columns = ["r", "col", "val"]
    long["y"] = y[long["r"].values]
    return long


def fit_woe_counts(cat_path, y, train_mask, fold_of_row, nrows=None):
    """Pass B1: counts per (fold, column, category, class) on TRAINING rows only."""
    total = None
    for chunk in _read(cat_path, nrows, dtype=str):
        long = _long_chunk(chunk, y)
        r = long["r"].values
        keep = train_mask[r]
        long = long[keep].copy()
        long["fold"] = fold_of_row[long["r"].values]
        c = long.groupby(["fold", "col", "val", "y"]).size()
        total = c if total is None else total.add(c, fill_value=0)
    if total is None:
        raise ValueError("No categorical data read.")
    return total.unstack("y", fill_value=0).rename(columns={0: "non", 1: "ev"}).reindex(columns=["non", "ev"], fill_value=0)


def _woe_table(counts, n_non, n_ev):
    t = counts.copy()
    t["woe"] = np.log(((t["non"] + 0.5) / n_non) / ((t["ev"] + 0.5) / n_ev))   # formula as written in the paper
    return t["woe"]


def build_woe_tables(counts, y, train_mask, fold_of_row, n_folds):
    """Returns ({fold: woe Series} for out-of-fold use, full-train woe Series)."""
    by_fold = counts.groupby(level="fold")
    cat_total = counts.groupby(level=["col", "val"]).sum()
    n_non_all, n_ev_all = int(((y == 0) & train_mask).sum()), int(((y == 1) & train_mask).sum())
    full = _woe_table(cat_total, n_non_all, n_ev_all)
    oof = {}
    for f in range(n_folds):
        in_f = fold_of_row == f
        n_non_f, n_ev_f = int(((y == 0) & train_mask & in_f).sum()), int(((y == 1) & train_mask & in_f).sum())
        cf = counts.xs(f, level="fold") if f in counts.index.get_level_values("fold") else counts.iloc[0:0].droplevel("fold")
        rest = cat_total.sub(cf, fill_value=0).clip(lower=0)
        oof[f] = _woe_table(rest, n_non_all - n_non_f, n_ev_all - n_ev_f)
    return oof, full


def woe_features(cat_path, y, train_mask, fold_of_row, oof, full, visited, stations, nrows=None):
    """Pass B2: map categories to WoE and summarise at feature / line / station / path level (16 features).
    INTERP: feature level = stats over all of a part's WoE values; line level = stats over per-line mean WoE;
    station level = stats over per-station mean WoE; path level = stats over per-station mean WoE restricted to the
    stations on the part's numeric path (stations with at least one numeric measurement)."""
    st_pos = {s: i for i, s in enumerate(stations)}
    out = []
    for chunk in _read(cat_path, nrows, dtype=str):
        long = _long_chunk(chunk, y)
        meta = parse_columns(chunk.columns.drop("Id")).set_index("col")
        long["line"] = long["col"].map(meta["line"]); long["station"] = long["col"].map(meta["station"])
        r = long["r"].values
        woe = np.zeros(len(long))
        is_train = train_mask[r]
        keys = pd.MultiIndex.from_arrays([long["col"], long["val"]])
        # test rows: full-train WoE
        te = ~is_train
        if te.any():
            woe[te] = full.reindex(keys[te]).fillna(0.0).values
        for f, tab in oof.items():          # training rows: out-of-fold WoE
            m = is_train & (fold_of_row[r] == f)
            if m.any():
                woe[m] = tab.reindex(keys[m]).fillna(0.0).values
        long["woe"] = woe
        feats = {}
        g = long.groupby("r")["woe"].agg(STATS)
        for s in STATS: feats[f"woe_feature_{s}"] = g[s]
        for level in ("line", "station"):
            m = long.groupby(["r", level])["woe"].mean().reset_index()
            gg = m.groupby("r")["woe"].agg(STATS)
            for s in STATS: feats[f"woe_{level}_{s}"] = gg[s]
        st_mean = long.groupby(["r", "station"])["woe"].mean().reset_index()
        on_path = visited[st_mean["r"].values, st_mean["station"].map(st_pos).fillna(0).astype(int).values] \
                  & st_mean["station"].isin(st_pos).values
        gp = st_mean[on_path].groupby("r")["woe"].agg(STATS)
        for s in STATS: feats[f"woe_path_{s}"] = gp[s]
        fr = pd.DataFrame(feats)
        out.append(fr.reindex(chunk.index))
    res = pd.concat(out)
    return res.fillna(0.0).reset_index(drop=True)


def build_reference_features(numeric_path, date_path, cat_path, nrows=None, n_folds=5, test_size=0.20, seed=42):
    """Full feature pipeline. Returns X (engineered features), y, is_train mask."""
    feats, y, ids, visited, stations = numeric_and_date_features(numeric_path, date_path, nrows)
    n = len(y)
    idx = np.arange(n)
    tr_idx, te_idx = _split(idx, y, test_size, seed)
    train_mask = np.zeros(n, dtype=bool); train_mask[tr_idx] = True
    fold_of_row = np.full(n, -1)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(skf.split(tr_idx, y[tr_idx])):
        fold_of_row[tr_idx[va]] = f
    counts = fit_woe_counts(cat_path, y, train_mask, fold_of_row, nrows)
    oof, full = build_woe_tables(counts, y, train_mask, fold_of_row, n_folds)
    woe = woe_features(cat_path, y, train_mask, fold_of_row, oof, full, visited, stations, nrows)
    X = pd.concat([feats, woe], axis=1)
    return X, y, train_mask


def _split(idx, y, test_size, seed):
    from sklearn.model_selection import train_test_split
    return train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
