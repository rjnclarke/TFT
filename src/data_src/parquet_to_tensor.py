
"""Build TFT-ready zipped NPZ shards from parquet and run consistency checks.

Key semantics
-------------
Each NPZ stores full per-series arrays:

    historic_numeric: [num_series, num_days, 2] where features are [sales, price]
    known_numeric   : [num_series, num_days, known_features]
    sales           : [num_series, num_days]
    day_nums        : [num_series, num_days]
    static_context  : [num_series, static_cat_features]

Index arrays contain rows:

    [series_idx, t_idx]

Where:

    context = t_idx - CONTEXT : t_idx
    target  = t_idx           : t_idx + TARGET

So one sample is an 84-day block:

    56 context days + 28 target days

Split semantics
---------------
- Test is exactly one final full 84-day block per product.
- If data ends after an incomplete week, hanging days are dropped.
- Valid is exactly the 84-day block before test.
- Train is weekly rolling 84-day windows before valid.

Example with week_start_day=1:

    Train sample 1:
        context 1–56
        target  57–84

    Train sample 2:
        context 8–63
        target  64–91

    Test:
        final complete 84-day block.
"""

import argparse
import gc
import io
import json
import random
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


class ForecastConfig:
    CONTEXT = 56
    TARGET = 28
    PERIOD = CONTEXT + TARGET
    TOTAL = 1941

    STATIC_CAT_COLS = [
        "product_id",
        "department_id",
        "category_id",
        "store_id",
        "state_code",
    ]

    HIST_NUMERIC_COLS = ["sales", "price"]

    TIME_NUMERIC_COLS = [
        "dist_week_start",
        "dist_month_start",
        "woy_sin",
        "woy_cos",
        "time_trend",
    ]

    SNAP_COLS = [
        "snap_california",
        "snap_texas",
        "snap_wisconsin",
    ]

    SNAP_START_WINDOW = (3, 10)
    SNAP_END_WINDOW = (3, 3)

    EVENT_WINDOWS = {
        "Christmas": (42, 14),
        "Thanksgiving": (21, 7),
        "Halloween": (14, 2),
        "NewYear": (7, 3),
        "ValentinesDay": (10, 1),
        "Mother's day": (14, 2),
        "Father's day": (7, 1),
        "Easter": (14, 3),
        "OrthodoxEaster": (14, 3),
        "LentStart": (0, 14),
        "LentWeek2": (0, 7),
        "Ramadan starts": (0, 28),
        "Eid al-Fitr": (7, 3),
        "EidAlAdha": (7, 3),
        "Chanukah End": (7, 3),
        "Pesach End": (3, 3),
        "Purim End": (3, 2),
        "OrthodoxChristmas": (7, 3),
        "SuperBowl": (7, 1),
        "IndependenceDay": (14, 2),
        "MemorialDay": (7, 2),
        "LaborDay": (7, 2),
        "PresidentsDay": (7, 1),
        "MartinLutherKingDay": (7, 1),
        "ColumbusDay": (7, 1),
        "VeteransDay": (3, 1),
        "StPatricksDay": (7, 1),
        "Cinco De Mayo": (7, 1),
        "NBAFinalsStart": (3, 7),
        "NBAFinalsEnd": (3, 1),
    }


def add_day_num(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric day_num exists."""
    if "day_num" not in df.columns:
        df = df.copy()
        df["day_num"] = df["day_id"].str.replace("d_", "", regex=False).astype(np.int32)
    return df


def is_week_start(d: int, week_start_day: int = 1) -> bool:
    """
    True when day d is aligned to the chosen week start.

    week_start_day=1 gives starts:
        d_1, d_8, d_15, ...

    This matches:
        context 1–56 -> target 57–84
        context 8–63 -> target 64–91

    If using M5 real Monday alignment, you may want week_start_day=3 because
    M5 d_1 is Saturday and d_3 is Monday.
    """
    return ((d - week_start_day) % 7) == 0


def compute_split_boundaries(max_day: int, period: int, week_start_day: int = 1):
    """
    Compute train/valid/test full-window boundaries.

    Test is the last full 84-day block ending on a complete week end.
    If max_day ends mid-week, hanging days after the last week end are dropped.

    week_start_day=1:
        week starts: d_1, d_8, d_15, ...
        week ends  : d_7, d_14, d_21, ...

    For max_day=1941 and week_start_day=1:
        effective_end = 1939
        dropped days  = 2
        test          = 1856–1939
        valid         = 1772–1855
        train         = windows ending <=1771
    """

    week_end_offset = 6
    hanging_days = (max_day - week_start_day - week_end_offset) % 7
    effective_end = max_day - hanging_days

    test_end = effective_end
    test_start = test_end - period + 1

    valid_end = test_start - 1
    valid_start = valid_end - period + 1

    train_end = valid_start - 1

    return {
        "max_day": int(max_day),
        "week_start_day": int(week_start_day),
        "effective_end": int(effective_end),
        "dropped_hanging_days": int(hanging_days),
        "train_end": int(train_end),
        "valid_start": int(valid_start),
        "valid_end": int(valid_end),
        "test_start": int(test_start),
        "test_end": int(test_end),
    }


def make_calendar_features(df: pd.DataFrame):
    base_cols = ["day_id", "date", "event_name_primary", "event_name_secondary"] + ForecastConfig.SNAP_COLS

    cal = df[base_cols].drop_duplicates("day_id").copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["day_num"] = cal["day_id"].str.replace("d_", "", regex=False).astype(np.int32)
    cal = cal.sort_values("day_num").reset_index(drop=True)

    n_days = len(cal)
    min_day = int(cal["day_num"].min())
    max_day = int(cal["day_num"].max())

    cal["dist_week_start"] = ((cal["day_num"] - 1) % 7) / 6.0
    cal["dist_month_start"] = (cal["date"].dt.day - 1) / (cal["date"].dt.days_in_month - 1)

    week_angle = 2 * np.pi * (cal["date"].dt.isocalendar().week.astype(int) - 1) / 52
    cal["woy_sin"] = np.sin(week_angle)
    cal["woy_cos"] = np.cos(week_angle)
    cal["time_trend"] = (cal["day_num"] - 1) / (max_day - 1)

    event_cols = []

    for name, (before_w, after_w) in ForecastConfig.EVENT_WINDOWS.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        before_col = f"event_before_{safe_name}"
        after_col = f"event_after_{safe_name}"

        before_arr = np.zeros(n_days, np.float32)
        after_arr = np.zeros(n_days, np.float32)

        occ = cal[
            (cal["event_name_primary"] == name)
            | (cal["event_name_secondary"] == name)
        ]["day_num"].to_numpy()

        for t in occ:
            start, end = max(min_day, int(t) - before_w), int(t) - 1
            if end >= start:
                idx = slice(start - min_day, end - min_day + 1)
                before_arr[idx] = np.linspace(1, 0, end - start + 1)

            start, end = int(t), min(max_day, int(t) + after_w)
            if end >= start:
                idx = slice(start - min_day, end - min_day + 1)
                after_arr[idx] = np.linspace(1, 0, end - start + 1)

        cal[before_col] = before_arr
        cal[after_col] = after_arr
        event_cols += [before_col, after_col]

    for name in ForecastConfig.SNAP_COLS:
        before_w = ForecastConfig.SNAP_START_WINDOW[0]
        after_w = ForecastConfig.SNAP_END_WINDOW[1]

        before_col = f"{name}_before"
        after_col = f"{name}_after"

        before_arr = np.zeros(n_days, np.float32)
        after_arr = np.zeros(n_days, np.float32)

        active = cal.loc[cal[name] == 1, "day_num"].to_numpy()

        for t in active:
            start, end = max(min_day, int(t) - before_w), int(t) - 1
            if end >= start:
                idx = slice(start - min_day, end - min_day + 1)
                before_arr[idx] = np.linspace(1, 0, end - start + 1)

            start, end = int(t), min(max_day, int(t) + after_w)
            if end >= start:
                idx = slice(start - min_day, end - min_day + 1)
                after_arr[idx] = np.linspace(1, 0, end - start + 1)

        cal[before_col] = before_arr
        cal[after_col] = after_arr

    for c in [x for x in cal.columns if x.startswith(("event_before_", "event_after_", "snap_"))]:
        cal[c] = cal[c].astype(np.float32)

    known_cols = (
        ["day_num"]
        + ForecastConfig.TIME_NUMERIC_COLS
        + ForecastConfig.SNAP_COLS
        + [f"{c}_before" for c in ForecastConfig.SNAP_COLS]
        + [f"{c}_after" for c in ForecastConfig.SNAP_COLS]
        + event_cols
    )

    dup_days = int(cal["day_id"].duplicated().sum())
    missing_days = int((cal["day_num"].diff().fillna(1) != 1).sum())

    print(f"[Calendar check] Duplicated days: {dup_days}, Non-consecutive jumps: {missing_days}, Shape: {cal.shape}")
    print(cal.head(3)[["day_id", "day_num"]])

    return cal[["day_id"] + known_cols], known_cols


def build_cat_maps(data_dir: Path):
    unique_values = {c: set() for c in ForecastConfig.STATIC_CAT_COLS}

    for path in sorted(data_dir.glob("monster_*.parquet")):
        df = pd.read_parquet(path, columns=ForecastConfig.STATIC_CAT_COLS)

        for col in ForecastConfig.STATIC_CAT_COLS:
            unique_values[col].update(df[col].dropna().unique())

        del df
        gc.collect()

    cat_maps = {
        col: {v: i for i, v in enumerate(sorted(vals))}
        for col, vals in unique_values.items()
    }

    cardinalities = {col: len(vals) for col, vals in unique_values.items()}

    print("[Categorical cardinalities]")
    print(json.dumps(cardinalities, indent=2))

    return cat_maps


def print_split_boundary_report(boundaries: dict):
    print("[Split boundaries based on full 84-day windows]")
    print(f"  max_day              : {boundaries['max_day']}")
    print(f"  week_start_day       : {boundaries['week_start_day']}")
    print(f"  effective week end   : {boundaries['effective_end']}")
    print(f"  dropped hanging days : {boundaries['dropped_hanging_days']}")
    print(f"  train full windows   : window_end <= {boundaries['train_end']}")
    print(f"  valid full window    : {boundaries['valid_start']}–{boundaries['valid_end']}")
    print(f"  test full window     : {boundaries['test_start']}–{boundaries['test_end']}")


def process_shards(
    DATA_DIR: Path,
    OUTPUT_TFT_DIR: Path,
    DEBUG_MODE: bool = True,
    week_start_day: int = 1,
):
    store_files = sorted(DATA_DIR.glob("monster_*.parquet"))

    if DEBUG_MODE:
        store_files = store_files[:1]

    if not store_files:
        raise FileNotFoundError(f"No monster_*.parquet files found in {DATA_DIR}")

    OUTPUT_TFT_DIR.mkdir(parents=True, exist_ok=True)

    cat_maps = build_cat_maps(DATA_DIR)

    calendar_path = None
    for f in ["calendar.parquet", "m5_calendar.parquet"]:
        p = DATA_DIR / f
        if p.exists():
            calendar_path = p
            break

    if calendar_path is not None:
        print(f"[Calendar] Using {calendar_path.name}")
        calendar_df = pd.read_parquet(calendar_path)
    else:
        print("[Calendar] Deriving from first store shard.")
        sample_path = sorted(DATA_DIR.glob("monster_*.parquet"))[0]
        tmp = pd.read_parquet(sample_path)[
            [
                "day_id",
                "date",
                "event_name_primary",
                "event_name_secondary",
                "snap_california",
                "snap_texas",
                "snap_wisconsin",
            ]
        ].drop_duplicates("day_id")
        tmp = add_day_num(tmp).sort_values("day_num").reset_index(drop=True)
        calendar_df = tmp

    cal_features, known_col_list = make_calendar_features(calendar_df)
    print(f"[Calendar built] {len(cal_features)} days, {len(known_col_list)} known temporal features")

    for path in store_files:
        print(f"\n🚀 Processing {path.name}")

        df = pd.read_parquet(path, columns=None)
        df = add_day_num(df)
        df = df.sort_values(["series_id", "day_num"]).reset_index(drop=True)

        if DEBUG_MODE:
            print("[Debug] Processing single full shard")

        orig_price = df[["series_id", "day_id", "price"]].copy()

        cal = cal_features.drop_duplicates("day_id")[["day_id"] + known_col_list]
        df = df.merge(cal, on="day_id", how="left", validate="many_to_one")

        df.drop(columns=["price"], errors="ignore", inplace=True)
        df = df.merge(orig_price, on=["series_id", "day_id"], how="left", validate="one_to_one")

        if "day_num_x" in df.columns and "day_num_y" in df.columns:
            df["day_num"] = df["day_num_x"]
            df.drop(columns=["day_num_x", "day_num_y"], inplace=True)
        elif "day_num_x" in df.columns:
            df["day_num"] = df["day_num_x"]
            df.drop(columns=["day_num_x"], inplace=True)
        elif "day_num_y" in df.columns:
            df["day_num"] = df["day_num_y"]
            df.drop(columns=["day_num_y"], inplace=True)

        df["day_num"] = df["day_num"].astype(np.int32)
        df = df.sort_values(["series_id", "day_num"]).reset_index(drop=True)

        df["price"] = df["price"].astype(np.float32)
        df["price"] = df.groupby("series_id", sort=False)["price"].ffill().fillna(0).astype(np.float32)

        for c in ForecastConfig.SNAP_COLS:
            x, y = f"{c}_x", f"{c}_y"

            if x in df.columns and y in df.columns:
                df[c] = df[y]
                df.drop(columns=[x, y], inplace=True)
            elif x in df.columns:
                df[c] = df[x]
                df.drop(columns=[x], inplace=True)
            elif y in df.columns:
                df[c] = df[y]
                df.drop(columns=[y], inplace=True)

        for col in ForecastConfig.STATIC_CAT_COLS:
            df[f"{col}_idx"] = df[col].map(cat_maps[col]).astype(np.int16)

        if DEBUG_MODE:
            sample_sid = df.series_id.iloc[0]
            chk = df.loc[df.series_id == sample_sid, ["day_id", "day_num", "sales", "price"]].head(12)
            print(f"[Debug] First chronological rows for {sample_sid}:\n{chk}\n")

        groups = df.groupby("series_id", sort=False)

        series_id_list = []
        static_cats = []
        hist_numeric = []
        known_numeric = []
        sales = []
        day_nums = []

        for sid, g in groups:
            g = g.sort_values("day_num").reset_index(drop=True)

            jumps = np.diff(g["day_num"].to_numpy())
            if len(jumps) and not np.all(jumps == 1):
                bad = np.where(jumps != 1)[0][:5]
                print(f"[Warn] Non-consecutive days for {sid}; bad positions: {bad}")

            series_id_list.append(sid)
            static_cats.append(g[[f"{c}_idx" for c in ForecastConfig.STATIC_CAT_COLS]].iloc[0].to_numpy(np.int16))
            hist_numeric.append(g[ForecastConfig.HIST_NUMERIC_COLS].to_numpy(np.float32))
            known_numeric.append(g[known_col_list].to_numpy(np.float32))
            sales.append(g["sales"].to_numpy(np.float32))
            day_nums.append(g["day_num"].to_numpy(np.int32))

        static_cats = np.stack(static_cats)
        hist_numeric = np.stack(hist_numeric)
        known_numeric = np.stack(known_numeric)
        sales = np.stack(sales)
        day_nums_arr = np.stack(day_nums)
        series_ids_arr = np.array(series_id_list)

        ctx = ForecastConfig.CONTEXT
        tgt = ForecastConfig.TARGET
        period = ForecastConfig.PERIOD

        max_day_actual = int(max(max(d) for d in day_nums))

        boundaries = compute_split_boundaries(
            max_day=max_day_actual,
            period=period,
            week_start_day=week_start_day,
        )

        train_end = boundaries["train_end"]
        valid_start = boundaries["valid_start"]
        valid_end = boundaries["valid_end"]
        test_start = boundaries["test_start"]
        test_end = boundaries["test_end"]

        print_split_boundary_report(boundaries)

        train_idx = []
        valid_idx = []
        test_idx = []

        for s, days in enumerate(day_nums):
            for t in range(ctx, len(days) - tgt + 1):
                window_start_day = int(days[t - ctx])
                forecast_start_day = int(days[t])
                target_end_day = int(days[t + tgt - 1])
                window_end_day = target_end_day

                # Full 84-day block must be consecutive.
                if (forecast_start_day - window_start_day) != ctx:
                    continue
                if (target_end_day - forecast_start_day) != (tgt - 1):
                    continue
                if (window_end_day - window_start_day) != (period - 1):
                    continue

                # Weekly rolling alignment by full-window start.
                if not is_week_start(window_start_day, week_start_day=week_start_day):
                    continue

                # Exactly one test block per product.
                if window_start_day == test_start and window_end_day == test_end:
                    test_idx.append([s, t])

                # Exactly one valid block per product.
                elif window_start_day == valid_start and window_end_day == valid_end:
                    valid_idx.append([s, t])

                # Train is weekly rolling before validation.
                elif window_end_day <= train_end:
                    train_idx.append([s, t])

        train_idx = np.array(train_idx, np.int32)
        valid_idx = np.array(valid_idx, np.int32)
        test_idx = np.array(test_idx, np.int32)

        print("[Window counts]")
        print(f"  train = {len(train_idx)}")
        print(f"  valid = {len(valid_idx)}")
        print(f"  test  = {len(test_idx)}")

        expected_eval = len(series_id_list)
        if len(valid_idx) != expected_eval:
            print(f"[Warn] Expected valid windows = {expected_eval}, got {len(valid_idx)}")
        if len(test_idx) != expected_eval:
            print(f"[Warn] Expected test windows = {expected_eval}, got {len(test_idx)}")

        buffer = io.BytesIO()

        np.savez(
            buffer,
            static_context=static_cats,
            historic_numeric=hist_numeric,
            known_numeric=known_numeric,
            sales=sales,
            day_nums=day_nums_arr,
            series_ids=series_ids_arr,
            train_index=train_idx,
            valid_index=valid_idx,
            test_index=test_idx,
            known_col_list=np.array(known_col_list),
            hist_numeric_col_list=np.array(ForecastConfig.HIST_NUMERIC_COLS),
            static_cat_col_list=np.array(ForecastConfig.STATIC_CAT_COLS),
            split_boundaries_json=np.array(json.dumps(boundaries)),
        )

        buffer.seek(0)

        out_path = (
            OUTPUT_TFT_DIR / f"{path.stem}_DEBUG.zip"
            if DEBUG_MODE
            else OUTPUT_TFT_DIR / f"{path.stem}.zip"
        )

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{path.stem}.npz", buffer.read())

        meta = {
            "context": ctx,
            "target": tgt,
            "period": period,
            "total_days_config": ForecastConfig.TOTAL,
            "num_products": int(static_cats.shape[0]),
            "num_days_per_series": int(hist_numeric.shape[1]),
            "num_historic_numeric_features": int(hist_numeric.shape[-1]),
            "num_known_temporal_features": int(known_numeric.shape[-1]),
            "split_boundaries": boundaries,
            "index_semantics": (
                "[series_idx, t_idx], where t_idx is forecast start position; "
                "context is t_idx-CONTEXT:t_idx; target is t_idx:t_idx+TARGET"
            ),
        }

        meta_path = out_path.with_name(f"{path.stem}_metadata.json")

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✅ Complete: {path.stem}")
        print(f"   series={static_cats.shape[0]} days={hist_numeric.shape[1]} known_features={known_numeric.shape[-1]}")
        print(f"   wrote {out_path}")
        print(f"   wrote {meta_path}")

        del df, static_cats, hist_numeric, known_numeric, sales, day_nums, groups
        gc.collect()


def load_npz_from_zip(zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        npz_names = [n for n in zf.namelist() if n.endswith(".npz")]

        if not npz_names:
            raise FileNotFoundError(f"No .npz file found inside {zip_path}")

        with zf.open(npz_names[0]) as f:
            return np.load(io.BytesIO(f.read()), allow_pickle=False)


def find_zip(output_dir: Path, zip_name: Optional[str] = None):
    if zip_name:
        p = output_dir / zip_name

        if not p.exists():
            p = Path(zip_name)

        if not p.exists():
            raise FileNotFoundError(f"Zip not found: {zip_name}")

        return p

    zips = sorted(output_dir.glob("*DEBUG*.zip"))

    if not zips:
        zips = sorted(output_dir.glob("monster_*.zip"))

    if not zips:
        raise FileNotFoundError(f"No shard zip files found in {output_dir}")

    return zips[0]


def max_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    if len(a) == 0 or len(b) == 0:
        return np.nan

    return float(np.nanmax(np.abs(a - b)))


def arr_to_string(a, max_items: Optional[int] = None):
    a = np.asarray(a)

    if max_items is not None and len(a) > max_items:
        a = a[:max_items]

    return np.array2string(
        a,
        precision=3,
        separator=", ",
        suppress_small=False,
        max_line_width=160,
    )


def print_sample_values(
    split: str,
    parquet_context: pd.DataFrame,
    parquet_target: pd.DataFrame,
    tensor_context: np.ndarray,
    tensor_target: np.ndarray,
):
    print(f"\n[{split.upper()} VALUES]")

    print("  Context parquet rows, first 5:")
    print(parquet_context.head(5).to_string(index=False))

    print("  Context parquet rows, last 5:")
    print(parquet_context.tail(5).to_string(index=False))

    print("  Target parquet rows, all 28:")
    print(parquet_target.to_string(index=False))

    print("  Tensor context sales, first 10:")
    print("   ", arr_to_string(tensor_context[:10, 0]))

    print("  Tensor context sales, last 10:")
    print("   ", arr_to_string(tensor_context[-10:, 0]))

    print("  Tensor target sales, all 28:")
    print("   ", arr_to_string(tensor_target[:, 0]))

    print("  Tensor context price, first 10:")
    print("   ", arr_to_string(tensor_context[:10, 1]))

    print("  Tensor context price, last 10:")
    print("   ", arr_to_string(tensor_context[-10:, 1]))

    print("  Tensor target price, all 28:")
    print("   ", arr_to_string(tensor_target[:, 1]))


def run_consistency_check(
    DATA_DIR: Path,
    OUTPUT_TFT_DIR: Path,
    DEBUG_MODE: bool = True,
    zip_name: Optional[str] = None,
    sample_series: int = 300,
    seed: int = 123,
    splits_to_check: Optional[List[str]] = None,
    strict: bool = False,
):
    random.seed(seed)
    np.random.seed(seed)

    if splits_to_check is None:
        splits_to_check = ["train", "valid", "test"]

    sample_zip = find_zip(OUTPUT_TFT_DIR, zip_name=zip_name)

    print(f"🔍 Checking shard: {sample_zip.name}")

    data = load_npz_from_zip(sample_zip)

    base_name = sample_zip.stem.replace("_DEBUG", "")
    store_match = list(DATA_DIR.glob(base_name + ".parquet"))

    if not store_match:
        raise FileNotFoundError(f"No parquet match found for {base_name}. Expected {base_name}.parquet in {DATA_DIR}")

    store_parquet = store_match[0]

    print(f"Reading original parquet: {store_parquet.name}")

    df_full = pd.read_parquet(store_parquet)
    df_full = add_day_num(df_full)
    df_full = df_full.sort_values(["series_id", "day_num"]).reset_index(drop=True)

    # Apply the same price fill used during processing.
    df_full["price_filled"] = (
        df_full.groupby("series_id", sort=False)["price"]
        .ffill()
        .fillna(0)
        .astype(np.float32)
    )

    series_ids_parquet = df_full["series_id"].drop_duplicates().to_numpy()

    if "series_ids" in data.files:
        series_ids_npz = data["series_ids"].astype(str)
    else:
        print("[Warn] npz has no series_ids array; assuming parquet order.")
        series_ids_npz = series_ids_parquet.astype(str)

    print(f"Total series in parquet: {len(series_ids_parquet)}")
    print(f"Total series in tensor : {len(series_ids_npz)}")
    print(f"[Range] Parquet days {df_full['day_num'].min()}–{df_full['day_num'].max()}")

    print("[NPZ arrays]")
    for k in data.files:
        arr = data[k]
        print(f"  {k:24s} shape={arr.shape} dtype={arr.dtype}")

    if "split_boundaries_json" in data.files:
        try:
            boundaries = json.loads(str(data["split_boundaries_json"]))
            print_split_boundary_report(boundaries)
        except Exception as e:
            print(f"[Warn] Could not read split boundaries from npz: {e}")

    ctx = ForecastConfig.CONTEXT
    tgt = ForecastConfig.TARGET
    period = ForecastConfig.PERIOD

    if data["historic_numeric"].ndim != 3:
        raise ValueError("historic_numeric should have shape [series, days, features].")

    if data["historic_numeric"].shape[1] <= ctx:
        print(
            "\n[Problem] historic_numeric only has CONTEXT or fewer timesteps. "
            "This looks like the old broken export where only the first context window was saved."
        )

        if strict:
            raise AssertionError("Old/broken tensor format detected.")

    index_by_split = {
        "train": data["train_index"] if "train_index" in data.files else np.empty((0, 2), np.int32),
        "valid": data["valid_index"] if "valid_index" in data.files else np.empty((0, 2), np.int32),
        "test": data["test_index"] if "test_index" in data.files else np.empty((0, 2), np.int32),
    }

    print("[Index counts]")
    for split_name, idx in index_by_split.items():
        print(f"  {split_name:6s}: {len(idx)}")

    expected_eval = len(series_ids_npz)

    if len(index_by_split["valid"]) != expected_eval:
        print(f"[Warn] Expected valid windows = {expected_eval}, got {len(index_by_split['valid'])}")

    if len(index_by_split["test"]) != expected_eval:
        print(f"[Warn] Expected test windows = {expected_eval}, got {len(index_by_split['test'])}")

    samples = {}

    for split in splits_to_check:
        idx = index_by_split.get(split, np.empty((0, 2), np.int32))

        if idx.size == 0:
            print(f"[Skip] No {split} indices in this shard")
            continue

        # Deterministic, readable examples:
        # - train: first training point, usually product 0, earliest rolling window
        # - valid: first valid point
        # - test : first test point
        candidate = None

        if DEBUG_MODE:
            max_s = min(sample_series, len(series_ids_npz))

            for row in idx:
                s, t = int(row[0]), int(row[1])
                if s < max_s:
                    candidate = (s, t)
                    break

        if candidate is None:
            s, t = random.choice(idx).tolist()
            candidate = (int(s), int(t))

        samples[split] = candidate

        sid = series_ids_npz[candidate[0]]
        print(f"[Sample selected] {split:6s} -> series_idx={candidate[0]} sid={sid} t_idx={candidate[1]}")

    if not samples:
        print("⚙️ No sample windows found; skipping comparisons.")
        return

    failures = 0

    def compare_source(split: str, s_idx: int, t_idx: int):
        nonlocal failures

        sid = str(series_ids_npz[s_idx])

        g = (
            df_full[df_full["series_id"].astype(str) == sid]
            .sort_values("day_num")
            .reset_index(drop=True)
        )

        if g.empty:
            print(f"[FAIL] {split}: no parquet rows for sid={sid}")
            failures += 1
            return

        if t_idx < ctx:
            print(f"[FAIL] {split}: t_idx={t_idx} is less than CONTEXT={ctx}")
            failures += 1
            return

        if t_idx + tgt > len(g):
            print(f"[FAIL] {split}: target exceeds parquet length: t_idx+tgt={t_idx + tgt}, len={len(g)}")
            failures += 1
            return

        parquet_context = (
            g.iloc[t_idx - ctx:t_idx][["day_num", "sales", "price_filled"]]
            .rename(columns={"price_filled": "price"})
            .reset_index(drop=True)
        )

        parquet_target = (
            g.iloc[t_idx:t_idx + tgt][["day_num", "sales", "price_filled"]]
            .rename(columns={"price_filled": "price"})
            .reset_index(drop=True)
        )

        tensor_context = data["historic_numeric"][s_idx, t_idx - ctx:t_idx, :]
        tensor_target = data["historic_numeric"][s_idx, t_idx:t_idx + tgt, :]

        tensor_day_nums_context = None
        tensor_day_nums_target = None

        if "day_nums" in data.files:
            tensor_day_nums_context = data["day_nums"][s_idx, t_idx - ctx:t_idx]
            tensor_day_nums_target = data["day_nums"][s_idx, t_idx:t_idx + tgt]

        expected_context_days = parquet_context["day_num"].to_numpy(np.int32)
        expected_target_days = parquet_target["day_num"].to_numpy(np.int32)

        window_start_day = int(expected_context_days[0])
        context_end_day = int(expected_context_days[-1])
        forecast_start_day = int(expected_target_days[0])
        window_end_day = int(expected_target_days[-1])

        context_days_ok = True
        target_days_ok = True

        if tensor_day_nums_context is not None:
            context_days_ok = np.array_equal(expected_context_days, tensor_day_nums_context)

        if tensor_day_nums_target is not None:
            target_days_ok = np.array_equal(expected_target_days, tensor_day_nums_target)

        sales_context_diff = max_abs_diff(
            parquet_context["sales"].to_numpy(np.float32),
            tensor_context[:, 0],
        )

        price_context_diff = max_abs_diff(
            parquet_context["price"].to_numpy(np.float32),
            tensor_context[:, 1],
        )

        sales_target_diff = max_abs_diff(
            parquet_target["sales"].to_numpy(np.float32),
            tensor_target[:, 0],
        )

        price_target_diff = max_abs_diff(
            parquet_target["price"].to_numpy(np.float32),
            tensor_target[:, 1],
        )

        full_window_length = window_end_day - window_start_day + 1

        print(f"\n[{split.upper()} POINT]")
        print(f"  series_id          : {sid}")
        print(f"  series_idx         : {s_idx}")
        print(f"  t_idx              : {t_idx}")
        print(f"  full window days   : {window_start_day}–{window_end_day}")
        print(f"  full window length : {full_window_length}")
        print(f"  context days       : {window_start_day}–{context_end_day}")
        print(f"  target days        : {forecast_start_day}–{window_end_day}")
        print(f"  context length     : {len(parquet_context)}")
        print(f"  target length      : {len(parquet_target)}")
        print(f"  full period        : {period}")
        print(f"  context day nums   : {'OK' if context_days_ok else 'MISMATCH'}")
        print(f"  target day nums    : {'OK' if target_days_ok else 'MISMATCH'}")
        print(f"  Max |Δ ctx sales|  : {sales_context_diff:.6f}")
        print(f"  Max |Δ ctx price|  : {price_context_diff:.6f}")
        print(f"  Max |Δ tgt sales|  : {sales_target_diff:.6f}")
        print(f"  Max |Δ tgt price|  : {price_target_diff:.6f}")

        ok = (
            full_window_length == period
            and len(parquet_context) == ctx
            and len(parquet_target) == tgt
            and context_days_ok
            and target_days_ok
            and sales_context_diff == 0
            and price_context_diff == 0
            and sales_target_diff == 0
            and price_target_diff == 0
        )

        print_sample_values(
            split=split,
            parquet_context=parquet_context,
            parquet_target=parquet_target,
            tensor_context=tensor_context,
            tensor_target=tensor_target,
        )

        if ok:
            print(f"\n[{split.upper()} RESULT] ✅ Tensor matches parquet exactly")
        else:
            failures += 1
            print(f"\n[{split.upper()} RESULT] ❌ Differences detected")

            if tensor_day_nums_context is not None:
                print("  Parquet context day nums:")
                print("   ", arr_to_string(expected_context_days))
                print("  Tensor context day nums:")
                print("   ", arr_to_string(tensor_day_nums_context))

            if tensor_day_nums_target is not None:
                print("  Parquet target day nums:")
                print("   ", arr_to_string(expected_target_days))
                print("  Tensor target day nums:")
                print("   ", arr_to_string(tensor_day_nums_target))

    for split, (s, t) in samples.items():
        compare_source(split, s, t)

    print("\n✅ Consistency check complete.")

    if failures:
        print(f"❌ Failures: {failures}")

        if strict:
            raise AssertionError(f"Consistency check failed with {failures} failure(s).")
    else:
        print("✅ All checked samples matched.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build TFT NPZ shards from parquet and/or run consistency checks."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing monster_*.parquet and optional calendar.parquet. If omitted, resolved to TFT repo root/monster_parts.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output zip/npz shards. If omitted, resolved to TFT repo root/monster_tensors_tft.",
    )

    parser.add_argument(
        "--mode",
        choices=["process", "check", "both"],
        default="both",
        help="Run processing, consistency check, or both.",
    )

    debug_group = parser.add_mutually_exclusive_group()

    debug_group.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Process/check single-shard debug mode.",
    )

    debug_group.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Process/check all shards.",
    )

    parser.set_defaults(debug=True)

    parser.add_argument(
        "--zip",
        dest="zip_name",
        default=None,
        help="Specific zip filename/path to check. Defaults to first DEBUG zip, then first monster_*.zip.",
    )

    parser.add_argument(
        "--sample-series",
        type=int,
        default=300,
        help="In debug check mode, prefer examples within this many initial series.",
    )

    parser.add_argument(
        "--splits",
        default="train,valid,test",
        help="Comma-separated splits to check, e.g. train,valid,test or test.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for sample selection.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if consistency mismatches are found.",
    )

    parser.add_argument(
        "--week-start-day",
        type=int,
        default=1,
        help=(
            "Day number treated as the first week start. "
            "Default 1 gives d_1,d_8,d_15 starts. "
            "For M5 Monday alignment, you may want 3."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    # Resolve defaults relative to the TFT package root so callers can run
    # from anywhere without moving files. TFT repo root is two levels above
    # this file: TFT/src/data_src -> parents[2]
    TFT_ROOT = Path(__file__).resolve().parents[2]

    DATA_DIR = args.data_dir or (TFT_ROOT / "monster_parts")
    OUTPUT_TFT_DIR = args.output_dir or (TFT_ROOT / "monster_tensors_tft")
    OUTPUT_TFT_DIR.mkdir(parents=True, exist_ok=True)

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    print("[Config]")
    print(f"  data_dir        : {DATA_DIR}")
    print(f"  output_dir      : {OUTPUT_TFT_DIR}")
    print(f"  mode            : {args.mode}")
    print(f"  debug           : {args.debug}")
    print(f"  splits          : {splits}")
    print(f"  week_start_day  : {args.week_start_day}")
    print(f"  context         : {ForecastConfig.CONTEXT}")
    print(f"  target          : {ForecastConfig.TARGET}")
    print(f"  period          : {ForecastConfig.PERIOD}")

    if args.mode in ("process", "both"):
        process_shards(
            DATA_DIR=DATA_DIR,
            OUTPUT_TFT_DIR=OUTPUT_TFT_DIR,
            DEBUG_MODE=args.debug,
            week_start_day=args.week_start_day,
        )

    if args.mode in ("check", "both"):
        run_consistency_check(
            DATA_DIR=DATA_DIR,
            OUTPUT_TFT_DIR=OUTPUT_TFT_DIR,
            DEBUG_MODE=args.debug,
            zip_name=args.zip_name,
            sample_series=args.sample_series,
            seed=args.seed,
            splits_to_check=splits,
            strict=args.strict,
        )


if __name__ == "__main__":
    main()