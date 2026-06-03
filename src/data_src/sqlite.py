"""
Build SQLite database for the TFT demo / inference app.

This database is app-facing and inference-facing, not training-facing.

It stores:
  - store / department / product hierarchy
  - raw categorical labels and model categorical index mappings
  - one final 84-day test block per product-store series
  - raw calendar/event/SNAP information for those days
  - raw price and model-filled price
  - enough information to reconstruct a valid TFT input later

App flow supported:
  select store
    -> show departments
    -> show products
    -> pull selected series test window
    -> rebuild TFT input
    -> run prediction
    -> store/display prediction

The DB stores the final test window only:
  56 context days + 28 target days = 84 days.

For all 10 stores this is approximately:
  10 * 3049 * 84 ~= 2.56 million rows,
which is fine for SQLite.
"""

import argparse
import gc
import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

CONTEXT = 56
TARGET = 28
PERIOD = CONTEXT + TARGET

STATIC_CAT_COLS = [
    "product_id",
    "department_id",
    "category_id",
    "store_id",
    "state_code",
]

REQUIRED_PARQUET_COLS = [
    "series_id",
    "day_id",
    "date",
    "sales",
    "price",
    "product_id",
    "department_id",
    "category_id",
    "store_id",
    "state_code",
    "event_name_primary",
    "event_name_secondary",
    "snap_california",
    "snap_texas",
    "snap_wisconsin",
]


# -----------------------------------------------------------------------------
# BASIC HELPERS
# -----------------------------------------------------------------------------

def add_day_num(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric day number from day_id, e.g. d_1856 -> 1856."""
    if "day_num" not in df.columns:
        df = df.copy()
        df["day_num"] = (
            df["day_id"]
            .str.replace("d_", "", regex=False)
            .astype(np.int32)
        )
    return df


def compute_test_boundaries(max_day: int, period: int = PERIOD, week_start_day: int = 1):
    """
    Compute final complete 84-day test block.

    week_start_day=1 means:
      week starts: d_1, d_8, d_15, ...
      week ends  : d_7, d_14, d_21, ...

    If the dataset ends after the last complete week, hanging days are dropped.

    Example:
      max_day=1941, week_start_day=1
      effective_end=1939
      test=1856-1939
      context=1856-1911
      target=1912-1939
    """

    week_end_offset = 6
    hanging_days = (max_day - week_start_day - week_end_offset) % 7
    effective_end = max_day - hanging_days

    test_end = effective_end
    test_start = test_end - period + 1

    context_start = test_start
    context_end = test_start + CONTEXT - 1

    target_start = context_end + 1
    target_end = test_end

    return {
        "max_day": int(max_day),
        "week_start_day": int(week_start_day),
        "effective_end": int(effective_end),
        "dropped_hanging_days": int(hanging_days),
        "test_start": int(test_start),
        "test_end": int(test_end),
        "context_start": int(context_start),
        "context_end": int(context_end),
        "target_start": int(target_start),
        "target_end": int(target_end),
    }


def chunked(rows: List[Tuple], chunk_size: int = 10_000):
    for i in range(0, len(rows), chunk_size):
        yield rows[i:i + chunk_size]


def executemany_chunked(
    conn: sqlite3.Connection,
    sql: str,
    rows: Iterable[Tuple],
    chunk_size: int = 10_000,
):
    rows = list(rows)
    if not rows:
        return

    cur = conn.cursor()
    for part in chunked(rows, chunk_size=chunk_size):
        cur.executemany(sql, part)


def clean_for_sqlite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert NaN/NaT to None and datetimes to strings where needed.
    """
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")

    out = out.where(pd.notnull(out), None)
    return out


# -----------------------------------------------------------------------------
# CATEGORY MAPS
# -----------------------------------------------------------------------------

def build_cat_maps(data_dir: Path):
    """
    Build categorical mappings exactly like the tensor export:
    sorted unique raw values across all monster_*.parquet files.
    """

    unique_values = {c: set() for c in STATIC_CAT_COLS}

    parquet_files = sorted(data_dir.glob("monster_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No monster_*.parquet files found in {data_dir}")

    print("[Categorical mappings] Scanning parquet shards...")

    for path in parquet_files:
        df = pd.read_parquet(path, columns=STATIC_CAT_COLS)

        for col in STATIC_CAT_COLS:
            unique_values[col].update(df[col].dropna().unique())

        del df
        gc.collect()

    cat_maps = {
        col: {v: i for i, v in enumerate(sorted(vals))}
        for col, vals in unique_values.items()
    }

    print("[Categorical cardinalities]")
    for col in STATIC_CAT_COLS:
        print(f"  {col:15s}: {len(cat_maps[col]):,}")

    return cat_maps


# -----------------------------------------------------------------------------
# SQLITE SCHEMA
# -----------------------------------------------------------------------------

def connect_db(db_path: Path, overwrite: bool = False):
    if overwrite and db_path.exists():
        print(f"[DB] Removing existing database: {db_path}")
        db_path.unlink()

    conn = sqlite3.connect(db_path)

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")

    return conn


def create_schema(conn: sqlite3.Connection):
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS stores (
            store_id        TEXT PRIMARY KEY,
            state_code      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS categories (
            category_id     TEXT PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS departments (
            department_id   TEXT PRIMARY KEY,
            category_id     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS products (
            product_id      TEXT PRIMARY KEY,
            department_id   TEXT NOT NULL,
            category_id     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cat_mappings (
            column_name     TEXT NOT NULL,
            raw_value       TEXT NOT NULL,
            encoded_idx     INTEGER NOT NULL,
            PRIMARY KEY (column_name, raw_value)
        );

        CREATE TABLE IF NOT EXISTS series (
            series_id           TEXT PRIMARY KEY,

            shard_name          TEXT NOT NULL,
            series_idx_in_shard INTEGER NOT NULL,

            product_id          TEXT NOT NULL,
            department_id       TEXT NOT NULL,
            category_id         TEXT NOT NULL,
            store_id            TEXT NOT NULL,
            state_code          TEXT NOT NULL,

            product_id_idx      INTEGER NOT NULL,
            department_id_idx   INTEGER NOT NULL,
            category_id_idx     INTEGER NOT NULL,
            store_id_idx        INTEGER NOT NULL,
            state_code_idx      INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS test_windows (
            series_id           TEXT PRIMARY KEY,

            shard_name          TEXT NOT NULL,
            store_id            TEXT NOT NULL,
            product_id          TEXT NOT NULL,
            department_id       TEXT NOT NULL,
            category_id         TEXT NOT NULL,
            state_code          TEXT NOT NULL,

            max_day             INTEGER NOT NULL,
            effective_end       INTEGER NOT NULL,
            dropped_hanging_days INTEGER NOT NULL,

            context_start_day   INTEGER NOT NULL,
            context_end_day     INTEGER NOT NULL,
            target_start_day    INTEGER NOT NULL,
            target_end_day      INTEGER NOT NULL,
            test_start_day      INTEGER NOT NULL,
            test_end_day        INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS test_observations (
            series_id               TEXT NOT NULL,
            shard_name              TEXT NOT NULL,

            store_id                TEXT NOT NULL,
            product_id              TEXT NOT NULL,
            department_id           TEXT NOT NULL,
            category_id             TEXT NOT NULL,
            state_code              TEXT NOT NULL,

            day_num                 INTEGER NOT NULL,
            day_id                  TEXT NOT NULL,
            date                    TEXT,

            sales                   REAL,
            price_raw               REAL,
            price_filled            REAL,

            event_name_primary      TEXT,
            event_name_secondary    TEXT,

            snap_california         INTEGER,
            snap_texas              INTEGER,
            snap_wisconsin          INTEGER,

            window_part             TEXT NOT NULL CHECK(window_part IN ('context', 'target')),

            PRIMARY KEY (series_id, day_num)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,

            series_id           TEXT NOT NULL,
            model_name          TEXT NOT NULL,

            day_num             INTEGER NOT NULL,
            day_id              TEXT,
            date                TEXT,

            y_true              REAL,
            y_pred              REAL,

            p10                 REAL,
            p50                 REAL,
            p90                 REAL,

            created_at          TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_series_store_dept
            ON series(store_id, department_id);

        CREATE INDEX IF NOT EXISTS idx_series_store_product
            ON series(store_id, product_id);

        CREATE INDEX IF NOT EXISTS idx_series_product
            ON series(product_id);

        CREATE INDEX IF NOT EXISTS idx_obs_series_day
            ON test_observations(series_id, day_num);

        CREATE INDEX IF NOT EXISTS idx_obs_store_dept_product
            ON test_observations(store_id, department_id, product_id);

        CREATE INDEX IF NOT EXISTS idx_obs_day
            ON test_observations(day_num);

        CREATE INDEX IF NOT EXISTS idx_pred_series_model
            ON predictions(series_id, model_name);

        CREATE VIEW IF NOT EXISTS v_store_departments AS
            SELECT DISTINCT
                store_id,
                state_code,
                department_id,
                category_id
            FROM series;

        CREATE VIEW IF NOT EXISTS v_store_department_products AS
            SELECT DISTINCT
                store_id,
                department_id,
                category_id,
                product_id,
                series_id
            FROM series;

        CREATE VIEW IF NOT EXISTS v_test_context AS
            SELECT *
            FROM test_observations
            WHERE window_part = 'context';

        CREATE VIEW IF NOT EXISTS v_test_target AS
            SELECT *
            FROM test_observations
            WHERE window_part = 'target';
        """
    )

    conn.commit()


def insert_cat_mappings(conn: sqlite3.Connection, cat_maps: dict):
    rows = []

    for col, mapping in cat_maps.items():
        for raw_value, encoded_idx in mapping.items():
            rows.append((col, str(raw_value), int(encoded_idx)))

    sql = """
        INSERT OR REPLACE INTO cat_mappings(
            column_name,
            raw_value,
            encoded_idx
        )
        VALUES (?, ?, ?)
    """

    with conn:
        executemany_chunked(conn, sql, rows, chunk_size=20_000)

    print(f"[DB] Inserted categorical mapping rows: {len(rows):,}")


# -----------------------------------------------------------------------------
# PROCESS ONE PARQUET SHARD
# -----------------------------------------------------------------------------

def process_parquet_file(
    conn: sqlite3.Connection,
    parquet_path: Path,
    cat_maps: dict,
    week_start_day: int = 1,
):
    print(f"\n[Shard] {parquet_path.name}")

    # Read only the columns needed for the app and inference reconstruction.
    df = pd.read_parquet(parquet_path, columns=REQUIRED_PARQUET_COLS)

    df = add_day_num(df)
    df = df.sort_values(["series_id", "day_num"]).reset_index(drop=True)

    # Preserve original price separately.
    df["price_raw"] = df["price"]

    # Match the tensor pipeline price handling.
    df["price_filled"] = (
        df.groupby("series_id", sort=False)["price"]
        .ffill()
        .fillna(0)
        .astype(np.float32)
    )

    # Make dates SQLite-friendly.
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    max_day = int(df["day_num"].max())

    boundaries = compute_test_boundaries(
        max_day=max_day,
        period=PERIOD,
        week_start_day=week_start_day,
    )

    print("[Test window]")
    print(f"  max_day               : {boundaries['max_day']}")
    print(f"  effective week end    : {boundaries['effective_end']}")
    print(f"  dropped hanging days  : {boundaries['dropped_hanging_days']}")
    print(f"  context               : {boundaries['context_start']}–{boundaries['context_end']}")
    print(f"  target                : {boundaries['target_start']}–{boundaries['target_end']}")
    print(f"  full test block       : {boundaries['test_start']}–{boundaries['test_end']}")

    shard_name = parquet_path.stem

    # Static series info, ordered exactly like tensor export:
    # sort by series_id, day_num, then groupby(sort=False).
    static = (
        df[
            [
                "series_id",
                "product_id",
                "department_id",
                "category_id",
                "store_id",
                "state_code",
            ]
        ]
        .drop_duplicates("series_id")
        .reset_index(drop=True)
    )

    static["shard_name"] = shard_name
    static["series_idx_in_shard"] = np.arange(len(static), dtype=np.int32)

    for col in STATIC_CAT_COLS:
        static[f"{col}_idx"] = static[col].map(cat_maps[col]).astype(np.int32)

    # -------------------------------------------------------------------------
    # Insert hierarchy tables
    # -------------------------------------------------------------------------

    stores = (
        static[["store_id", "state_code"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    categories = (
        static[["category_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    departments = (
        static[["department_id", "category_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    products = (
        static[["product_id", "department_id", "category_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    series_rows = static[
        [
            "series_id",
            "shard_name",
            "series_idx_in_shard",
            "product_id",
            "department_id",
            "category_id",
            "store_id",
            "state_code",
            "product_id_idx",
            "department_id_idx",
            "category_id_idx",
            "store_id_idx",
            "state_code_idx",
        ]
    ].itertuples(index=False, name=None)

    with conn:
        executemany_chunked(
            conn,
            """
            INSERT OR REPLACE INTO stores(store_id, state_code)
            VALUES (?, ?)
            """,
            list(stores),
        )

        executemany_chunked(
            conn,
            """
            INSERT OR REPLACE INTO categories(category_id)
            VALUES (?)
            """,
            list(categories),
        )

        executemany_chunked(
            conn,
            """
            INSERT OR REPLACE INTO departments(department_id, category_id)
            VALUES (?, ?)
            """,
            list(departments),
        )

        executemany_chunked(
            conn,
            """
            INSERT OR REPLACE INTO products(product_id, department_id, category_id)
            VALUES (?, ?, ?)
            """,
            list(products),
        )

        executemany_chunked(
            conn,
            """
            INSERT OR REPLACE INTO series(
                series_id,
                shard_name,
                series_idx_in_shard,
                product_id,
                department_id,
                category_id,
                store_id,
                state_code,
                product_id_idx,
                department_id_idx,
                category_id_idx,
                store_id_idx,
                state_code_idx
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            list(series_rows),
            chunk_size=10_000,
        )

    # -------------------------------------------------------------------------
    # Insert test windows: one per series/product
    # -------------------------------------------------------------------------

    test_windows = static.copy()

    test_windows["max_day"] = boundaries["max_day"]
    test_windows["effective_end"] = boundaries["effective_end"]
    test_windows["dropped_hanging_days"] = boundaries["dropped_hanging_days"]

    test_windows["context_start_day"] = boundaries["context_start"]
    test_windows["context_end_day"] = boundaries["context_end"]
    test_windows["target_start_day"] = boundaries["target_start"]
    test_windows["target_end_day"] = boundaries["target_end"]
    test_windows["test_start_day"] = boundaries["test_start"]
    test_windows["test_end_day"] = boundaries["test_end"]

    test_window_rows = test_windows[
        [
            "series_id",
            "shard_name",
            "store_id",
            "product_id",
            "department_id",
            "category_id",
            "state_code",
            "max_day",
            "effective_end",
            "dropped_hanging_days",
            "context_start_day",
            "context_end_day",
            "target_start_day",
            "target_end_day",
            "test_start_day",
            "test_end_day",
        ]
    ].itertuples(index=False, name=None)

    with conn:
        executemany_chunked(
            conn,
            """
            INSERT OR REPLACE INTO test_windows(
                series_id,
                shard_name,
                store_id,
                product_id,
                department_id,
                category_id,
                state_code,
                max_day,
                effective_end,
                dropped_hanging_days,
                context_start_day,
                context_end_day,
                target_start_day,
                target_end_day,
                test_start_day,
                test_end_day
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            list(test_window_rows),
            chunk_size=10_000,
        )

    # -------------------------------------------------------------------------
    # Insert raw 84-day test observations
    # -------------------------------------------------------------------------

    obs = df[
        (df["day_num"] >= boundaries["test_start"])
        & (df["day_num"] <= boundaries["test_end"])
    ].copy()

    obs["shard_name"] = shard_name

    obs["window_part"] = np.where(
        obs["day_num"] <= boundaries["context_end"],
        "context",
        "target",
    )

    obs = obs[
        [
            "series_id",
            "shard_name",
            "store_id",
            "product_id",
            "department_id",
            "category_id",
            "state_code",
            "day_num",
            "day_id",
            "date",
            "sales",
            "price_raw",
            "price_filled",
            "event_name_primary",
            "event_name_secondary",
            "snap_california",
            "snap_texas",
            "snap_wisconsin",
            "window_part",
        ]
    ]

    obs = clean_for_sqlite(obs)

    obs_rows = obs.itertuples(index=False, name=None)

    with conn:
        executemany_chunked(
            conn,
            """
            INSERT OR REPLACE INTO test_observations(
                series_id,
                shard_name,
                store_id,
                product_id,
                department_id,
                category_id,
                state_code,
                day_num,
                day_id,
                date,
                sales,
                price_raw,
                price_filled,
                event_name_primary,
                event_name_secondary,
                snap_california,
                snap_texas,
                snap_wisconsin,
                window_part
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            list(obs_rows),
            chunk_size=20_000,
        )

    print("[Inserted]")
    print(f"  series              : {len(static):,}")
    print(f"  test windows        : {len(test_windows):,}")
    print(f"  test observations   : {len(obs):,}")

    # Basic sanity check.
    expected_obs = len(static) * PERIOD
    if len(obs) != expected_obs:
        print(f"[Warn] Expected {expected_obs:,} test observations, got {len(obs):,}")

    # Show one example for confidence.
    sample_sid = static["series_id"].iloc[0]
    sample_obs = obs[obs["series_id"] == sample_sid].sort_values("day_num")

    print("[Sample]")
    print(f"  series_id       : {sample_sid}")
    print(f"  rows            : {len(sample_obs)}")
    print(f"  context days    : {sample_obs[sample_obs['window_part'] == 'context']['day_num'].min()}–"
          f"{sample_obs[sample_obs['window_part'] == 'context']['day_num'].max()}")
    print(f"  target days     : {sample_obs[sample_obs['window_part'] == 'target']['day_num'].min()}–"
          f"{sample_obs[sample_obs['window_part'] == 'target']['day_num'].max()}")
    print("  first rows:")
    print(sample_obs[["day_num", "date", "sales", "price_raw", "price_filled", "event_name_primary", "window_part"]].head(5).to_string(index=False))

    del df, static, obs, test_windows
    gc.collect()


# -----------------------------------------------------------------------------
# SUMMARY / EXAMPLE QUERIES
# -----------------------------------------------------------------------------

def print_summary(conn: sqlite3.Connection):
    print("\n[Database summary]")

    tables = [
        "stores",
        "categories",
        "departments",
        "products",
        "cat_mappings",
        "series",
        "test_windows",
        "test_observations",
        "predictions",
    ]

    for table in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:20s}: {n:,}")

    print("\n[Example app queries]")

    print(
        """
-- 1. Stores
SELECT store_id, state_code
FROM stores
ORDER BY store_id;
        """.strip()
    )

    print(
        """
-- 2. Departments in selected store
SELECT DISTINCT department_id, category_id
FROM series
WHERE store_id = ?
ORDER BY department_id;
        """.strip()
    )

    print(
        """
-- 3. Products in selected store and department
SELECT product_id, series_id
FROM series
WHERE store_id = ?
  AND department_id = ?
ORDER BY product_id;
        """.strip()
    )

    print(
        """
-- 4. Raw 84-day test block for selected product-store series
SELECT
    day_num,
    day_id,
    date,
    sales,
    price_raw,
    price_filled,
    event_name_primary,
    event_name_secondary,
    snap_california,
    snap_texas,
    snap_wisconsin,
    window_part
FROM test_observations
WHERE series_id = ?
ORDER BY day_num;
        """.strip()
    )

    print(
        """
-- 5. Static model categorical indices for selected series
SELECT
    product_id_idx,
    department_id_idx,
    category_id_idx,
    store_id_idx,
    state_code_idx
FROM series
WHERE series_id = ?;
        """.strip()
    )


def write_metadata_json(db_path: Path, data_dir: Path, cat_maps: dict, week_start_day: int):
    meta = {
        "db_path": str(db_path),
        "data_dir": str(data_dir),
        "context": CONTEXT,
        "target": TARGET,
        "period": PERIOD,
        "week_start_day": week_start_day,
        "static_cat_cols": STATIC_CAT_COLS,
        "cat_cardinalities": {k: len(v) for k, v in cat_maps.items()},
        "description": (
            "SQLite DB for demo app. Stores product hierarchy, categorical mappings, "
            "and raw final 84-day test-window observations for each product-store series."
        ),
    }

    out_path = db_path.with_suffix(".metadata.json")

    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Metadata] Wrote {out_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SQLite DB containing raw test-window data for TFT demo app."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing monster_*.parquet files. If omitted, resolved to TFT repo root/monster_parts",
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Output SQLite database path. If omitted, resolved to TFT repo root/monster_db/monster_demo.db",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing database before building.",
    )

    parser.add_argument(
        "--week-start-day",
        type=int,
        default=1,
        help=(
            "Week start day used for final complete test block. "
            "Default 1 means d_1,d_8,d_15 starts. "
            "Use 3 for M5 Monday alignment if needed."
        ),
    )

    parser.add_argument(
        "--debug-one-shard",
        action="store_true",
        help="Only process the first monster_*.parquet shard.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve TFT repo root and default locations when omitted so callers
    # don't need to run from a particular cwd.
    TFT_ROOT = Path(__file__).resolve().parents[2]

    DATA_DIR = args.data_dir or (TFT_ROOT / "monster_parts")
    DB_PATH = args.db_path or (TFT_ROOT / "monster_db" / "monster_demo.db")

    parquet_files = sorted(DATA_DIR.glob("monster_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No monster_*.parquet files found in {DATA_DIR}")

    if args.debug_one_shard:
        parquet_files = parquet_files[:1]

    print("[Config]")
    print(f"  data_dir        : {DATA_DIR}")
    print(f"  db_path         : {DB_PATH}")
    print(f"  overwrite       : {args.overwrite}")
    print(f"  week_start_day  : {args.week_start_day}")
    print(f"  shards          : {len(parquet_files)}")
    print(f"  context         : {CONTEXT}")
    print(f"  target          : {TARGET}")
    print(f"  period          : {PERIOD}")

    cat_maps = build_cat_maps(DATA_DIR)

    conn = connect_db(DB_PATH, overwrite=args.overwrite)
    create_schema(conn)
    insert_cat_mappings(conn, cat_maps)

    try:
        for path in parquet_files:
            process_parquet_file(
                conn=conn,
                parquet_path=path,
                cat_maps=cat_maps,
                week_start_day=args.week_start_day,
            )

        print_summary(conn)
        write_metadata_json(DB_PATH, DATA_DIR, cat_maps, args.week_start_day)

    finally:
        conn.close()

    print(f"\n✅ SQLite app database written to: {DB_PATH}")


if __name__ == "__main__":
    main()