import sqlite3
from pathlib import Path
import random
import torch
import numpy as np

from load_model import load_model


DB_CANDIDATES = [
    Path(__file__).parent / "tft_demo.sqlite",
    Path(__file__).parent / "monster_demo.db",
    Path(__file__).parent.parent / "monster_demo.db",
    Path(__file__).parent.parent / "tft_demo.sqlite",
    Path(__file__).parent / "demo_app.sqlite",
    Path(__file__).parent.parent / "demo_app.sqlite",
]


def find_db():
    for p in DB_CANDIDATES:
        if not p.exists():
            continue
        try:
            conn = sqlite3.connect(p)
            cur = conn.cursor()
            row = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='series'").fetchone()
            conn.close()
            if row:
                return p
        except Exception:
            continue

    raise FileNotFoundError(f"No suitable DB found with 'series' table among: {DB_CANDIDATES}")


def pick_random_series(conn: sqlite3.Connection):
    cur = conn.cursor()
    row = cur.execute(
        "SELECT series_id, product_id_idx, department_id_idx, category_id_idx, store_id_idx, state_code_idx FROM series ORDER BY RANDOM() LIMIT 1;"
    ).fetchone()
    if row is None:
        raise ValueError("No series found in DB")
    sid, pid_i, dept_i, cat_i, store_i, state_i = row
    return sid, [pid_i, dept_i, cat_i, store_i, state_i]


def fetch_test_obs(conn: sqlite3.Connection, series_id: str):
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT day_num, sales, price_filled, window_part FROM test_observations WHERE series_id = ? ORDER BY day_num;",
        (series_id,),
    ).fetchall()
    if not rows:
        raise ValueError(f"No observations for series {series_id}")
    return rows


def make_batch_from_rows(rows, static_idx, context=56, target=28, known_dim=75):
    # rows: list of (day_num, sales, price_filled, window_part)
    arr = np.array([[r[1] if r[1] is not None else 0.0, r[2] if r[2] is not None else 0.0] for r in rows], dtype=np.float32)

    if arr.shape[0] < context + target:
        raise ValueError("Not enough days in series to build full test window")

    historic_numeric = arr  # [84, 2]

    past_numeric = historic_numeric[:context, :][None, ...].astype(np.float32)  # [1,56,2]
    future_price = historic_numeric[context:context + target, 1:2][None, ...].astype(np.float32)  # [1,28,1]

    # we don't have the full known_numeric features in the DB; use zeros as fallback
    past_known = np.zeros((1, context, known_dim), dtype=np.float32)
    future_known = np.zeros((1, target, known_dim), dtype=np.float32)

    batch = {
        "static_cats": torch.from_numpy(np.array([static_idx], dtype=np.int64)).long(),
        "past_numeric": torch.from_numpy(past_numeric).float(),
        "past_known": torch.from_numpy(past_known).float(),
        "future_known": torch.from_numpy(future_known).float(),
        "future_price": torch.from_numpy(future_price).float(),
    }

    return batch


def load_npz_from_zip(zip_path: Path):
    import zipfile, io

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        npz_names = [n for n in zf.namelist() if n.endswith(".npz")]
        if not npz_names:
            raise FileNotFoundError(f"No .npz inside {zip_path}")
        with zf.open(npz_names[0]) as f:
            return np.load(io.BytesIO(f.read()), allow_pickle=False)


def make_batch_from_shard(d, s_idx: int, t_idx: int, context=56, target=28):
    # d is the npz dict-like loaded from shard
    static_context = d["static_context"]
    historic_numeric = d["historic_numeric"]
    known_numeric = d["known_numeric"]
    sales = d["sales"]

    s_idx = int(s_idx)
    t_idx = int(t_idx)

    c = context
    h = target

    idx_context = t_idx - c + np.arange(c)
    idx_target = t_idx + np.arange(h)

    past_numeric = historic_numeric[s_idx, idx_context, :][None, ...].astype(np.float32)
    past_known = known_numeric[s_idx, idx_context, :][None, ...].astype(np.float32)
    future_known = known_numeric[s_idx, idx_target, :][None, ...].astype(np.float32)
    future_price = historic_numeric[s_idx, idx_target, 1:2][None, ...].astype(np.float32)

    static_cats = static_context[s_idx][None, :].astype(np.int64)

    batch = {
        "static_cats": torch.from_numpy(static_cats).long(),
        "past_numeric": torch.from_numpy(past_numeric).float(),
        "past_known": torch.from_numpy(past_known).float(),
        "future_known": torch.from_numpy(future_known).float(),
        "future_price": torch.from_numpy(future_price).float(),
        "target": torch.from_numpy(sales[s_idx, idx_target][None, ...]).float(),
        "file": zip_path.name if 'zip_path' in locals() else None,
        "series_idx": torch.tensor([s_idx]).long(),
        "t_idx": torch.tensor([t_idx]).long(),
    }

    return batch


def run_for_shard_series_tidx(shard_path: str | Path, series_idx: int, t_idx: int, checkpoint_path=None, device="cpu"):
    d = load_npz_from_zip(Path(shard_path))

    # build batch for that exact (s_idx,t_idx)
    batch = make_batch_from_shard(d, series_idx, t_idx, context=56, target=28)

    # get static series_id from shard metadata if needed
    # load model
    model, metadata, CFG = load_model(checkpoint_path, device=device)

    device_t = next(model.parameters()).device
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device_t)

    with torch.no_grad():
        pred, agg = model(batch)

    pred = pred.cpu().numpy()[0]
    print(f"Shard: {shard_path}, series_idx: {series_idx}, t_idx: {t_idx}")
    print(f"Prediction (28 days): {pred.tolist()}")



def run_once(db_path=None, checkpoint_path=None, device="cpu"):
    if db_path is None:
        db_path = find_db()

    conn = sqlite3.connect(db_path)

    sid, static_idx = pick_random_series(conn)
    rows = fetch_test_obs(conn, sid)

    model, metadata, CFG = load_model(checkpoint_path, device=device)

    known_dim = int(metadata.get("known_dim", 75))
    batch = make_batch_from_rows(rows, static_idx, context=int(CFG.CONTEXT), target=int(CFG.TARGET), known_dim=known_dim)

    # ensure tensors on same device as model
    device_t = next(model.parameters()).device
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device_t)

    with torch.no_grad():
        pred, agg = model(batch)

    pred = pred.cpu().numpy()[0]

    print(f"Series: {sid}")
    print(f"Static (encoded indices): {static_idx}")
    print(f"Prediction (28 days): {pred.tolist()}")


def run_for_series(series_id: str, db_path: Path | str | None = None, checkpoint_path=None, device="cpu"):
    if db_path is None:
        db_path = find_db()

    conn = sqlite3.connect(db_path)
    rows = fetch_test_obs(conn, series_id)
    cur = conn.cursor()
    row = cur.execute(
        "SELECT product_id_idx, department_id_idx, category_id_idx, store_id_idx, state_code_idx FROM series WHERE series_id = ?",
        (series_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Series {series_id} not found in series table")
    static_idx = list(row)

    model, metadata, CFG = load_model(checkpoint_path, device=device)

    known_dim = int(metadata.get("known_dim", 75))
    batch = make_batch_from_rows(rows, static_idx, context=int(CFG.CONTEXT), target=int(CFG.TARGET), known_dim=known_dim)

    device_t = next(model.parameters()).device
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device_t)

    with torch.no_grad():
        pred, agg = model(batch)

    pred = pred.cpu().numpy()[0]

    print(f"Series: {series_id}")
    print(f"Static (encoded indices): {static_idx}")
    print(f"Prediction (28 days): {pred.tolist()}")



if __name__ == "__main__":
    run_once()
