import sys
from pathlib import Path
import argparse
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch

# ensure local TFT module imports resolve
sys.path.insert(0, str(Path(__file__).parent))

from get_prediction import find_db, fetch_test_obs, make_batch_from_rows, load_npz_from_zip, find_db as gp_find_db
from load_model import load_model
import json
import zipfile, io


def _load_known_feature_names():
    # Try to read known_col_list from any shard npz under TFT/monster_tensors_tft
    root = Path(__file__).parent
    shards = sorted((root / "monster_tensors_tft").glob("*.zip")) if (root / "monster_tensors_tft").exists() else []
    for z in shards:
        try:
            with zipfile.ZipFile(z, 'r') as zf:
                npz_names = [n for n in zf.namelist() if n.endswith('.npz')]
                if not npz_names:
                    continue
                with zf.open(npz_names[0]) as f:
                    data = np.load(io.BytesIO(f.read()), allow_pickle=True)
                    if 'known_col_list' in data.files:
                        try:
                            return [str(x) for x in data['known_col_list'].tolist()]
                        except Exception:
                            return [str(x) for x in data['known_col_list']]
        except Exception:
            continue
    # fallback to generic names
    return [f"known_{i}" for i in range(75)]


def compare_series(series_id: str, db_path: str | Path | None = None, checkpoint: str | None = None, device: str = "cpu", show: bool = True, out: str | None = None, show_historic: bool = False, do_compare: bool = True, record: bool = False):
    if db_path is None:
        db_path = find_db()

    conn = sqlite3.connect(db_path)

    # fetch static encoded indices for this series
    cur = conn.cursor()
    row = cur.execute(
        "SELECT product_id_idx, department_id_idx, category_id_idx, store_id_idx, state_code_idx FROM series WHERE series_id = ?",
        (series_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Series {series_id} not found in DB {db_path}")

    static_idx = list(row)

    # fetch 84-day test observations
    rows = fetch_test_obs(conn, series_id)

    # extract true target (sales) values from target window
    target_vals = [r[1] if r[1] is not None else 0.0 for r in rows if r[3] == 'target']
    context_vals = [r[1] if r[1] is not None else 0.0 for r in rows if r[3] == 'context']

    # build batch for model (uses zeros for unknown known features)
    batch = make_batch_from_rows(rows, static_idx)

    model, metadata, CFG = load_model(checkpoint, device=device)

    device_t = next(model.parameters()).device
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device_t)

    model.eval()
    with torch.no_grad():
        pred, agg = model(batch)

    pred = pred.cpu().numpy()[0]
    agg = agg.cpu().numpy()[0]

    # map agg weights to feature names
    hist_dim = int(metadata.get('hist_dim', 2))
    known_dim = int(metadata.get('known_dim', 75))

    hist_names = ['sales', 'price'][:hist_dim]
    known_names = _load_known_feature_names()[:known_dim]

    raw_feature_names = hist_names + known_names

    if len(raw_feature_names) != len(agg):
        # fallback to generic naming if mismatch
        raw_feature_names = [f"f_{i}" for i in range(len(agg))]

    # top-3 features
    top_idx = np.argsort(-agg)[:3]
    top_features = [(raw_feature_names[i], float(agg[i])) for i in top_idx]

    # metrics
    def rmse(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    monthly_rmse = rmse(pred, target_vals) if do_compare else None
    total_units = float(np.sum(pred))

    # plotting
    x = np.arange(1, len(pred) + 1)
    plt.figure(figsize=(10, 4))
    if show_historic and context_vals:
        ctx_x = np.arange(-len(context_vals), 0)
        plt.plot(ctx_x, context_vals, marker='.', alpha=0.6, label='historic')

    if do_compare:
        plt.plot(x, target_vals, marker='o', label='actual')

    plt.plot(x, pred, marker='o', label='prediction')
    plt.title(f"Series: {series_id}")
    plt.xlabel('forecast day')
    plt.ylabel('sales')
    plt.grid(True)
    plt.legend()

    if out:
        plt.savefig(out, bbox_inches='tight')

    if show:
        plt.show()

    print('Series:', series_id)
    print('Static indices:', static_idx)
    if do_compare:
        print('True target:', target_vals)
    print('Predicted :', pred.tolist())
    print('Top 3 features:')
    for name, w in top_features:
        print(f"  {name:30s} {w:0.6f}")
    if monthly_rmse is not None:
        print(f"Monthly RMSE (28 days): {monthly_rmse:0.4f}")
    print(f"Total units predicted (28 days): {total_units:0.2f}")

    # record prediction time and optionally persist predictions into DB
    pred_time = datetime.utcnow().isoformat() + 'Z'
    print(f"Prediction time (UTC): {pred_time}")

    if record:
        # insert per-day predictions into predictions table
        model_name = Path(checkpoint).name if checkpoint else 'best_skeleton'
        with conn:
            cur = conn.cursor()
            for i, y in enumerate(pred):
                day_num = None
                # find day_num for target day i
                target_days = [r for r in rows if r[3] == 'target']
                if i < len(target_days):
                    day_num = target_days[i][0]
                    day_id = target_days[i][2] if len(target_days[i]) > 2 else None
                else:
                    day_id = None

                cur.execute(
                    "INSERT INTO predictions(series_id, model_name, day_num, day_id, date, y_true, y_pred) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (series_id, model_name, day_num, day_id, None, None if not do_compare else target_vals[i], float(y)),
                )
        print("Recorded predictions into DB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--series-id', required=True)
    p.add_argument('--db', default=None)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default=None, help='Save plot to file')
    p.add_argument('--show-historic', action='store_true', help='Include historic context in plot')
    p.add_argument('--no-compare', dest='do_compare', action='store_false', help='Do not plot/compare true target')
    p.add_argument('--record', action='store_true', help='Record per-day predictions into DB predictions table')
    args = p.parse_args()

    compare_series(
        args.series_id,
        db_path=args.db,
        checkpoint=args.checkpoint,
        device=args.device,
        out=args.out,
        show_historic=args.show_historic,
        do_compare=args.do_compare,
        record=args.record,
    )


if __name__ == '__main__':
    main()
