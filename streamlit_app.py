import streamlit as st
from pathlib import Path
import sqlite3
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

from get_prediction import find_db, fetch_test_obs, make_batch_from_rows
from load_model import load_model
import zipfile, io


def _load_known_feature_names():
    root = Path(__file__).parent
    shards_dir = root / "monster_tensors_tft"
    if shards_dir.exists():
        for z in sorted(shards_dir.glob("*.zip")):
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
    return [f"known_{i}" for i in range(75)]


st.set_page_config(page_title="TFT Demo App", layout="wide")

st.title("TFT Forecast — Demo")

device = 'cpu'
# initialize compare/historic session flags
if 'do_compare' not in st.session_state:
    st.session_state['do_compare'] = False
if 'show_historic' not in st.session_state:
    st.session_state['show_historic'] = False
record = False

# Resolve DB and checkpoint internally (not user-facing)
db_path = find_db()
checkpoint = None

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Hierarchical selector: store -> department -> category -> product
stores = [r[0] for r in cur.execute("SELECT DISTINCT store_id FROM series ORDER BY store_id").fetchall()]
store = st.selectbox("Store", stores)

# User requested Category before Department
categories = [r[0] for r in cur.execute("SELECT DISTINCT category_id FROM series WHERE store_id=? ORDER BY category_id", (store,)).fetchall()]
category = st.selectbox("Category", categories)

departments = [r[0] for r in cur.execute("SELECT DISTINCT department_id FROM series WHERE store_id=? AND category_id=? ORDER BY department_id", (store, category)).fetchall()]
department = st.selectbox("Department", departments)

products = [r[0] for r in cur.execute("SELECT DISTINCT product_id FROM series WHERE store_id=? AND department_id=? AND category_id=? ORDER BY product_id", (store, department, category)).fetchall()]
product = st.selectbox("Product", products)

# Resolve series_id for the chosen product-store combination
sid_row = cur.execute(
    "SELECT series_id FROM series WHERE store_id=? AND department_id=? AND category_id=? AND product_id=? LIMIT 1",
    (store, department, category, product),
).fetchone()
if sid_row:
    selected = sid_row[0]
else:
    selected = None
    st.warning("No series found for this selection")

tog_col1, tog_col2 = st.columns([1, 1])
with tog_col1:
    compare_toggle = st.checkbox("Compare with truth", value=st.session_state['do_compare'], key='compare_toggle')
    st.session_state['do_compare'] = compare_toggle
with tog_col2:
    historic_toggle = st.checkbox("Show historic", value=st.session_state['show_historic'], key='historic_toggle')
    st.session_state['show_historic'] = historic_toggle

# Run prediction only when user presses the Predict button
predict_pressed = st.button("Run prediction", key='predict_btn')
if predict_pressed:
    if selected is None:
        st.warning("Please select a product first.")
    else:
        with st.spinner("Building input and running model..."):
            rows = fetch_test_obs(conn, selected)
            # get static index
            row = cur.execute(
                "SELECT product_id_idx, department_id_idx, category_id_idx, store_id_idx, state_code_idx FROM series WHERE series_id = ?",
                (selected,),
            ).fetchone()
            if row is None:
                st.error("Series not found in DB")
            else:
                static_idx = list(row)

                model, metadata, CFG = load_model(checkpoint if checkpoint else None, device=device)

                known_dim = int(metadata.get('known_dim', 75))
                batch = make_batch_from_rows(rows, static_idx, context=int(CFG.CONTEXT), target=int(CFG.TARGET), known_dim=known_dim)

                device_t = next(model.parameters()).device
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(device_t)

                model.eval()
                start_t = datetime.utcnow()
                with torch.no_grad():
                    pred_tensor, agg_tensor = model(batch)
                    # ensure non-negative sales by applying softplus (matches training loss)
                    pred_tensor = F.softplus(pred_tensor)
                pred = pred_tensor.cpu().numpy()[0]
                agg = agg_tensor.cpu().numpy()[0]
                end_t = datetime.utcnow()
                fetch_secs = (end_t - start_t).total_seconds()

                # map agg to names
                hist_dim = int(metadata.get('hist_dim', 2))
                known_dim = int(metadata.get('known_dim', 75))
                hist_names = ['sales', 'price'][:hist_dim]
                known_names = _load_known_feature_names()[:known_dim]
                raw_feature_names = hist_names + known_names
                if len(raw_feature_names) != len(agg):
                    raw_feature_names = [f"f_{i}" for i in range(len(agg))]

                top_idx = np.argsort(-agg)[:3]
                top_features = [(raw_feature_names[i], float(agg[i])) for i in top_idx]

                target_vals = [r[1] if r[1] is not None else 0.0 for r in rows if r[3] == 'target']
                context_vals = [r[1] if r[1] is not None else 0.0 for r in rows if r[3] == 'context']

                def rmse(a, b):
                    a = np.asarray(a)
                    b = np.asarray(b)
                    return float(np.sqrt(np.mean((a - b) ** 2)))

                # Always compute rmse if true targets exist
                monthly_rmse = rmse(pred, target_vals) if len(target_vals) == len(pred) else None
                total_units = float(np.sum(pred))
                actual_total = float(np.sum(target_vals)) if len(target_vals) == len(pred) else None

                # store prediction outputs in session state so toggles can re-render without re-running model
                st.session_state['last_pred'] = pred.tolist()
                st.session_state['last_agg'] = agg.tolist()
                st.session_state['last_rows'] = rows
                st.session_state['last_fetch_secs'] = fetch_secs
                st.session_state['last_selected'] = selected
                st.session_state['last_top_features'] = top_features
                st.session_state['last_monthly_rmse'] = monthly_rmse
                st.session_state['last_total_units'] = total_units
                st.session_state['last_actual_total'] = actual_total
                st.session_state['last_target_vals'] = target_vals
                st.session_state['last_context_vals'] = context_vals
                st.session_state['last_pred_time'] = end_t.isoformat() + 'Z'

                if record:
                    model_name = 'best_skeleton'
                    with conn:
                        for i, y in enumerate(pred):
                            target_days = [r for r in rows if r[3] == 'target']
                            if i < len(target_days):
                                day_num = target_days[i][0]
                                day_id = target_days[i][2] if len(target_days[i]) > 2 else None
                            else:
                                day_num = None
                                day_id = None
                            cur.execute(
                                "INSERT INTO predictions(series_id, model_name, day_num, day_id, date, y_true, y_pred) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (selected, model_name, day_num, day_id, st.session_state['last_pred_time'], None if not st.session_state.get('do_compare') else target_vals[i], float(y)),
                            )
                st.success("Prediction complete — use toggles to update the view")

# Rendering block: draw last prediction if it exists and matches the current selection
if 'last_pred' in st.session_state and st.session_state.get('last_selected') == selected:
    pred = np.array(st.session_state['last_pred'])
    top_features = st.session_state.get('last_top_features', [])
    target_vals = st.session_state.get('last_target_vals', [])
    context_vals = st.session_state.get('last_context_vals', [])
    fetch_secs = st.session_state.get('last_fetch_secs', 0.0)
    monthly_rmse = st.session_state.get('last_monthly_rmse')
    total_units = st.session_state.get('last_total_units', float(np.sum(pred)))
    actual_total = st.session_state.get('last_actual_total')

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(1, len(pred) + 1)
    if st.session_state.get('show_historic') and context_vals:
        ctx_x = np.arange(-len(context_vals) + 1, 1)
        ax.plot(ctx_x, context_vals, marker='.', markersize=3, alpha=0.6, label='historic')
    if st.session_state.get('do_compare') and len(target_vals) == len(pred):
        ax.plot(x, target_vals, marker='o', markersize=3, label='actual')
    ax.plot(x, pred, marker='o', markersize=3, label='prediction')
    ax.set_title(f"Series: {selected}")
    ax.set_xlabel('forecast day')
    ax.set_ylabel('sales')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Totals: predicted always; actual only when compare is on
    tcol1, tcol2, tcol3 = st.columns([1, 1, 1])
    tcol1.metric(label="Predicted total (28 days)", value=f"{total_units:,.0f}")
    if st.session_state.get('do_compare') and actual_total is not None:
        tcol2.metric(label="Actual total (28 days)", value=f"{actual_total:,.0f}")
        diff = total_units - actual_total
        tcol3.metric(label="Difference (pred - actual)", value=f"{diff:,.0f}")
    else:
        tcol2.metric(label="Actual total (28 days)", value="N/A")
        tcol3.metric(label="Difference (pred - actual)", value="N/A")

    st.write(f"Fetch time (s): **{fetch_secs:0.2f}**")

    if st.session_state.get('do_compare'):
        if monthly_rmse is not None:
            mae = float(np.mean(np.abs(np.asarray(pred) - np.asarray(target_vals))))
            st.write(f"RMSE (28 days): **{monthly_rmse:0.2f}**, MAE: **{mae:0.2f}**")
        else:
            st.write("RMSE (28 days): N/A")

    st.subheader("Top 3 features")
    labels = [t[0] for t in top_features]
    sizes = np.array([t[1] for t in top_features])
    if sizes.sum() > 0:
        sizes = sizes / sizes.sum()
    fig2, ax2 = plt.subplots(figsize=(2, 2))
    ax2.pie(sizes, labels=None, startangle=90)
    ax2.axis('equal')
    total_pct = sizes.sum() if sizes.sum() > 0 else 1.0
    legend_labels = [f"{labels[i]} ({100.0 * sizes[i]/total_pct:0.1f}%)" for i in range(len(labels))]
    ax2.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    st.pyplot(fig2)

st.write("---")
st.write("Select a product to run prediction. Use the toggles above to compare with truth or show historic context.")
