# TFT Demo (M5 forecasting utilities)

Short guide to prepare data, train the model, and run the Streamlit demo in this repository.

## Overview

This workspace contains utilities to convert the M5 Kaggle CSVs into parquet and TFT-ready tensors, build a small application SQLite DB, train the TFT model, and run a Streamlit demo to inspect 28-day forecasts and comparisons with ground truth.

## Prerequisites

- Python 3.8+ (3.10 recommended)
- virtualenv or venv
- GPU recommended for training but CPU works for inference
- Install dependencies (example):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt || pip install streamlit torch pandas numpy matplotlib sqlite3
```

If `requirements.txt` is present, prefer that. Adjust `torch` install for CUDA if needed.

## 1) Download the dataset

Download the M5 Forecasting Accuracy dataset from Kaggle into the repository root under `raw_csv_data/` (create the folder if missing):

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Required files (place in `raw_csv_data/`):

- `calendar.csv`
- `sales_train_evaluation.csv`
- `sales_train_validation.csv`
- `sell_prices.csv`

Example:

```bash
mkdir -p raw_csv_data
# move downloaded files into raw_csv_data/
```

## 2) Create parquet / monster_parts

There are notebooks and scripts in `TFT/` to build the parquet inputs. You can either run the notebook or the conversion script. Options:

- Run the notebook: open `TFT/csv_to_parquet.ipynb` in Jupyter and execute the cells.
- Or run the conversion script (if available):

```bash
python TFT/src/data_src/parquet_to_tensor.py --raw-dir raw_csv_data --out-dir TFT/monster_parts
```

After this step you should have parquet files (``monster_parts``) under `TFT/monster_parts/`.

## 3) Build TFT tensors (monster_tensors_tft)

Convert parquet to TFT tensors used by the model. Use the provided script/notebook:

```bash
python TFT/src/data_src/parquet_to_tensor.py --parquet-dir TFT/monster_parts --out-dir TFT/monster_tensors_tft
```

This generates the shard files in `TFT/monster_tensors_tft/`.

## 4) Create the application SQLite DB

Build the `monster_demo.db` (used by the demo) from the parquet or processed data using the sqlite helper:

```bash
python TFT/src/data_src/sqlite.py --data-dir TFT/monster_parts --out-db TFT/monster_db/monster_demo.db
```

Adjust flags if your file layout differs. After this you should have `TFT/monster_db/monster_demo.db`.

## 5) Train the model and save weights

Open the training notebook and run the training pipeline to produce saved weights.

Recommended:

```bash
jupyter lab TFT/src/train_src/tft.ipynb
# or open TFT/tft.ipynb and run cells
```

Training will write checkpoints to `TFT/saved_weights/` (or `saved_weights_skeleton/`); note the path to the best checkpoint (e.g. `TFT/saved_weights/saved_weights_skeleton/best_skeleton.pt`).

If you already have a pretrained checkpoint, place it under `TFT/saved_weights/` or update the `streamlit_app.py` `checkpoint` variable to point to it.

## 6) Run the Streamlit demo

Start the demo from the `TFT/` folder (ensure virtualenv is active):

```bash
cd TFT
streamlit run streamlit_app.py
```

The app will:
- let you select Store → Category → Department → Product
- run predictions using the saved model
- show a 28-day forecast, optional historic context, and optional comparison with truth
- display top-3 attention features and totals

Notes:
- The demo uses the SQLite DB as the canonical data source. If the DB is missing, create it as described above.
- For large product lists you may prefer to implement a searchable selector (typeahead) — the codebase contains the UI scaffolding.

## Troubleshooting

- If Streamlit fails to find the DB, check `TFT/monster_db/` and ensure `monster_demo.db` exists.
- If model loading fails, verify the checkpoint path and that the checkpoint matches the model code in `TFT/model/`.
- If you run out of memory during training, train on smaller subsets or use a GPU.

## Developer notes

- Key scripts and files:
  - `TFT/src/data_src/parquet_to_tensor.py` — parquet → tensors
  - `TFT/src/data_src/sqlite.py` — build app DB
  - `TFT/src/train_src/tft.ipynb` — training notebook
  - `TFT/streamlit_app.py` — Streamlit demo
  - `TFT/model/model.py` — TFT model definition

If you want, I can add a `requirements.txt` with tested pins, or implement a searchable Product selector in the Streamlit app next.
