import json
from pathlib import Path
import torch


def _default_cfg():
    return {
        "context": 56,
        "target": 28,
        "hidden_dim": 128,
        "embedding_dims": {
            "product_id": 32,
            "department_id": 8,
            "category_id": 8,
            "store_id": 8,
            "state_code": 8,
        },
    }


def load_checkpoint(path: Path | str | None = None):
    if path is None:
        path = Path(__file__).parent / "saved_weights" / "saved_weights_skeleton" / "best_skeleton.pt"
    path = Path(path)
    if not path.exists():
        # fallback to other common location
        alt = Path(__file__).parent / "saved_weights_skeleton" / "best_skeleton.pt"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    ck = torch.load(path, map_location="cpu")

    if isinstance(ck, dict) and "model_state" in ck:
        model_state = ck["model_state"]
        metadata = ck.get("metadata")
        cfg = ck.get("cfg")
    else:
        model_state = ck
        metadata = None
        cfg = None

    return model_state, metadata, cfg


def load_model(checkpoint_path: Path | str | None = None, device: str = "cpu"):
    model_state, metadata, cfg = load_checkpoint(checkpoint_path)

    # lazy import model module so we can set CFG before instantiation
    import importlib
    mod = importlib.import_module("model.model")

    # minimal CFG used by model.model
    class CFG:
        pass

    base = _default_cfg()
    if cfg and isinstance(cfg, dict):
        base.update(cfg)

    CFG.CONTEXT = int(base.get("context", 56))
    CFG.TARGET = int(base.get("target", 28))
    CFG.HIDDEN_DIM = int(base.get("hidden_dim", 128))
    CFG.EMBEDDING_DIMS = base.get("embedding_dims", _default_cfg()["embedding_dims"])

    # attach to module so TFTModel can reference CFG
    setattr(mod, "CFG", CFG)

    # infer metadata if not present
    if metadata is None:
        # try reading a metadata json from monster_tensors_tft
        meta_file = Path(__file__).parent / "monster_tensors_tft" / "monster_CA_1_metadata.json"
        if meta_file.exists():
            md = json.load(open(meta_file))
            metadata = {
                "static_cat_cols": [
                    "product_id",
                    "department_id",
                    "category_id",
                    "store_id",
                    "state_code",
                ],
                "hist_dim": int(md.get("num_historic_numeric_features", 2)),
                "known_dim": int(md.get("num_known_temporal_features", 75)),
                "cardinalities": {},
            }
        else:
            metadata = {
                "static_cat_cols": [
                    "product_id",
                    "department_id",
                    "category_id",
                    "store_id",
                    "state_code",
                ],
                "hist_dim": 2,
                "known_dim": 75,
                "cardinalities": {},
            }

    # if cardinalities not present, infer from saved model_state embedding weights
    card = metadata.get("cardinalities") or {}

    for k, v in model_state.items():
        # look for keys like 'static_block.embeddings.product_id.weight'
        parts = k.split('.')
        if len(parts) >= 4 and parts[0] == "static_block" and parts[1] == "embeddings" and parts[-1] == "weight":
            name = parts[2]
            if hasattr(v, "shape"):
                card[name] = int(v.shape[0])

    metadata["cardinalities"] = card

    # instantiate model
    TFT = getattr(mod, "TFTModel")
    embedding_dims = CFG.EMBEDDING_DIMS

    model = TFT(
        metadata=metadata,
        embedding_dims=embedding_dims,
        hidden_dim=getattr(CFG, "HIDDEN_DIM", 128),
    )

    # load weights
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return model, metadata, CFG


if __name__ == "__main__":
    m, meta, cfg = load_model()
    print("Loaded model:")
    print(m.params)
