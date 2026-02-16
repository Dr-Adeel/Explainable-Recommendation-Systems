from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_amazon_csr")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/amazon/processed")
    ap.add_argument("--out_dir", type=str, default="data/amazon/processed/als")
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "interactions_train.parquet"
    meta_path = processed_dir / "interactions_meta.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_users = int(meta["users"])
    n_items = int(meta["items"])

    df = pd.read_parquet(train_path)
    df["user_idx"] = df["user_idx"].astype("int64")
    df["item_idx"] = df["item_idx"].astype("int64")

    if "value" in df.columns:
        df["value"] = df["value"].astype("float32")
    elif "rating" in df.columns:
        df["value"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).astype("float32")
    else:
        df["value"] = 1.0

    if (df["value"] < 0).any():
        raise ValueError("Found negative values in 'value'. Implicit ALS requires non-negative confidence.")

    rows = df["user_idx"].to_numpy(dtype=np.int64)
    cols = df["item_idx"].to_numpy(dtype=np.int64)
    data = df["value"].to_numpy(dtype=np.float32)

    X = csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    save_npz(out_dir / "train_csr.npz", X)

    stats = {
        "shape": [int(n_users), int(n_items)],
        "nnz": int(X.nnz),
        "density": float(X.nnz / (n_users * n_items)) if n_users and n_items else 0.0,
        "value_min": float(data.min()) if len(data) else 0.0,
        "value_max": float(data.max()) if len(data) else 0.0,
        "value_mean": float(data.mean()) if len(data) else 0.0,
    }
    (out_dir / "csr_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info(f"Saved CSR: {out_dir / 'train_csr.npz'}")
    # Log stats in a compact, human-friendly format
    logger.info(f"shape: {stats['shape']}")
    logger.info(f"nnz: {stats['nnz']}")
    logger.info(f"density: {stats['density']:.5f}")


if __name__ == "__main__":
    main()
