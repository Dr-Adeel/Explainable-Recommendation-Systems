from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image


CATALOG_PATH = Path("data/image_hf/processed/catalog.parquet")
OUT_DIR = Path("data/image_hf/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main(limit: int | None = None) -> None:
    df = pd.read_parquet(CATALOG_PATH).copy()
    if "product_id" not in df.columns:
        raise ValueError("catalog.parquet must contain 'product_id'")

    df["product_id"] = df["product_id"].astype(int)

    if "image_path" not in df.columns:
        df["image_path"] = ""

    # Load dataset once (downloads locally)
    ds = load_dataset("ashraq/fashion-product-images-small", split="train")

    # Detect the id column used by the dataset
    id_col = None
    for candidate in ["id", "product_id", "productId"]:
        if candidate in ds.column_names:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(f"Cannot find an id column in dataset columns: {ds.column_names}")

    # Build mapping: dataset_id -> dataset_index
    ids = ds[id_col]
    id_to_idx = {int(v): i for i, v in enumerate(ids)}

    work_df = df if limit is None else df.head(limit)
    ok = 0
    miss = 0

    for i, row in work_df.iterrows():
        pid = int(row["product_id"])
        out_path = OUT_DIR / f"{pid}.jpg"

        if out_path.exists() and out_path.stat().st_size > 0:
            df.at[i, "image_path"] = str(out_path).replace("\\", "/")
            ok += 1
            continue

        ds_idx = id_to_idx.get(pid)
        if ds_idx is None:
            miss += 1
            continue

        ex = ds[ds_idx]
        img = ex.get("image")
        if img is None:
            miss += 1
            continue

        if isinstance(img, Image.Image):
            pil = img
        else:
            pil = img.convert("RGB")

        pil = pil.convert("RGB")
        pil.save(out_path, format="JPEG", quality=92, optimize=True)

        df.at[i, "image_path"] = str(out_path).replace("\\", "/")
        ok += 1

    df.to_parquet(CATALOG_PATH, index=False)
    print(f"id_col={id_col}")
    print(f"Downloaded OK: {ok}")
    print(f"Missing id matches: {miss}")
    print(f"Images dir: {OUT_DIR}")


if __name__ == "__main__":
    main(limit=None)  # None = all rows in catalog.parquet

