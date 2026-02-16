from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DATA_DIR = Path("data/image_hf")
CATALOG_PATH = DATA_DIR / "processed" / "catalog.parquet"
OUT_PATH = DATA_DIR / "processed" / "image_embeddings.parquet"


def load_catalog() -> pd.DataFrame:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Missing catalog: {CATALOG_PATH}")
    df = pd.read_parquet(CATALOG_PATH)

    required = {"product_id", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"catalog.parquet missing columns: {missing}")

    df = df.dropna(subset=["product_id", "image_path"]).copy()
    df["image_path"] = df["image_path"].astype(str)
    return df


def open_image(path: str) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None


@torch.inference_mode()
def main():
    # Parameters
    max_images = 5000       # change to 20000+ later if you want
    batch_size = 32         # adjust if you run out of RAM
    random_seed = 42

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # Load catalog
    df = load_catalog()
    df = df.drop_duplicates(subset=["product_id"])

    # Keep only existing images
    exists_mask = df["image_path"].map(lambda p: Path(p).exists())
    df = df[exists_mask].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No valid images found in catalog.")

    # Sample for a first run
    if max_images is not None and len(df) > max_images:
        df = df.sample(n=max_images, random_state=random_seed).reset_index(drop=True)

    print(f"catalog_rows={len(df)}")

    # Load CLIP
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    # Compute embeddings
    product_ids: List[int] = []
    image_paths: List[str] = []
    names: List[str] = []
    embeddings: List[List[float]] = []

    title_col = "productDisplayName" if "productDisplayName" in df.columns else None

    n_ok = 0
    n_fail = 0

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        imgs: List[Image.Image] = []
        meta: List[Dict[str, Any]] = []

        for _, row in batch.iterrows():
            img = open_image(row["image_path"])
            if img is None:
                n_fail += 1
                continue
            imgs.append(img)
            meta.append({
                "product_id": int(row["product_id"]),
                "image_path": str(row["image_path"]),
                "name": str(row[title_col]) if title_col else "",
            })

        if not imgs:
            continue

        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = out.pooler_output  # (B, hidden)
        image_features = model.visual_projection(pooled)  # (B, 512)
        image_features = torch.nn.functional.normalize(image_features, dim=1)




        


        feats = image_features.detach().cpu().tolist()

        for m, e in zip(meta, feats):
            product_ids.append(m["product_id"])
            image_paths.append(m["image_path"])
            names.append(m["name"])
            embeddings.append(e)
            n_ok += 1

        if (start // batch_size) % 10 == 0:
            print(f"processed={min(start + batch_size, len(df))}/{len(df)} ok={n_ok} fail={n_fail}")

    out = pd.DataFrame({
        "product_id": product_ids,
        "name": names,
        "image_path": image_paths,
        "embedding": embeddings,
    }).drop_duplicates(subset=["product_id"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print(f"saved={OUT_PATH}")
    print(f"rows={len(out)}")


if __name__ == "__main__":
    main()
