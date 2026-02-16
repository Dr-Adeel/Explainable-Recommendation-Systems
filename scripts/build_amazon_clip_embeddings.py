from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DATA_DIR = Path("data/amazon/processed")
ITEMS_PATH = DATA_DIR / "items_with_images.parquet"

OUT_EMB = DATA_DIR / "amazon_image_embeddings.parquet"
OUT_CATALOG = DATA_DIR / "amazon_clip_catalog.parquet"

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_items(limit: int | None = 3500) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS_PATH)
    df = df[df["image_path"].astype(str).str.len() > 0].copy()

    # if limit provided, sample to keep the run bounded; if None, keep all
    if limit is not None and len(df) > int(limit):
        df = df.sample(n=int(limit), random_state=42)

    df["item_idx"] = df["item_idx"].astype(int)
    df["title"] = df.get("title", "").fillna("").astype(str)
    df["main_category"] = df.get("main_category", "").fillna("").astype(str)
    df["image_path"] = df["image_path"].astype(str)
    return df


def batched(iterable: List[Tuple[int, str]], batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


@torch.inference_mode()
def main(limit: int | None = 3500) -> None:
    if not ITEMS_PATH.exists():
        raise FileNotFoundError(f"Missing: {ITEMS_PATH}")

    df = load_items(limit=limit)
    rows = len(df)
    if rows == 0:
        raise RuntimeError("No rows with image_path found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    pairs = [(int(r.item_idx), str(r.image_path)) for r in df.itertuples(index=False)]
    batch_size = 32 if device == "cuda" else 16

    item_ids: List[int] = []
    embeddings: List[np.ndarray] = []

    skipped = 0

    for batch in batched(pairs, batch_size=batch_size):
        images = []
        keep_ids = []

        for item_idx, img_path in batch:
            p = Path(img_path)
            if not p.exists():
                skipped += 1
                continue
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                keep_ids.append(item_idx)
            except Exception:
                skipped += 1

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        feats = model.get_image_features(**inputs)  # [B, 512]
        if not isinstance(feats, torch.Tensor):
            if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                feats = feats.image_embeds
            elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            else:
                raise RuntimeError("Unexpected image features output type")

        feats = torch.nn.functional.normalize(feats, dim=1)
        feats_np = feats.detach().cpu().numpy().astype("float32")

        item_ids.extend(keep_ids)
        embeddings.extend([feats_np[i] for i in range(feats_np.shape[0])])

    emb_df = pd.DataFrame(
        {
            "item_idx": item_ids,
            "embedding": embeddings,
        }
    )

    cat_df = df[["item_idx", "title", "main_category", "image_path"]].copy()
    cat_df = cat_df[cat_df["item_idx"].isin(set(item_ids))].drop_duplicates("item_idx")

    emb_df.to_parquet(OUT_EMB, index=False)
    cat_df.to_parquet(OUT_CATALOG, index=False)

    print("A6 complete.")
    print(f"device={device}")
    print(f"rows_in={rows} rows_embedded={len(emb_df)} skipped={skipped}")
    print(f"saved_embeddings={OUT_EMB.as_posix()}")
    print(f"saved_catalog={OUT_CATALOG.as_posix()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build CLIP image embeddings for Amazon catalog")
    parser.add_argument("--limit", type=int, default=3500, help="Max number of images to process (use 0 for no limit)")
    args = parser.parse_args()
    lim = None if args.limit == 0 else args.limit
    main(limit=lim)
