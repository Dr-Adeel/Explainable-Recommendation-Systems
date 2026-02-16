from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np

# ensure project root is on sys.path so `src` package is importable when run as script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders.text_encoder import get_encoder

DATA_DIR = Path("data/amazon/processed")
ITEMS_PATH = DATA_DIR / "items_with_images.parquet"
OUT_DIR = Path("data/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "text_embeddings.parquet"


def main(model_name: str = "all-MiniLM-L6-v2"):
    if not ITEMS_PATH.exists():
        raise FileNotFoundError(f"Missing items file: {ITEMS_PATH}")

    df = pd.read_parquet(ITEMS_PATH)
    # use title + description where available
    texts = []
    ids = []
    for r in df.itertuples(index=False):
        iid = int(getattr(r, "item_idx", -1) or -1)
        title = str(getattr(r, "title", "") or "")
        desc = str(getattr(r, "description", "") or "")
        txt = (title + " ") + desc
        texts.append(txt)
        ids.append(iid)

    enc = get_encoder(model_name=model_name)
    print("Encoding texts (this may take a while with transformer models)...")
    embs = enc.encode(texts)

    # write parquet: item_idx + embedding (as list)
    out_df = pd.DataFrame({"item_idx": ids, "embedding": [list(e.astype(float)) for e in embs]})
    out_df.to_parquet(OUT_PATH, index=False)
    print(f"Saved text embeddings to {OUT_PATH} (n={len(out_df)}, dim={embs.shape[1]})")


if __name__ == '__main__':
    main()
