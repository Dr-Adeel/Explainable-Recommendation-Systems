from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.recommenders.multimodal import load_parquet, fuse_embeddings

DATA_DIR = Path("data/amazon/processed")
IMAGE_EMB = DATA_DIR / "amazon_image_embeddings.parquet"
TEXT_EMB = Path("data/embeddings/text_embeddings.parquet")
OUT_DIR = Path("data/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "multimodal_embeddings.parquet"


def main(image_weight: float = 0.6, text_weight: float = 0.4):
    if not IMAGE_EMB.exists():
        raise FileNotFoundError(f"Missing image embeddings: {IMAGE_EMB}")
    if not TEXT_EMB.exists():
        raise FileNotFoundError(f"Missing text embeddings: {TEXT_EMB}")

    img = load_parquet(IMAGE_EMB)
    txt = load_parquet(TEXT_EMB)

    print(f"Loaded image embeddings n={len(img)}, text embeddings n={len(txt)}")
    fused = fuse_embeddings(img, txt, image_weight=image_weight, text_weight=text_weight)
    fused.to_parquet(OUT_PATH, index=False)
    print(f"Saved fused multimodal embeddings to {OUT_PATH} (n={len(fused)})")


if __name__ == '__main__':
    main()
