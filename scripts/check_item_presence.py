from pathlib import Path
import pandas as pd

item = 2556

paths = {
    'items': Path('data/amazon/processed/items_with_images.parquet'),
    'emb': Path('data/amazon/processed/amazon_image_embeddings.parquet'),
    'cat': Path('data/amazon/processed/amazon_clip_catalog.parquet'),
}

for name,p in paths.items():
    if not p.exists():
        print(f'{name} not found at {p}')
        continue
    df = pd.read_parquet(p)
    has = item in df['item_idx'].astype(int).tolist()
    print(f'{name}: found={has} rows={len(df)}')
