from pathlib import Path
import pandas as pd
import numpy as np

# Candidate paths
candidates = [
    Path("data/amazon/processed/amazon_image_embeddings.parquet"),
    Path("data/amazon/processed_small/amazon_image_embeddings.parquet"),
    Path("image_hf/processed/amazon_image_embeddings.parquet"),
]
parquet_path = None
for p in candidates:
    if p.exists():
        parquet_path = p
        break

if parquet_path is None:
    raise FileNotFoundError("No amazon_image_embeddings.parquet found in expected locations.")

print(f"Loading embeddings from: {parquet_path}")

df = pd.read_parquet(parquet_path)
if 'item_idx' not in df.columns or 'embedding' not in df.columns:
    raise RuntimeError("Parquet must contain 'item_idx' and 'embedding' columns")

# take sample
N = 100
sample = df.head(N).copy()

rows = []
for r in sample.itertuples(index=False):
    iid = int(getattr(r, 'item_idx'))
    emb = getattr(r, 'embedding')
    arr = np.array(emb, dtype=np.float32)
    size = arr.size
    mean = float(arr.mean()) if size>0 else 0.0
    std = float(arr.std()) if size>0 else 0.0
    norm = float(np.linalg.norm(arr)) if size>0 else 0.0
    first5 = ','.join([f"{x:.4f}" for x in arr[:5]]) if size>0 else ''
    rows.append({
        'item_idx': iid,
        'dim': int(size),
        'mean': mean,
        'std': std,
        'norm': norm,
        'first5': first5,
    })

out_dir = Path('reports')
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / 'embeddings_preview.csv'
import csv
with out_csv.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['item_idx','dim','mean','std','norm','first5'])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print('Wrote:', out_csv)
print('\nPreview:')
for r in rows[:20]:
    print(r)
