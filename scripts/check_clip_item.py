import pyarrow.parquet as pq
import numpy as np
import pathlib, sys
p = 'data/amazon/processed/amazon_image_embeddings.parquet'
if not pathlib.Path(p).exists():
    print('EMB file missing:', p)
    sys.exit(0)
df = pq.read_table(p).to_pandas()
if 'item_idx' not in df.columns or 'embedding' not in df.columns:
    print('parquet missing cols', df.columns.tolist())
    sys.exit(0)

for item in [4694, 4695]:
    row = df[df['item_idx'] == item]
    if row.empty:
        print(item, 'NOT IN EMB')
    else:
        emb = row.iloc[0]['embedding']
        arr = np.array(emb, dtype=np.float32)
        print(item, 'len', len(arr), 'norm', float(np.linalg.norm(arr)), 'sum', float(arr.sum()), 'min', float(arr.min()), 'max', float(arr.max()))
