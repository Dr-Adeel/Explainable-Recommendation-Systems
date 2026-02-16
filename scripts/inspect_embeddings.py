import pathlib
p=pathlib.Path('data/amazon/processed/amazon_image_embeddings.parquet')
print('path', p)
print('exists', p.exists())
if p.exists():
    import pyarrow.parquet as pq
    t=pq.read_table(p)
    print('rows=', t.num_rows)
    df=t.to_pandas()
    print('unique item_idx count', len(df['item_idx'].unique()))
    print('first 5 item_idx:', df['item_idx'].head().tolist())
    print('last 5 item_idx:', df['item_idx'].tail().tolist())
