import pandas as pd
p='data/amazon/processed/amazon_clip_catalog.parquet'
df=pd.read_parquet(p)
print('rows', len(df))
print('first5', df['item_idx'].head().tolist())
print('last5', df['item_idx'].tail().tolist())
