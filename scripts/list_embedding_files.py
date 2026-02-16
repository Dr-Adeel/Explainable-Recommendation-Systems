import pathlib
p=pathlib.Path('data/amazon')
for fp in sorted(p.rglob('*amazon_image_embeddings*.parquet')):
    print(fp)
for fp in sorted(p.rglob('*amazon_clip_catalog*.parquet')):
    print(fp)
for fp in sorted(p.rglob('**/*.parquet')):
    # list sizes for processed folder
    if 'processed' in str(fp.parent):
        try:
            print(fp, fp.stat().st_size)
        except Exception:
            pass
