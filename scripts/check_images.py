from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/amazon/processed")
ITEMS_PATH = DATA_DIR / "items_with_images.parquet"


def main():
    if not ITEMS_PATH.exists():
        print(f"Missing items file: {ITEMS_PATH}")
        return

    df = pd.read_parquet(ITEMS_PATH)
    df["image_path"] = df.get("image_path", "").fillna("").astype(str)

    total = len(df)
    with_images = int((df["image_path"].str.len() > 0).sum())
    without_path = total - with_images

    print(f"total_items={total}")
    print(f"items_with_image_path={with_images}")
    print(f"items_without_image_path={without_path}")

    # check file existence for those with image_path
    missing_files = []
    sample_missing_paths = []
    for r in df.itertuples(index=False):
        ip = str(getattr(r, "image_path", "") or "")
        if not ip:
            continue
        p = Path(ip)
        if not p.exists():
            missing_files.append(ip)
            if len(sample_missing_paths) < 10:
                sample_missing_paths.append(ip)

    print(f"items_with_image_path_but_file_missing={len(missing_files)}")
    if sample_missing_paths:
        print("Sample missing files:")
        for s in sample_missing_paths:
            print(s)

    # show some item_idx examples without image_path
    no_path_examples = df[df["image_path"].str.len() == 0].head(10)
    if not no_path_examples.empty:
        print("Sample items with no image_path (item_idx, item_raw_id, title):")
        for r in no_path_examples.itertuples(index=False):
            print(f"{getattr(r,'item_idx',None)}, {getattr(r,'item_raw_id',None)}, {getattr(r,'title',None)}")


if __name__ == '__main__':
    main()
