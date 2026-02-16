from pathlib import Path
import pandas as pd
import argparse
import time

DATA_DIR = Path("data/amazon/processed")
ITEMS_PATH = DATA_DIR / "items_with_images.parquet"
IMAGES_DIR = Path("data/amazon/images")


def build_image_map(images_dir: Path) -> dict:
    m = {}
    if not images_dir.exists():
        return m
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        # expect prefix like 000123_... or 123_...
        parts = name.split("_")
        if not parts:
            continue
        prefix = parts[0]
        try:
            idx = int(prefix)
        except Exception:
            # maybe zero-padded, try removing leading zeros
            try:
                idx = int(prefix.lstrip("0"))
            except Exception:
                continue
        m[int(idx)] = str(p).replace("\\", "/")
    return m


def main(dry_run: bool):
    if not ITEMS_PATH.exists():
        print(f"Missing items file: {ITEMS_PATH}")
        return

    df = pd.read_parquet(ITEMS_PATH)
    df["image_path"] = df.get("image_path", "").fillna("").astype(str)

    img_map = build_image_map(IMAGES_DIR)
    missing = df[df["image_path"].str.len() == 0]
    fillable = missing[missing["item_idx"].isin(img_map.keys())]

    print(f"total_items={len(df)}")
    print(f"items_with_image_path={int((df['image_path'].str.len()>0).sum())}")
    print(f"items_without_image_path={len(missing)}")
    print(f"items_fillable_from_images_dir={len(fillable)}")

    if not fillable.empty:
        print("Sample mappings to apply:")
        for r in fillable.head(10).itertuples(index=False):
            idx = int(getattr(r, "item_idx"))
            print(f"{idx} -> {img_map[idx]}")

    if dry_run:
        print("Dry-run mode: no file will be written. Rerun with --apply to persist changes.")
        return

    # apply changes
    timestamp = int(time.time())
    backup = ITEMS_PATH.with_suffix(f".parquet.bak.{timestamp}")
    ITEMS_PATH.replace(backup)
    print(f"Backed up original to {backup}")

    # update df
    def mapper(row):
        if row.get("image_path", ""):
            return row["image_path"]
        return img_map.get(int(row["item_idx"]), "")

    df["image_path"] = df.apply(mapper, axis=1)
    df.to_parquet(ITEMS_PATH, index=False)
    print(f"Updated {ITEMS_PATH} with new image_path values.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Persist changes to parquet file")
    args = ap.parse_args()
    main(dry_run=not args.apply)
