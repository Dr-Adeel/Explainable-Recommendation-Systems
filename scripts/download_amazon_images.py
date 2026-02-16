from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests


IN_DIR = Path("data/amazon/processed")
ITEMS_PATH = IN_DIR / "items.parquet"

OUT_DIR = Path("data/amazon")
IMG_DIR = OUT_DIR / "images"
OUT_ITEMS_PATH = IN_DIR / "items_with_images.parquet"

IMG_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "img"


def choose_ext_from_url(url: str) -> str:
    url = url.lower()
    if ".png" in url:
        return "png"
    if ".webp" in url:
        return "webp"
    return "jpg"


def download_one(
    url: str,
    out_path: Path,
    session: requests.Session,
    timeout: int = 20,
    max_retries: int = 3,
    sleep_base: float = 0.8,
) -> Tuple[bool, str]:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, "cached"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }

    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, headers=headers, timeout=timeout, stream=True)
            if r.status_code != 200:
                last_err = f"http_{r.status_code}"
                time.sleep(sleep_base * attempt)
                continue

            content_type = (r.headers.get("content-type") or "").lower()
            if "image" not in content_type:
                last_err = f"not_image_content_type:{content_type}"
                time.sleep(sleep_base * attempt)
                continue

            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            with tmp_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)

            if tmp_path.stat().st_size == 0:
                tmp_path.unlink(missing_ok=True)
                last_err = "empty_file"
                time.sleep(sleep_base * attempt)
                continue

            tmp_path.replace(out_path)
            return True, "downloaded"

        except Exception as e:
            last_err = f"exception:{type(e).__name__}"
            time.sleep(sleep_base * attempt)

    return False, last_err or "failed"


def main() -> None:
    if not ITEMS_PATH.exists():
        raise FileNotFoundError(f"Missing: {ITEMS_PATH}")

    items = pd.read_parquet(ITEMS_PATH)

    required = {"item_idx", "item_raw_id"}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"items.parquet missing columns: {sorted(missing)}")

    if "image_url" not in items.columns:
        raise ValueError("items.parquet has no image_url column. Check A3 merge with metadata.")

    # Config (laptop-friendly)
    # Increase to cover all items with image_url so missing image_path entries
    # are filled. Set high value to avoid sampling.
    MAX_IMAGES = 10000          # you can increase later
    MIN_SIZE_BYTES = 5_000     # skip broken tiny files

    # Keep only rows with URL
    df = items.copy()
    df["image_url"] = df["image_url"].fillna("").astype(str)
    df = df[df["image_url"].str.len() > 10].copy()

    # Prioritize items that appear more in train: if you later add item_freq column, sort by it.
    # For now: just take first MAX_IMAGES
    if len(df) > MAX_IMAGES:
        df = df.sample(n=MAX_IMAGES, random_state=42)

    ok = 0
    fail = 0
    cached = 0

    session = requests.Session()

    image_path_col = []
    status_col = []

    for row in df.itertuples(index=False):
        item_idx = int(getattr(row, "item_idx"))
        raw_id = str(getattr(row, "item_raw_id"))
        url = str(getattr(row, "image_url"))

        title = ""
        if "title" in df.columns:
            title = str(getattr(row, "title", "")) or ""

        ext = choose_ext_from_url(url)
        fname = f"{item_idx:06d}_{sanitize_filename(raw_id)}_{sanitize_filename(title)[:40]}.{ext}"
        out_path = IMG_DIR / fname

        success, status = download_one(url, out_path, session=session)

        if success:
            if status == "cached":
                cached += 1
            else:
                ok += 1

            # ensure not tiny
            if out_path.exists() and out_path.stat().st_size < MIN_SIZE_BYTES:
                out_path.unlink(missing_ok=True)
                success = False
                status = "too_small"

        if not success:
            fail += 1
            image_path_col.append("")
            status_col.append(status)
        else:
            image_path_col.append(str(out_path).replace("\\", "/"))
            status_col.append(status)

    # Merge back into full items table
    df_out = df.copy()
    df_out["image_path"] = image_path_col
    df_out["image_status"] = status_col

    # Keep only successful image_path for join
    df_ok = df_out[df_out["image_path"].str.len() > 0][["item_idx", "image_path"]].drop_duplicates("item_idx")

    items2 = items.merge(df_ok, on="item_idx", how="left")
    items2["image_path"] = items2["image_path"].fillna("").astype(str)

    # Save
    items2.to_parquet(OUT_ITEMS_PATH, index=False)

    # Summary
    total_attempted = int(len(df_out))
    with_images = int((items2["image_path"].str.len() > 0).sum())

    print("A4 complete.")
    print(f"attempted={total_attempted} downloaded={ok} cached={cached} failed={fail}")
    print(f"items_total={len(items2)} items_with_images={with_images}")
    print(f"saved_items={OUT_ITEMS_PATH.as_posix()}")
    print(f"images_dir={IMG_DIR.as_posix()}")


if __name__ == "__main__":
    main()
