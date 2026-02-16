from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from PIL import Image
from io import BytesIO


DATASET = "ashraq/fashion-product-images-small"
BASE_URL = "https://datasets-server.huggingface.co"

OUT_BASE = Path("data/image_hf")
RAW_DIR = OUT_BASE / "raw"
PROC_DIR = OUT_BASE / "processed"
IMG_DIR = OUT_BASE / "images"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ecommerce-reco/1.0"})


def http_get_json(url: str, params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def safe_image_url(cell: Any) -> Optional[str]:
    """
    Dataset Viewer returns image cells as a structured object.
    We try common shapes:
      - {"src": "..."}
      - {"url": "..."}
      - string URL
    """
    if cell is None:
        return None
    if isinstance(cell, str) and cell.startswith("http"):
        return cell
    if isinstance(cell, dict):
        for key in ("src", "url", "href"):
            v = cell.get(key)
            if isinstance(v, str) and v.startswith("http"):
                return v
    return None


def download_image(url: str, out_path: Path, timeout: int = 30) -> bool:
    try:
        r = SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(out_path, format="JPEG", quality=92)
        return True
    except Exception:
        return False


def get_config_and_split() -> Tuple[str, str]:
    # /splits?dataset=...
    data = http_get_json(f"{BASE_URL}/splits", params={"dataset": DATASET})
    # Expected: {"splits":[{"dataset":"...","config":"default","split":"train", ...}, ...]}
    splits = data.get("splits", [])
    if not splits:
        raise RuntimeError("No splits returned by /splits endpoint.")

    # Prefer default/train when present
    for s in splits:
        if s.get("config") == "default" and s.get("split") == "train":
            return "default", "train"

    # Otherwise pick the first
    return str(splits[0]["config"]), str(splits[0]["split"])


def fetch_rows(config: str, split: str, offset: int, length: int) -> Dict[str, Any]:
    # /rows supports max length 100
    return http_get_json(
        f"{BASE_URL}/rows",
        params={
            "dataset": DATASET,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        },
        timeout=60,
    )


def main():
    # Change these two values if you want more/less images
    target_n_images = 5000
    page_size = 100  # API limitation for /rows
    sleep_s = 0.05   # be polite to the service

    config, split = get_config_and_split()

    meta_path = RAW_DIR / "splits.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"dataset": DATASET, "config": config, "split": split}, f, ensure_ascii=False, indent=2)

    rows_out: List[Dict[str, Any]] = []
    downloaded = 0
    offset = 0

    # We loop until we reach target_n_images or the API returns no more rows.
    while downloaded < target_n_images:
        payload = fetch_rows(config=config, split=split, offset=offset, length=page_size)
        rows = payload.get("rows", [])
        if not rows:
            break

        for r in rows:
            # Viewer shape: {"row": {"id": ..., "image": {...}, ...}, ...}
            row_obj = r.get("row", {})
            if not isinstance(row_obj, dict) or not row_obj:
                continue

            # The dataset contains an "id" field and "productDisplayName", plus the image column
            pid = row_obj.get("id")
            img_cell = row_obj.get("image")
            img_url = safe_image_url(img_cell)

            if pid is None or img_url is None:
                continue

            img_path = IMG_DIR / f"{int(pid)}.jpg"
            ok = True
            if not img_path.exists():
                ok = download_image(img_url, img_path)
                time.sleep(sleep_s)

            if ok:
                downloaded += 1

            rows_out.append(
                {
                    "product_id": int(pid),
                    "productDisplayName": row_obj.get("productDisplayName"),
                    "gender": row_obj.get("gender"),
                    "masterCategory": row_obj.get("masterCategory"),
                    "subCategory": row_obj.get("subCategory"),
                    "articleType": row_obj.get("articleType"),
                    "baseColour": row_obj.get("baseColour"),
                    "season": row_obj.get("season"),
                    "year": row_obj.get("year"),
                    "usage": row_obj.get("usage"),
                    "image_url": img_url,
                    "image_path": str(img_path),
                    "image_downloaded": bool(ok),
                }
            )

            if downloaded >= target_n_images:
                break

        offset += page_size

    raw_rows_path = RAW_DIR / "rows_sample.json"
    with raw_rows_path.open("w", encoding="utf-8") as f:
        json.dump(rows_out[:200], f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(rows_out).drop_duplicates(subset=["product_id"])
    df = df[df["image_downloaded"] == True].reset_index(drop=True)

    csv_path = PROC_DIR / "catalog.csv"
    pq_path = PROC_DIR / "catalog.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)

    print(f"Dataset: {DATASET} | config={config} | split={split}")
    print(f"Images downloaded: {len(df)}")
    print(f"Catalog saved: {csv_path} and {pq_path}")


if __name__ == "__main__":
    main()
