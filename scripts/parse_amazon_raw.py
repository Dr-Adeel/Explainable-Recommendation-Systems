from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd

RAW_DIR = Path("data/amazon/raw")
SPLIT_DIR = RAW_DIR / "benchmark/0core/last_out_w_his"
META_PATH = RAW_DIR / "raw/meta_categories/meta_Amazon_Fashion.jsonl"

DEFAULT_OUT_DIR = Path("data/amazon/processed")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _infer_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def infer_schema(df: pd.DataFrame) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Tries to infer user, item, rating, timestamp columns from benchmark CSV.
    Returns: (user_col, item_col, rating_col_or_None, ts_col_or_None)
    """
    user_col = _infer_col(df, ["user_id", "reviewerID", "uid", "user", "user"])
    item_col = _infer_col(df, ["parent_asin", "asin", "item_id", "iid", "item"])
    rating_col = _infer_col(df, ["rating", "overall", "stars", "score"])
    ts_col = _infer_col(df, ["timestamp", "unixReviewTime", "time", "sort_timestamp"])

    if user_col is None or item_col is None:
        raise ValueError(
            f"Could not infer schema. Columns={list(df.columns)} "
            f"(need something like user_id + asin/parent_asin/item_id)"
        )
    return user_col, item_col, rating_col, ts_col


def load_meta(meta_path: Path) -> pd.DataFrame:
    """
    Loads metadata JSONL and returns a table with:
      item_raw_id, title, main_category, image_url
    Uses parent_asin if present, else asin.
    """
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    rows = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_raw = obj.get("parent_asin") or obj.get("asin")
            if not item_raw:
                continue

            title = obj.get("title") or ""
            main_category = obj.get("main_category") or ""

            image_url = ""
            images = obj.get("images")
            if isinstance(images, list) and images:
                # Prefer hi_res, else large, else thumb
                for im in images:
                    if not isinstance(im, dict):
                        continue
                    image_url = im.get("hi_res") or im.get("large") or im.get("thumb") or ""
                    if image_url:
                        break

            rows.append(
                {
                    "item_raw_id": str(item_raw),
                    "title": str(title),
                    "main_category": str(main_category),
                    "image_url": str(image_url),
                }
            )

    meta = pd.DataFrame(rows).drop_duplicates(subset=["item_raw_id"])
    return meta


def apply_scope(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    user_col: str,
    item_col: str,
    max_items: int,
    max_users: int,
    max_interactions: int,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select a subset that is easier to run locally while keeping splits consistent.
    Strategy:
      - sample items from TRAIN
      - filter splits to those items
      - sample users from filtered TRAIN
      - filter splits to those users
      - cap interactions per split (train first)
    """
    train = train.copy()
    valid = valid.copy()
    test = test.copy()

    # sample items based on train frequency
    item_counts = train[item_col].value_counts()
    keep_items = item_counts.head(max_items).index
    train = train[train[item_col].isin(keep_items)]
    valid = valid[valid[item_col].isin(keep_items)]
    test = test[test[item_col].isin(keep_items)]

    # sample users based on train activity
    user_counts = train[user_col].value_counts()
    keep_users = user_counts.head(max_users).index
    train = train[train[user_col].isin(keep_users)]
    valid = valid[valid[user_col].isin(keep_users)]
    test = test[test[user_col].isin(keep_users)]

    # cap interactions (prefer keeping train large)
    def cap(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(df) <= n:
            return df
        return df.sample(n=n, random_state=seed)

    # allocate caps: ~80/10/10 by default
    n_train = int(max_interactions * 0.8)
    n_valid = int(max_interactions * 0.1)
    n_test = max_interactions - n_train - n_valid

    train = cap(train, n_train)
    valid = cap(valid, n_valid)
    test = cap(test, n_test)

    return train, valid, test


def build_index_maps(train: pd.DataFrame, user_col: str, item_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, int]]:
    user_raw = train[user_col].astype(str).unique().tolist()
    item_raw = train[item_col].astype(str).unique().tolist()

    user2idx = {u: i for i, u in enumerate(user_raw)}
    item2idx = {it: i for i, it in enumerate(item_raw)}

    users = pd.DataFrame({"user_idx": list(user2idx.values()), "user_raw_id": list(user2idx.keys())})
    items = pd.DataFrame({"item_idx": list(item2idx.values()), "item_raw_id": list(item2idx.keys())})

    return users, items, user2idx, item2idx


def map_interactions(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    rating_col: Optional[str],
    ts_col: Optional[str],
) -> pd.DataFrame:
    out = pd.DataFrame()
    out["user_idx"] = df[user_col].astype(str).map(user2idx)
    out["item_idx"] = df[item_col].astype(str).map(item2idx)

    if rating_col is not None:
        out["rating"] = pd.to_numeric(df[rating_col], errors="coerce")
    if ts_col is not None:
        out["timestamp"] = pd.to_numeric(df[ts_col], errors="coerce")

    out = out.dropna(subset=["user_idx", "item_idx"]).copy()
    out["user_idx"] = out["user_idx"].astype(int)
    out["item_idx"] = out["item_idx"].astype(int)

    return out


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = SPLIT_DIR / "Amazon_Fashion.train.csv"
    valid_path = SPLIT_DIR / "Amazon_Fashion.valid.csv"
    test_path = SPLIT_DIR / "Amazon_Fashion.test.csv"

    train_raw = _read_csv(train_path)
    valid_raw = _read_csv(valid_path)
    test_raw = _read_csv(test_path)

    user_col, item_col, rating_col, ts_col = infer_schema(train_raw)

    # Convert key cols to string for consistent joins/mapping
    for df in (train_raw, valid_raw, test_raw):
        df[user_col] = df[user_col].astype(str)
        df[item_col] = df[item_col].astype(str)

    # Scope (adjust here if needed)
    MAX_ITEMS = 8000
    MAX_USERS = 5000
    MAX_INTERACTIONS = 100000

    train_raw, valid_raw, test_raw = apply_scope(
        train_raw,
        valid_raw,
        test_raw,
        user_col=user_col,
        item_col=item_col,
        max_items=MAX_ITEMS,
        max_users=MAX_USERS,
        max_interactions=MAX_INTERACTIONS,
    )

    users, items, user2idx, item2idx = build_index_maps(train_raw, user_col, item_col)

    train = map_interactions(train_raw, user_col, item_col, user2idx, item2idx, rating_col, ts_col)
    valid = map_interactions(valid_raw, user_col, item_col, user2idx, item2idx, rating_col, ts_col)
    test = map_interactions(test_raw, user_col, item_col, user2idx, item2idx, rating_col, ts_col)

    # Load metadata and merge into items
    meta = load_meta(META_PATH)
    items = items.merge(meta, on="item_raw_id", how="left")

    # Save outputs
    users.to_parquet(out_dir / "users.parquet", index=False)
    items.to_parquet(out_dir / "items.parquet", index=False)

    train.to_parquet(out_dir / "interactions_train.parquet", index=False)
    valid.to_parquet(out_dir / "interactions_valid.parquet", index=False)
    test.to_parquet(out_dir / "interactions_test.parquet", index=False)

    summary: Dict[str, Any] = {
        "source": {
            "splits_dir": str(SPLIT_DIR).replace("\\", "/"),
            "meta_path": str(META_PATH).replace("\\", "/"),
        },
        "schema": {
            "user_col": user_col,
            "item_col": item_col,
            "rating_col": rating_col,
            "timestamp_col": ts_col,
        },
        "scope": {
            "max_items": MAX_ITEMS,
            "max_users": MAX_USERS,
            "max_interactions": MAX_INTERACTIONS,
        },
        "counts": {
            "users": int(users.shape[0]),
            "items": int(items.shape[0]),
            "train_interactions": int(train.shape[0]),
            "valid_interactions": int(valid.shape[0]),
            "test_interactions": int(test.shape[0]),
        },
        "notes": {
            "item_ids_mapped": "item_raw_id -> item_idx (0..I-1)",
            "user_ids_mapped": "user_raw_id -> user_idx (0..U-1)",
        },
    }

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    interactions_meta = {
        "users": int(users.shape[0]),
        "items": int(items.shape[0]),
        "train_interactions": int(train.shape[0]),
        "valid_interactions": int(valid.shape[0]),
        "test_interactions": int(test.shape[0]),
        "has_timestamp": bool(ts_col),
    }
    with (out_dir / "interactions_meta.json").open("w", encoding="utf-8") as f:
        json.dump(interactions_meta, f, ensure_ascii=False, indent=2)

    print("A3 complete.")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build scoped Amazon interactions (A3)")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    args = parser.parse_args()
    main(Path(args.out_dir))
