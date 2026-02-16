from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prepare_amazon_als_data")


def _find_one(root: Path, pattern: str) -> Path:
    hits = list(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"File not found under {root}: {pattern}")
    if len(hits) > 1:
        # Prefer shorter path (usually the intended one)
        hits = sorted(hits, key=lambda p: len(str(p)))
    return hits[0]


def _detect_cols(df: pd.DataFrame) -> Tuple[str, str, str | None, str | None]:
    """
    Tries to infer columns in benchmark CSV files.

    We need:
    - user column
    - item column
    Optional:
    - rating column
    - timestamp column
    """
    cols = set(df.columns)

    user_candidates = ["user_id", "reviewerID", "uid", "user"]
    item_candidates = ["item_id", "asin", "parent_asin", "iid", "item"]

    rating_candidates = ["rating", "overall", "stars", "score"]
    ts_candidates = ["timestamp", "unixReviewTime", "time", "sort_timestamp"]

    user_col = next((c for c in user_candidates if c in cols), None)
    item_col = next((c for c in item_candidates if c in cols), None)
    rating_col = next((c for c in rating_candidates if c in cols), None)
    ts_col = next((c for c in ts_candidates if c in cols), None)

    # Some benchmark splits are "user,item" only. If no rating, we'll set value=1 later.
    if user_col is None or item_col is None:
        raise ValueError(f"Cannot detect user/item columns. Found columns: {sorted(df.columns.tolist())}")

    return user_col, item_col, rating_col, ts_col


def _make_value(series: pd.Series | None, mode: str) -> pd.Series:
    """
    Convert explicit ratings into implicit positive confidence.
    """
    if series is None:
        return pd.Series([1.0])  # placeholder, handled by caller

    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype("float32")

    if mode == "binary4":
        # 1 if rating >= 4 else 0
        return (s >= 4.0).astype("float32")
    if mode == "shift3":
        # max(0, rating - 3)
        return (s - 3.0).clip(lower=0.0).astype("float32")

    raise ValueError(f"Unknown mode: {mode}")


def _collect_ids(dfs: Iterable[pd.DataFrame], user_col: str, item_col: str) -> Tuple[pd.Index, pd.Index]:
    users = pd.Index([])
    items = pd.Index([])
    for df in dfs:
        users = users.append(pd.Index(df[user_col].astype(str).unique()))
        items = items.append(pd.Index(df[item_col].astype(str).unique()))
    users = pd.Index(pd.unique(users))
    items = pd.Index(pd.unique(items))
    return users, items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/amazon/raw", help="Root directory containing benchmark CSV files")
    ap.add_argument("--out_dir", type=str, default="data/amazon/processed", help="Output directory")
    ap.add_argument("--value_mode", type=str, default="shift3", choices=["shift3", "binary4"])
    ap.add_argument("--min_value", type=float, default=0.0, help="Drop interactions with value <= min_value")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = _find_one(raw_dir, "Amazon_Fashion.train.csv")
    valid_path = _find_one(raw_dir, "Amazon_Fashion.valid.csv")
    test_path = _find_one(raw_dir, "Amazon_Fashion.test.csv")

    logger.info("Using splits:")
    logger.info(f"  train: {train_path}")
    logger.info(f"  valid: {valid_path}")
    logger.info(f"  test : {test_path}")

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    test = pd.read_csv(test_path)

    user_col, item_col, rating_col, ts_col = _detect_cols(train)
    logger.info(f"Detected columns: user={user_col} item={item_col} rating={rating_col} ts={ts_col}")

    # Normalize columns to strings
    for df in (train, valid, test):
        df[user_col] = df[user_col].astype(str)
        df[item_col] = df[item_col].astype(str)

    # Build ID mappings on ALL splits (avoid unseen ids later)
    users, items = _collect_ids([train, valid, test], user_col=user_col, item_col=item_col)
    user2idx = {u: int(i) for i, u in enumerate(users.tolist())}
    item2idx = {it: int(i) for i, it in enumerate(items.tolist())}

    mappings_dir = out_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)
    (mappings_dir / "user2idx.json").write_text(json.dumps(user2idx), encoding="utf-8")
    (mappings_dir / "item2idx.json").write_text(json.dumps(item2idx), encoding="utf-8")

    logger.info(f"Mappings: users={len(user2idx)} items={len(item2idx)} saved to {mappings_dir}")

    def convert(df: pd.DataFrame) -> pd.DataFrame:
        u = df[user_col].map(user2idx).astype("int64")
        it = df[item_col].map(item2idx).astype("int64")

        if rating_col and rating_col in df.columns:
            value = _make_value(df[rating_col], mode=args.value_mode)
        else:
            value = pd.Series([1.0] * len(df), dtype="float32")

        out = pd.DataFrame({"user_idx": u, "item_idx": it, "value": value.astype("float32")})

        if ts_col and ts_col in df.columns:
            out["timestamp"] = pd.to_numeric(df[ts_col], errors="coerce").fillna(0).astype("int64")

        # Drop non-positive values if desired
        if args.min_value is not None:
            out = out[out["value"] > float(args.min_value)]

        return out

    train_out = convert(train)
    valid_out = convert(valid)
    test_out = convert(test)

    train_out.to_parquet(out_dir / "interactions_train.parquet", index=False)
    valid_out.to_parquet(out_dir / "interactions_valid.parquet", index=False)
    test_out.to_parquet(out_dir / "interactions_test.parquet", index=False)

    meta = {
        "value_mode": args.value_mode,
        "min_value": float(args.min_value),
        "users": int(len(user2idx)),
        "items": int(len(item2idx)),
        "train_interactions": int(len(train_out)),
        "valid_interactions": int(len(valid_out)),
        "test_interactions": int(len(test_out)),
        "has_timestamp": bool(ts_col and ts_col in train.columns),
    }
    (out_dir / "interactions_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("Done.")
    logger.info(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
