from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/image_hf/processed")
TRAIN_PATH = DATA_DIR / "interactions_train.parquet"
OUT_PATH = DATA_DIR / "fashion_item_neighbors.parquet"

# Parameters
TOP_N_PER_ITEM = 50

# Event weights (simple + explainable)
EVENT_WEIGHT = {
    "view": 1.0,
    "cart": 3.0,
    "purchase": 5.0,
}


def main() -> None:
    df = pd.read_parquet(TRAIN_PATH, columns=["user_id", "product_id", "event"])
    df["user_id"] = df["user_id"].astype(int)
    df["product_id"] = df["product_id"].astype(int)
    df["event"] = df["event"].astype(str)

    # Convert events to weights
    df["w"] = df["event"].map(EVENT_WEIGHT).fillna(1.0).astype(float)

    # For each user, get unique items + max weight per item (avoid spamming same item)
    user_items = (
        df.groupby(["user_id", "product_id"])["w"]
        .max()
        .reset_index()
        .sort_values(["user_id", "w"], ascending=[True, False])
    )

    # Build co-occurrence counts (weighted)
    neigh_scores = defaultdict(float)

    grouped = user_items.groupby("user_id")["product_id"].apply(list)
    grouped_w = user_items.groupby("user_id")["w"].apply(list)

    for items, ws in zip(grouped.tolist(), grouped_w.tolist()):
        if len(items) < 2:
            continue

        # simple: add pair score = wi*wj
        for i in range(len(items)):
            pi = items[i]
            wi = ws[i]
            for j in range(i + 1, len(items)):
                pj = items[j]
                wj = ws[j]
                s = wi * wj
                neigh_scores[(pi, pj)] += s
                neigh_scores[(pj, pi)] += s

    # Convert to DataFrame
    rows = [(p, n, float(s)) for (p, n), s in neigh_scores.items()]
    out = pd.DataFrame(rows, columns=["product_id", "neighbor_id", "score"])

    # Keep top neighbors per product
    out = out.sort_values(["product_id", "score"], ascending=[True, False])
    out = out.groupby("product_id").head(TOP_N_PER_ITEM).reset_index(drop=True)

    out.to_parquet(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(out)}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
