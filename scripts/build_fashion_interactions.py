from __future__ import annotations

import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ======================
# Paths
# ======================
DATA_DIR = Path("data/image_hf/processed")
CATALOG_PATH = DATA_DIR / "catalog.parquet"
OUT_DIR = DATA_DIR

# ======================
# Simulation parameters
# ======================
N_USERS = 5000
AVG_SESSIONS_PER_USER = 5
AVG_VIEWS_PER_SESSION = 10

P_CART = 0.15
P_PURCHASE = 0.35
P_REVISIT = 0.10

RANDOM_SEED = 42
TEST_SIZE = 0.2


# ======================
# Utilities
# ======================
def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


# ======================
# Main
# ======================
def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Loading catalog...")
    catalog = pd.read_parquet(CATALOG_PATH, columns=["product_id", "articleType"])
    catalog["product_id"] = catalog["product_id"].astype(int)
    catalog["articleType"] = catalog["articleType"].astype(str)

    products_by_type = {
        k: v["product_id"].tolist()
        for k, v in catalog.groupby("articleType")
    }
    article_types = list(products_by_type.keys())

    interactions = []

    start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.now()

    print("Simulating user interactions...")
    for user_id in range(N_USERS):
        # user preferences
        main_cat = random.choice(article_types)
        sec_cat = random.choice(article_types)

        n_sessions = max(1, int(np.random.poisson(AVG_SESSIONS_PER_USER)))

        user_seen = []

        for _ in range(n_sessions):
            cat = main_cat if random.random() < 0.7 else sec_cat
            products = products_by_type[cat]

            n_views = max(1, int(np.random.poisson(AVG_VIEWS_PER_SESSION)))

            for _ in range(n_views):
                # revisit logic
                if user_seen and random.random() < P_REVISIT:
                    pid = random.choice(user_seen)
                else:
                    pid = random.choice(products)
                    user_seen.append(pid)

                ts = random_date(start_date, end_date)
                interactions.append(
                    (user_id, pid, "view", ts)
                )

                # cart
                if random.random() < P_CART:
                    interactions.append(
                        (user_id, pid, "cart", ts + timedelta(minutes=5))
                    )

                    # purchase
                    if random.random() < P_PURCHASE:
                        interactions.append(
                            (user_id, pid, "purchase", ts + timedelta(minutes=15))
                        )

    df = pd.DataFrame(
        interactions,
        columns=["user_id", "product_id", "event", "timestamp"],
    )

    print(f"Total interactions: {len(df)}")

    print("Saving interactions.parquet...")
    df.to_parquet(OUT_DIR / "interactions.parquet", index=False)

    print("Train / test split...")
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["event"],
    )

    train_df.to_parquet(OUT_DIR / "interactions_train.parquet", index=False)
    test_df.to_parquet(OUT_DIR / "interactions_test.parquet", index=False)

    print("Done.")
    print(f"Train interactions: {len(train_df)}")
    print(f"Test interactions:  {len(test_df)}")


if __name__ == "__main__":
    main()
