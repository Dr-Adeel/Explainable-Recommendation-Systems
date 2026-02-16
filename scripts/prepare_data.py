from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")


def load_csv(filename: str, usecols=None) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_csv(path, usecols=usecols)


def build_interactions(orders: pd.DataFrame, order_products: pd.DataFrame) -> pd.DataFrame:
    interactions = order_products.merge(
        orders[["order_id", "user_id", "order_number"]],
        on="order_id",
        how="inner",
        validate="many_to_one",
    )

    interactions["event"] = "purchase"

    keep_cols = ["user_id", "product_id", "order_number", "event"]
    if "reordered" in interactions.columns:
        keep_cols.append("reordered")

    interactions = interactions[keep_cols].dropna()

    interactions["user_id"] = interactions["user_id"].astype("int64")
    interactions["product_id"] = interactions["product_id"].astype("int64")
    interactions["order_number"] = interactions["order_number"].astype("int64")

    return interactions.sort_values(["user_id", "order_number", "product_id"]).reset_index(drop=True)


def split_last_order_as_test(interactions: pd.DataFrame):
    last_order = interactions.groupby("user_id")["order_number"].transform("max")
    is_test = interactions["order_number"] == last_order
    return interactions[~is_test].copy(), interactions[is_test].copy()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("📥 Chargement...")
    products = load_csv("products.csv")
    orders = load_csv("orders.csv", usecols=["order_id", "user_id", "order_number"])
    order_products = load_csv("order_products__prior.csv", usecols=["order_id", "product_id", "reordered"])

    print("🧱 Construction interactions...")
    interactions = build_interactions(orders, order_products)

    print("✂️ Split train/test (dernière commande = test)...")
    train_df, test_df = split_last_order_as_test(interactions)

    print("💾 Sauvegarde dans data/processed/ ...")
    products.to_parquet(OUT_DIR / "products.parquet", index=False)
    train_df.to_parquet(OUT_DIR / "interactions_train.parquet", index=False)
    test_df.to_parquet(OUT_DIR / "interactions_test.parquet", index=False)

    print("✅ OK")
    print(f"Users: {interactions['user_id'].nunique()} | Products: {interactions['product_id'].nunique()}")
    print(f"Train interactions: {len(train_df)} | Test interactions: {len(test_df)}")


if __name__ == "__main__":
    main()
