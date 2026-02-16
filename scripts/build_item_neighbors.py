from __future__ import annotations

from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "item_neighbors.parquet"


def load_csv(path: Path, usecols=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Introuvable: {path}")
    return pd.read_csv(path, usecols=usecols)


def main():
    # === PARAMÈTRES ===
    TOP_ITEMS = 20000         # produits retenus
    MAX_ORDERS = 1000000      # nb max de commandes à traiter
    TOP_NEIGHBORS = 50        # voisins à garder
    # ==================

    print("📥 Chargement orders.csv et order_products__prior.csv ...")
    orders = load_csv(RAW_DIR / "orders.csv", usecols=["order_id", "user_id", "order_number"])
    op = load_csv(RAW_DIR / "order_products__prior.csv", usecols=["order_id", "product_id"])

    print("🧺 Construction paniers par order_id (produits uniques par commande)...")
    op = op.drop_duplicates(["order_id", "product_id"])

    print(f"📊 Calcul fréquence produit (nb de commandes) pour TOP {TOP_ITEMS}...")
    freq_by_orders = op["product_id"].value_counts()
    top_items = set(freq_by_orders.head(TOP_ITEMS).index.astype("int64").tolist())
    print(f"✅ TOP items retenus: {len(top_items)}")

    print("🔗 Filtrage op sur TOP items...")
    op = op[op["product_id"].isin(top_items)]

    print("🧮 Comptage co-occurrences sur commandes...")
    cooc = Counter()
    freq = Counter()

    grouped = op.groupby("order_id")["product_id"]

    processed = 0
    for _, prods in grouped:
        if MAX_ORDERS is not None and processed >= MAX_ORDERS:
            break

        items = sorted(set(int(x) for x in prods.tolist()))
        if len(items) < 2:
            processed += 1
            continue

        for i in items:
            freq[i] += 1
        for i, j in combinations(items, 2):
            cooc[(i, j)] += 1

        processed += 1
        if processed % 20000 == 0:
            print(f"  ... commandes traitées: {processed}")

    print(f"✅ Commandes traitées: {processed}")
    print(f"🔗 Paires uniques comptées: {len(cooc)}")

    print("🧠 Calcul Jaccard + Top voisins...")
    neighbors = defaultdict(list)

    for (i, j), cij in cooc.items():
        denom = freq[i] + freq[j] - cij
        if denom <= 0:
            continue
        score = cij / denom
        neighbors[i].append((j, score))
        neighbors[j].append((i, score))

    rows = []
    for i, neigh_list in neighbors.items():
        neigh_list.sort(key=lambda x: x[1], reverse=True)
        for j, s in neigh_list[:TOP_NEIGHBORS]:
            rows.append((i, j, float(s)))

    out_df = pd.DataFrame(rows, columns=["product_id", "neighbor_id", "score"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"💾 Sauvegarde: {OUT_PATH}")
    out_df.to_parquet(OUT_PATH, index=False)

    print("✅ Terminé !")
    print(f"Neighbors rows: {len(out_df)}")
    print(out_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
