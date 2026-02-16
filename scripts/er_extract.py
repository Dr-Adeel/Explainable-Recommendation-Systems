from pathlib import Path
import pandas as pd
import csv
import re

DATA_DIR = Path("data/amazon/processed")
ITEMS_PATH = DATA_DIR / "items_with_images.parquet"
OUT_DIR = Path("data/kg")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def simple_entities(text: str):
    # fallback simple extractor: split on non-word, keep tokens longer than 3
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    tokens = [t for t in tokens if len(t) > 3]
    # lowercase and dedupe while preserving order
    seen = set()
    out = []
    for t in tokens:
        tt = t.lower()
        if tt in seen:
            continue
        seen.add(tt)
        out.append(tt)
    return out[:20]


def extract():
    if not ITEMS_PATH.exists():
        print(f"Missing items file: {ITEMS_PATH}")
        return

    df = pd.read_parquet(ITEMS_PATH)
    df = df.astype(object).fillna("")

    triples_path = OUT_DIR / "triples.csv"
    entities_path = OUT_DIR / "entities.csv"

    ent_counts = {}

    with triples_path.open("w", newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(["subject", "predicate", "object"])  # headers

        for r in df.itertuples(index=False):
            item_idx = int(getattr(r, "item_idx", -1) or -1)
            title = str(getattr(r, "title", "") or "")
            desc = str(getattr(r, "description", "") or "")
            category = str(getattr(r, "main_category", "") or "")

            # category triple
            if category:
                writer.writerow([item_idx, "has_category", category])
                ent_counts.setdefault(category.lower(), 0)
                ent_counts[category.lower()] += 1

            # title entities
            ents = simple_entities(title)
            for e in ents:
                writer.writerow([item_idx, "has_entity", e])
                ent_counts.setdefault(e, 0)
                ent_counts[e] += 1

            # description entities (if any)
            ents2 = simple_entities(desc)
            for e in ents2:
                writer.writerow([item_idx, "has_entity", e])
                ent_counts.setdefault(e, 0)
                ent_counts[e] += 1

    # write entities summary
    with entities_path.open("w", newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(["entity", "count"])
        for e, c in sorted(ent_counts.items(), key=lambda x: -x[1])[:1000]:
            writer.writerow([e, c])

    print(f"Wrote triples to {triples_path} and entities to {entities_path}")


if __name__ == '__main__':
    extract()
