from pathlib import Path
import pandas as pd

# Choose processed or processed_small depending on availability
candidates = [
    Path("data/amazon/processed_small/interactions_train.parquet"),
    Path("data/amazon/processed/interactions_train.parquet"),
]
parquet_path = None
for p in candidates:
    if p.exists():
        parquet_path = p
        break

if parquet_path is None:
    raise FileNotFoundError("No interactions_train.parquet found in processed or processed_small")

print(f"Loading interactions from: {parquet_path}")

df = pd.read_parquet(parquet_path)

# determine weight column
if "value" in df.columns:
    df["weight"] = df["value"].astype(float)
elif "rating" in df.columns:
    df["weight"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).astype(float)
else:
    df["weight"] = 1.0

# ensure integer indices if present
if "user_idx" in df.columns:
    df["user_idx"] = df["user_idx"].astype(int)
if "item_idx" in df.columns:
    df["item_idx"] = df["item_idx"].astype(int)

out = df[[c for c in ["user_idx", "item_idx", "weight"] if c in df.columns]].head(200)

out_dir = Path("reports")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "interactions_sample.csv"
out.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(out.to_string(index=False))
