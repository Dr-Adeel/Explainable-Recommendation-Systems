from pathlib import Path
import json
import csv

candidates = [
    Path("data/amazon/processed/als/csr_stats.json"),
    Path("data/amazon/processed_small/als/csr_stats.json"),
]

found = []
for p in candidates:
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            n_users, n_items = d.get("shape", [None, None])
            rec = {
                "path": str(p),
                "n_users": int(n_users) if n_users is not None else None,
                "n_items": int(n_items) if n_items is not None else None,
                "nnz": int(d.get("nnz", 0)),
                "density": float(d.get("density", 0.0)),
                "value_min": float(d.get("value_min", 0.0)),
                "value_max": float(d.get("value_max", 0.0)),
                "value_mean": float(d.get("value_mean", 0.0)),
            }
            found.append(rec)
        except Exception as e:
            print(f"Failed to read {p}: {e}")

if not found:
    print("No csr_stats.json found in expected locations.")
    raise SystemExit(1)

# Print human readable
for r in found:
    print("= csr_stats from:", r['path'])
    print(f"n_users: {r['n_users']}")
    print(f"n_items: {r['n_items']}")
    print(f"nnz: {r['nnz']}")
    print(f"density: {r['density']:.5f}")
    print(f"value_min: {r['value_min']}")
    print(f"value_max: {r['value_max']}")
    print(f"value_mean: {r['value_mean']}")
    print()

# write CSV summary
out_dir = Path("reports")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "csr_stats_excerpt.csv"
with out_csv.open("w", newline='', encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["path","n_users","n_items","nnz","density","value_min","value_max","value_mean"])
    w.writeheader()
    for r in found:
        w.writerow(r)

print("Wrote:", out_csv)
