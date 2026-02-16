from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

MM_PATH = Path("data/embeddings/multimodal_embeddings.parquet")
OUT_IDX = Path("data/embeddings/multimodal.faiss")


def main():
    if not MM_PATH.exists():
        print(f"Missing multimodal embeddings: {MM_PATH}")
        return

    df = pd.read_parquet(MM_PATH)
    ids = df['item_idx'].astype(int).tolist()
    X = np.vstack(df['embedding'].values).astype(np.float32)

    try:
        import faiss

        # normalize for inner product
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = (X / norms).astype(np.float32)
        dim = Xn.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(Xn)
        faiss.write_index(idx, str(OUT_IDX))
        print(f"Built FAISS index at {OUT_IDX} (n={X.shape[0]}, dim={dim})")
    except Exception as e:
        print("FAISS not available or failed to build:", e)
        print("Install faiss-cpu or faiss-gpu to build fast indexes.")


if __name__ == '__main__':
    main()
