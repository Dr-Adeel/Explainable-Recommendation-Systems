from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explain.shap_surrogate import train_and_save_surrogate


MM_PATH = Path("data/embeddings/multimodal_embeddings.parquet")
OUT = Path("data/models/surrogate_rf.joblib")


def main():
    train_and_save_surrogate(MM_PATH, OUT, n_samples=20000)


if __name__ == '__main__':
    main()
