from __future__ import annotations

from typing import List, Optional
import numpy as np


class TextEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", fallback_dim: int = 256):
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self._model = None
        self._use_transformer = False
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self._use_transformer = True
        except Exception:
            # sentence-transformers not available, will fallback to TF-IDF + SVD
            self._model = None
            self._use_transformer = False

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if self._use_transformer and self._model is not None:
            # encoder returns numpy array
            emb = self._model.encode(texts, batch_size=batch_size, show_progress_bar=False)
            return np.asarray(emb, dtype=np.float32)

        # fallback: TF-IDF + TruncatedSVD to produce dense vectors
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(self.fallback_dim, X.shape[1] - 1 or 1))
        Xr = svd.fit_transform(X)
        # ensure consistent dim
        if Xr.shape[1] < self.fallback_dim:
            pad = np.zeros((Xr.shape[0], self.fallback_dim - Xr.shape[1]), dtype=np.float32)
            Xr = np.hstack([Xr.astype(np.float32), pad])
        return Xr.astype(np.float32)


def get_encoder(model_name: str = "all-MiniLM-L6-v2", fallback_dim: int = 256) -> TextEncoder:
    return TextEncoder(model_name=model_name, fallback_dim=fallback_dim)
