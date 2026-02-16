from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def _to_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, int]]:
    # df expected to have columns ['item_idx','embedding'] where embedding is list-like
    ids = df['item_idx'].astype(int).tolist()
    mat = np.vstack([np.asarray(x, dtype=np.float32) for x in df['embedding'].values])
    idx_map = {int(i): idx for idx, i in enumerate(ids)}
    return mat, idx_map


def fuse_embeddings(image_df: pd.DataFrame, text_df: pd.DataFrame, image_weight: float = 0.6, text_weight: float = 0.4) -> pd.DataFrame:
    # align on item_idx
    im_mat, im_map = _to_matrix(image_df)
    tx_mat, tx_map = _to_matrix(text_df)

    common_ids = sorted(set(im_map.keys()) & set(tx_map.keys()))
    if not common_ids:
        raise ValueError('No common item_idx between image and text embeddings')

    emb_list = []
    out_ids = []
    for iid in common_ids:
        im_vec = im_mat[im_map[iid]]
        tx_vec = tx_mat[tx_map[iid]]
        # if dims differ, pad smaller with zeros
        if im_vec.shape[0] != tx_vec.shape[0]:
            if im_vec.shape[0] > tx_vec.shape[0]:
                pad = np.zeros(im_vec.shape[0] - tx_vec.shape[0], dtype=np.float32)
                tx_vec = np.concatenate([tx_vec, pad])
            else:
                pad = np.zeros(tx_vec.shape[0] - im_vec.shape[0], dtype=np.float32)
                im_vec = np.concatenate([im_vec, pad])

        # normalize each modality
        imn = im_vec / (np.linalg.norm(im_vec) + 1e-12)
        txn = tx_vec / (np.linalg.norm(tx_vec) + 1e-12)

        fused = image_weight * imn + text_weight * txn
        # final normalization
        fused = fused.astype(np.float32)
        fused = fused / (np.linalg.norm(fused) + 1e-12)

        out_ids.append(int(iid))
        emb_list.append(fused.tolist())

    out_df = pd.DataFrame({'item_idx': out_ids, 'embedding': emb_list})
    return out_df


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
