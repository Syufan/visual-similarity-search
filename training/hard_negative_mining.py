# Phase 1 — Offline hard negative mining
# Every N epochs: re-index full gallery, mine top-K hardest negatives per anchor

import numpy as np
import faiss


def mine_hard_negatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> dict:
    """
    Returns {anchor_idx: [hard_negative_idx, ...]} for each anchor.
    Excludes pairs that share the same label (i.e., true positives).
    """
    # TODO: build flat L2 index → search top-(k+1) → filter out positives
    pass
