"""
Phase 1 — Offline Hard Negative Mining

Motivation: Batch Hard Mining only sees 64 images per step.
Offline mining searches the *full gallery* for the hardest negatives,
producing much more informative triplets and faster convergence.

Usage (called every N epochs during training):
    negatives = mine_hard_negatives(embeddings, labels, k=10)
    # negatives[i] = list of hard negative indices for anchor i
"""

import numpy as np
import faiss


def mine_hard_negatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> dict[int, list[int]]:
    """
    For each anchor, find the k nearest neighbours that are NOT the true positive.

    Args:
        embeddings: (N, D) float32, L2-normalised
        labels:     (N,)  int — paired samples share the same label
                    Convention: anchor i and its positive j have labels[i] == labels[j]
        k:          number of hard negatives to mine per anchor

    Returns:
        dict mapping anchor_idx → [hard_negative_idx, ...]
    """
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    n, d = embeddings.shape

    # flat inner-product index — exact search, no approximation error
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # search k+1 to have room after filtering out the positive and self
    _, indices = index.search(embeddings, k + 2)

    result: dict[int, list[int]] = {}
    for anchor_idx in range(n):
        hard_negatives = []
        for neighbour_idx in indices[anchor_idx]:
            if neighbour_idx == anchor_idx:
                continue
            if labels[neighbour_idx] == labels[anchor_idx]:
                continue  # true positive — skip
            hard_negatives.append(int(neighbour_idx))
            if len(hard_negatives) == k:
                break
        result[anchor_idx] = hard_negatives

    return result


def build_negative_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> list[tuple[int, int]]:
    """
    Flatten mine_hard_negatives into (anchor_idx, negative_idx) pairs
    for direct use in a PyTorch Dataset.
    """
    negatives = mine_hard_negatives(embeddings, labels, k=k)
    return [
        (anchor, neg)
        for anchor, neg_list in negatives.items()
        for neg in neg_list
    ]


def refresh_negatives_every_n_epochs(
    epoch: int,
    n: int,
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> dict[int, list[int]] | None:
    """
    Call at the top of each training epoch.
    Returns fresh hard negatives every N epochs, None otherwise.

    Example:
        negatives = refresh_negatives_every_n_epochs(epoch, N=5, embs, labels)
        if negatives:
            dataset.update_negatives(negatives)
    """
    if epoch % n == 0:
        return mine_hard_negatives(embeddings, labels, k=k)
    return None
