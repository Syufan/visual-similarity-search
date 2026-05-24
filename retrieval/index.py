import numpy as np
import faiss


def build_index(embeddings: np.ndarray) -> faiss.Index:
    n, d = embeddings.shape
    nlist = min(100, n // 10)
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(embeddings)
    index.train(embeddings)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: str) -> None:
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)


def search(index: faiss.Index, query: np.ndarray, k: int = 20):
    q = query.copy()
    faiss.normalize_L2(q)
    distances, indices = index.search(q, k)
    return distances, indices
