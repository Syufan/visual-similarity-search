from typing import List

import numpy as np
import torch
from PIL import Image

from retrieval.model import get_model, embed
from retrieval.index import load_index, search


class RetrievalPipeline:
    def __init__(self, checkpoint_path: str, index_path: str, device: str = "cpu"):
        self.model, self.preprocess = get_model(checkpoint_path, device)
        self.index = load_index(index_path)
        self.device = device

    def query(self, image: Image.Image, k: int = 20) -> List[dict]:
        tensor = self.preprocess(image).unsqueeze(0)
        vec = embed(tensor, self.device).numpy()
        distances, indices = search(self.index, vec, k)
        return [
            {"rank": i + 1, "index": int(idx), "score": float(dist)}
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0]))
        ]
