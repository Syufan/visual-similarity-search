import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from retrieval.model import get_model, embed
from retrieval.index import load_index, search

HF_DATASET_URL = "https://huggingface.co/datasets/Yufanjeff/tll-images/resolve/main/right/{name}.jpg"


class RetrievalPipeline:
    def __init__(
        self,
        checkpoint_path: str,
        index_path: str,
        device: str = "cpu",
        filenames_path: Optional[str] = None,
    ):
        self.model, self.preprocess = get_model(checkpoint_path, device)
        self.index = load_index(index_path)
        self.device = device
        self._filenames: Optional[List[str]] = None
        if filenames_path and Path(filenames_path).exists():
            with open(filenames_path) as f:
                self._filenames = json.load(f)

    def _url(self, idx: int) -> Optional[str]:
        if self._filenames and 0 <= idx < len(self._filenames):
            return HF_DATASET_URL.format(name=self._filenames[idx])
        return None

    def query(self, image: Image.Image, k: int = 20) -> List[dict]:
        tensor = self.preprocess(image).unsqueeze(0)
        vec = embed(tensor, self.device).numpy()
        distances, indices = search(self.index, vec, k)
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            entry = {"rank": i + 1, "index": int(idx), "score": float(dist)}
            url = self._url(int(idx))
            if url:
                entry["image_url"] = url
            results.append(entry)
        return results
