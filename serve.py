import io
import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File
from PIL import Image

from retrieval.pipeline import RetrievalPipeline

HF_REPO_ID = "Yufanjeff/visual-similarity-search-model"
CHECKPOINT_FILENAME = "best_clip_deploy.pt"


def _ensure_checkpoint(path: str) -> str:
    if not Path(path).exists():
        logging.getLogger(__name__).info(f"Downloading model from HF Hub: {HF_REPO_ID}")
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=CHECKPOINT_FILENAME)
    return path

logging.basicConfig(level=logging.INFO, format='{"ts":"%(asctime)s","msg":"%(message)s"}')
logger = logging.getLogger(__name__)

_pipeline: RetrievalPipeline = None
API_KEY = os.environ.get("API_KEY", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    if os.environ.get("TESTING") != "1":
        checkpoint_path = _ensure_checkpoint(os.environ.get("MODEL_PATH", CHECKPOINT_FILENAME))
        _pipeline = RetrievalPipeline(
            checkpoint_path=checkpoint_path,
            index_path=os.environ["INDEX_PATH"],
            device=os.environ.get("DEVICE", "cpu"),
        )
        logger.info("pipeline ready")
    else:
        logger.info("TESTING=1: skipping model load")
    yield


app = FastAPI(title="Visual Similarity Search", version="1.0.0", lifespan=lifespan)


def _auth(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _decode(file: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid image")


@app.get("/health")
def health():
    index_size = _pipeline.index.ntotal if _pipeline else -1
    return {"status": "ok", "index_size": index_size}


@app.post("/embed")
async def embed_image(file: UploadFile = File(...), _: None = Depends(_auth)):
    image = _decode(file)
    t0 = time.perf_counter()
    tensor = _pipeline.preprocess(image).unsqueeze(0)
    from retrieval.model import embed
    vec = embed(tensor, _pipeline.device)
    logger.info(f"embed latency={1000*(time.perf_counter()-t0):.1f}ms")
    return {"embedding": vec.squeeze().tolist()}


@app.post("/search")
async def search_image(
    file: UploadFile = File(...),
    k: int = 20,
    _: None = Depends(_auth),
):
    request_id = str(uuid.uuid4())[:8]
    image = _decode(file)

    t0 = time.perf_counter()
    results = _pipeline.query(image, k=k)
    latency_ms = 1000 * (time.perf_counter() - t0)

    logger.info(
        f"request_id={request_id} k={k} "
        f"latency={latency_ms:.1f}ms top1_score={results[0]['score']:.4f}"
    )
    return {"request_id": request_id, "results": results}
