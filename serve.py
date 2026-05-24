import io
import os
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File
from PIL import Image

from retrieval.pipeline import RetrievalPipeline

logging.basicConfig(level=logging.INFO, format='{"ts":"%(asctime)s","msg":"%(message)s"}')
logger = logging.getLogger(__name__)

_pipeline: RetrievalPipeline = None
API_KEY = os.environ.get("API_KEY", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = RetrievalPipeline(
        checkpoint_path=os.environ["MODEL_PATH"],
        index_path=os.environ["INDEX_PATH"],
        device=os.environ.get("DEVICE", "cpu"),
    )
    logger.info("pipeline ready")
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
