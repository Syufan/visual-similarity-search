---
title: Visual Similarity Search
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Visual Similarity Search — Deep Metric Learning on TLL

A portfolio project demonstrating end-to-end image retrieval: from metric learning research to a production-ready serving pipeline.

**Task:** Given a query image, retrieve the most *human-perceived* similar image from a candidate pool — spanning facial, shape, color/texture, and semantic/humor similarity simultaneously.

**Dataset:** [Totally-Looks-Like](https://totally-looks-like.com/) — 2,000 curated image pairs from Reddit r/totallynotrobots.

---

## Results

| Model | AUC | MAP@1 | MAP@5 | MAP@10 | NDCG@10 |
|-------|-----|-------|-------|--------|---------|
| ResNet152 (baseline, no fine-tune) | 0.697 | 0.083 | 0.167 | 0.240 | 0.153 |
| ResNet152 + Triplet / Batch Hard Mining | 0.787 | 0.147 | 0.263 | 0.357 | 0.235 |
| **CLIP ViT-B/32 + LoRA** | **0.925** | **0.293** | **0.547** | **0.710** | **0.481** |

> Evaluated on a 15% held-out validation split (~300 pairs). CLIP + LoRA achieves **2× MAP@10** vs fine-tuned ResNet152 while training only **2% of parameters**.

---

## Architecture

```
Query Image
    ↓
CLIP ViT-B/32  (frozen weights)
    + LoRA adapters  (r=8, trainable — 2% of params)
    ↓
[CLS] token  →  Projection Head (768 → 512 → 256)
    ↓
L2-normalized 256-dim embedding
    ↓
FAISS IVFFlat index  →  Top-K candidates
```

**Loss:** Triplet Loss + Batch Hard Mining — selects the hardest negative per anchor within each batch, avoiding vanishing gradients from easy triplets.

**Why CLIP + LoRA over full fine-tuning:**
TLL similarity spans facial, shape, and *semantic/humor* dimensions. CLIP's vision-language pretraining (400M image-text pairs) encodes cross-domain semantics that ResNet cannot capture. LoRA constrains the update to a low-rank subspace, preventing overfitting on the 2,000-pair training set.

---

## Project Structure

```
├── notebooks/
│   └── image_retrieval_algo-results.ipynb   # full pipeline: data → train → eval → FAISS
├── result/
│   ├── gallery_embeddings.npy               # precomputed 2000×256 gallery embeddings
│   ├── gallery.index                        # FAISS IVFFlat index (ready to query)
│   ├── submission_clip.csv                  # CLIP+LoRA test predictions
│   ├── submission_resnet152.csv             # ResNet152 test predictions
│   ├── ablation_runs_final.csv              # raw metrics for 48 ablation runs
│   ├── ablation_summary.csv                 # per-experiment aggregated results
│   └── experiment_log.json                  # full training logs
├── .gitignore
├── commit.md                                # commit message convention
├── LICENSE
└── README.md
```

---

## Reproduce

```bash
# 1. Install dependencies
pip install torch torchvision open_clip_torch faiss-cpu scikit-learn

# 2. Place data under data/ following the structure in the notebook header

# 3. Run the notebook end-to-end
jupyter notebook notebooks/image_retrieval_algo-results.ipynb
```

Pre-computed embeddings and the FAISS index are committed under `result/` — you can skip training and jump directly to the retrieval demo cell.

---

## Benchmarks

FAISS IVFFlat retrieval, 2000-vector gallery, 256-dim embeddings, k=20 (CPU, Apple M-series).

**Single-threaded latency (1000 queries)**

| Mean | P50 | P95 | P99 |
|------|-----|-----|-----|
| 0.049 ms | 0.046 ms | 0.050 ms | 0.074 ms |

**Concurrent QPS**

| Concurrency | QPS | P99 (ms) |
|-------------|-----|----------|
| 1 | 12,635 | 0.08 |
| 4 | 28,027 | 0.41 |
| 8 | 28,441 | 1.23 |
| 16 | 27,332 | 3.19 |
| 32 | 21,486 | 2.16 |

System saturates at ~8 concurrent workers (~28K QPS); beyond that, thread contention increases P99 without improving throughput.

---

## Roadmap

- [ ] FastAPI serving endpoint (`POST /search`, `GET /health`)
- [ ] P50/P95/P99 latency profiling + QPS benchmark
- [ ] Offline hard negative mining (full-gallery ANN)
- [ ] Error analysis — failure case gallery by similarity type
- [ ] Dockerfile + docker-compose
- [ ] GitHub Actions CI/CD → Railway deploy

---

## License

[MIT](LICENSE)
