# TLL Metric Learning — Project Summary

> 目标：算法岗作品集，展示从 research 到工业落地的完整深度。

---

## 1. 任务定义

**Totally-Looks-Like (TLL) Challenge**

给定一张 query 图像，从若干候选图中找出**人类认为最相似的那一张**。

这不是分类任务，也不是语义匹配任务。核心难点在于：TLL 的"相似"没有统一标准：

| 相似类型 | 示例 |
|---------|------|
| 面部相似 | 两张人脸的五官、轮廓接近 |
| 形状相似 | 一个人的发型和某动物的头部轮廓像 |
| 颜色/纹理相似 | 橙色的夕阳和橙色的南瓜 |
| 语义/幽默联想 | 懒散的猫和某位名人的表情 |

模型必须在**同一个 embedding 空间**里同时捕捉这四种异质相似性，这使得任何单一归纳偏置的模型都会在某些子类型上失效。

---

## 2. 数据集

| 属性 | 值 |
|------|---|
| 训练集 | 2000 对图像（left/right 两侧） |
| 测试集 | 2000 张 query，每张配 20 个候选 |
| Ground Truth | train.csv 提供正样本对；测试集为 test_candidates.csv |
| 图像来源 | Reddit r/totallynotrobots 用户提交，人工策划 |
| 标注质量 | 高质量，无需清洗；但多样性极强，类内方差大 |

**数据结构：**
```
data/
├── train.csv
├── test_candidates 2.csv  # columns: left(query), c0..c19(candidates)
├── sample-submission.csv  # 提交格式示例
├── train/
│   ├── left/              # anchor 图像（2000 张）
│   └── right/             # positive 图像（2000 张）
└── test 2/
    ├── left/              # test query 图像（2000 张）
    └── right/             # test candidate 池（2000 张，每 query 从中取 20 个）
```

---

## 3. 核心挑战

1. **无统一相似标准**：多维度异质相似性，embedding 空间需要兼容多种特征
2. **小数据集**：2000 对训练样本，预训练迁移至关重要，全量微调容易过拟合
3. **负样本定义模糊**：任意非配对图像都是负样本，但某些"负样本"实际上也很相似
4. **评估指标多维**：Nearest Neighbor Search 准确率低（ResNet152 baseline 仅 3.83%），AUC 和 MAP@K 更能反映真实排序质量

---

## 4. 技术方案

### 4.1 模型架构

**Experiment 1 — Baseline**
- ResNet152（ImageNet 预训练），不做 Triplet 训练
- 直接提取 2048 维特征，接 Projection Head 降到 256 维
- 目的：确立 before fine-tuning 的指标下限

**Experiment 2 — Fine-tuned ResNet152**
- ResNet152 骨干 + 两层 Projection Head (2048→512→256)
- Triplet Loss + Batch Hard Mining 端到端微调
- 输出 L2 归一化 embedding，内积 = cosine similarity

**Experiment 3 — CLIP + LoRA**
- OpenAI CLIP ViT-B/32 视觉编码器（4 亿图文对预训练）
- 自实现 LoRA（Low-Rank Adaptation）：冻结原始权重，只训练低秩矩阵
  - `ΔW = BA`，`B ∈ R^(d×r)`，`A ∈ R^(r×k)`，`r=8`
  - 可训练参数仅为全量微调的 **2%**
- [CLS] token → Projection Head (768→512→256)
- 动机：CLIP 的语义表示对 TLL 跨域相似性（语义、幽默联想）更友好

### 4.2 损失函数

**Triplet Loss + Batch Hard Mining**

$$\mathcal{L} = \mathbb{E}\left[\max\left(0,\ d(a,p) - d(a,n^*) + m\right)\right]$$

其中 $n^* = \arg\min_{n \in \text{batch}} d(a, n)$（batch 内最难负样本）。

随机负样本策略会导致大多数 triplet loss=0，梯度停滞。Batch Hard Mining 每步只在最难的样本对上计算梯度，收敛更快，泛化更好。

### 4.3 训练工程

| 技术 | 配置 | 作用 |
|------|------|------|
| AMP (fp16) | `use_amp=True` | 显存和速度均提升约 2× |
| Gradient Checkpointing | `use_grad_checkpoint=False`（4090 关闭） | 以计算换显存，小 GPU 开启 |
| Gradient Accumulation | `grad_accumulation_steps=1` | 等效大 batch，4090 已够 |
| Warmup + Cosine Annealing | `warmup_epochs=2` | 防训练初期震荡，平滑收敛 |
| Early Stopping | `patience=5` | 防过拟合，节省算力 |
| 断点续训 | 每 epoch 自动保存 rolling checkpoint | 中途崩溃可恢复，不浪费费用 |

**RTX 4090 推荐配置：**
- `batch_size=64`，`num_workers=8`
- 训练数据路径：`/root/autodl-tmp/cv project data/`
- Checkpoint 路径：`/root/autodl-tmp/checkpoints/`（持久盘）

### 4.4 检索加速

**FAISS IVFFlat**（Inverted File Index）

- Gallery embedding 做 k-means 分成 `nlist` 个簇
- 查询只扫描最近的 `nprobe` 个簇，复杂度从 O(N) 降至 O(N·nprobe/nlist)
- 动态 `nlist = min(100, n//10)` 防止小数据集 clustering 失败

---

## 5. 评估体系

| 指标 | 含义 | 为什么用 |
|------|------|---------|
| **AUC** | 正样本对距离 < 负样本对距离的概率 | 对 gallery 大小不敏感，稳定 |
| **MAP@K** | query 的正确答案出现在 top-K 的比例 | 图像检索工业标准 |
| **NDCG@10** | 正样本排名位置的折扣累计增益 | 比 MAP 更考虑排名位置 |
| **NNS Accuracy** | 最近邻正好是正样本的比例 | 最严格，工业部署直接指标 |

**Paper Baseline 结果（无 Triplet 训练）：**

| 模型 | NNS Acc | AUC | MAP@K |
|------|---------|-----|-------|
| VGG16 | 0.60% | 0.6364 | 0.0536 |
| ResNet50 | 3.45% | 0.9792 | 0.3023 |
| ResNet152 | **3.83%** | **0.9630** | **0.3397** |
| EfficientNet B0 | 0.28% | 0.6234 | 0.0245 |
| DenseNet121 | 0.25% | 0.6186 | 0.0215 |

**实验结果（val split = 15%，约 300 对）：**

| 模型 | AUC | MAP@1 | MAP@5 | MAP@10 | NDCG@10 |
|------|-----|-------|-------|--------|---------|
| Baseline ResNet152 | 0.6970 | 0.0833 | 0.1667 | 0.2400 | 0.1531 |
| Fine-tuned ResNet152 | 0.7867 | 0.1467 | 0.2633 | 0.3567 | 0.2346 |
| **CLIP + LoRA** | **0.9254** | **0.2933** | **0.5467** | **0.7100** | **0.4809** |

> **注：** Paper baseline（AUC=0.963）使用全量 2000 对评估；本实验使用 15% val split（~300 对），负样本空间更小，两者评估口径不同，不直接可比。
>
> **关键发现：**
> - Triplet fine-tuning：AUC +9pts，MAP@10 +12pts，但 train loss 0.013 vs val loss 0.217（16×），明显过拟合
> - CLIP+LoRA：MAP@10 是 fine-tuned ResNet 的 **2×**，train/val loss 同步下降（LoRA 正则化生效）
> - 根因：TLL 的跨域相似性（语义/幽默联想）在 CLIP 4亿图文对预训练的语义空间里天然更可分

---

## 6. 工业化路线图

### Phase 1：训练深度

**Offline Hard Negative Mining**
- 当前：Batch Hard Mining 只在一个 batch（64张）内找最难负样本
- 升级：用全库 embedding 做 ANN 检索，找语义接近但非正样本的图像作为 hard negative
- 价值：更难的负样本 → 更鲁棒的 embedding，体现对训练动力学的深度理解
- 实现：每 N epoch 重新 index 全库，mine top-K hardest negatives per anchor

### Phase 2：数据工程

**Near-Duplicate Collapse**
- 对 gallery 做 embedding 相似度矩阵，cosine > 0.98 的视为近重复
- 近重复会导致虚假召回（模型"找到"了相同图片而非真正相似的）
- 输出：去重后的 clean gallery + 重复率统计报告

### Phase 3：系统工程

**Incremental Index Update**
- 对比：`index.add()` 增量更新 vs 全量 rebuild 的延迟和吞吐
- 生产场景：新图片每日入库，不能每次重建
- 量化：N=1000 新增时，增量 vs 重建的时间差

**P50/P95/P99 Latency Profiling**
- 当前 benchmark 只有均值，生产看的是 P99
- 实现：1000 次查询的延迟分布直方图 + 分位数统计

**并发 / QPS 压测**
- 生产关注的是并发吞吐，而非单查询延迟
- 实现：用 `concurrent.futures` 模拟 N=1/4/8/16/32 并发请求，记录每档 QPS 和 P99
- 输出：吞吐-延迟曲线，找到系统饱和点（QPS 不再线性增长的拐点）
- 价值：量化 FAISS 在 CPU 多线程下的水平扩展上限，为是否需要 GPU-FAISS 提供决策依据

### Phase 4：生产系统设计

---

#### 4.1 Embedding Drift 监控与 Re-index

**问题：** 模型训练后静止，但线上数据分布持续变化（用户上传新内容、流行风格迁移）。Embedding 空间的"语义坐标系"逐渐失效，表现为线上 AUC 缓慢下滑，但离线指标不变——这是最难发现的问题之一。

**检测方案（三层）：**

| 层级 | 方法 | 触发条件 |
|------|------|---------|
| 在线指标 | 每日采样 1000 query 计算 AUC，与 rolling 7 日均值对比 | AUC 连续 3 天下降 >3% |
| 分布检测 | 每周对 5000 张图采样 embedding，计算新旧分布的 Jensen-Shannon 散度 | JS > 0.05 |
| 近邻稳定性 | 对固定 anchor set（100 张），检查 top-5 近邻变化率 | 变化率 >20% |

三层任一触发 → 报警 + 自动发起 retraining job。

**Re-index 策略（零停机）：**
1. 新模型在 shadow 环境重新 index 全量 gallery
2. 双写阶段：新查询同时打到新旧 index，对比结果
3. 流量灰度：新 index 承接 5% → 25% → 100% 流量
4. 原子切换：确认 P99 latency 和 AUC 均达标后，旧 index 下线

---

#### 4.2 Cold Start 策略

**问题：** 新图片入库时没有 fine-tuned embedding，如果直接跳过，新内容永远无法被检索到；如果等待重训，延迟可能是数小时甚至数天。

**分级兜底方案：**

```
新图片入库
    ↓
[立即] 用通用 backbone（未 fine-tune 的 ResNet152）提取特征
       → 加入 gallery，质量低但至少可被检索
    ↓
[异步] 积累 N 张新图（N=500）后，触发增量 fine-tune
       → 用新数据 fine-tune LoRA adapters（复用已训练权重，只需几分钟）
    ↓
[替换] 新 embedding 覆盖 gallery 中的 fallback 特征
```

**关键设计：**
- 通用 backbone embedding 与 fine-tuned embedding 存在分布差异，混在同一 gallery 中会降低检索质量。解决方案：维护两个独立索引，查询时 merge 结果，并在结果中标注置信度。
- LoRA 的优势在此尤为突出：fine-tune 成本极低（只更新 2% 参数），可以频繁触发增量适应而不需要重训完整模型。

---

#### 4.3 Vector DB Sharding（亿级扩展）

**规模估算：**

| 规模 | 向量数 | 256-dim float32 存储 | 单机 FAISS |
|------|--------|---------------------|-----------|
| 当前（实验） | 2,000 | 2 MB | 可行 |
| 中型平台 | 100 万 | 1 GB | 可行（内存允许） |
| 大型平台 | 1 亿 | 100 GB | 需要分片 |

**水平分片方案：**

```
Query
  ↓
Router（一致性哈希，按 item_id % N_shards 路由）
  ↓
┌──────┬──────┬──────┬──────┐
│ Shard 0 │ Shard 1 │ Shard 2 │ Shard N │   ← 每个 shard 独立 FAISS IVF 索引
└──────┴──────┴──────┴──────┘
  ↓
Merger（收集各 shard top-K，全局归并，返回最终 top-K）
```

**工具选型对比：**

| 方案 | 适合场景 | 优势 | 劣势 |
|------|---------|------|------|
| FAISS IndexShards | 自建系统，延迟敏感 | 极低延迟，完全可控 | 需要自己管理分片、故障恢复 |
| Milvus | 生产级托管，团队规模中等 | 内置分片、副本、监控 | 引入重依赖，运维复杂度高 |
| Qdrant / Weaviate | 云原生，快速上线 | 托管服务，按需扩缩 | 数据出域，成本较高 |

**本项目的扩展路径：** FAISS IVFFlat（当前）→ FAISS HNSW（更快查询，更高内存）→ FAISS IndexShards（多机）→ Milvus（运营级）。

---

#### 4.4 Retrain 触发机制与部署流水线

**触发类型（优先级从高到低）：**

| 触发器 | 条件 | 典型场景 |
|--------|------|---------|
| 在线指标告警 | AUC 连续下降 >3% | 数据分布突变（如热点事件） |
| 数据量触发 | 新增标注对 ≥ 1000 | 定期众包标注完成 |
| Drift 告警 | JS 散度超阈值 | 渐进式分布偏移 |
| 定时触发 | 每周一 02:00 | 兜底，确保模型最多滞后 7 天 |

**部署流水线（CI/CD for ML）：**

```
触发 → 数据准备（新旧数据 mix，比例 7:3）
     → 训练（从 best_clip.pt 热启动，只需训练 LoRA adapters）
     → 离线评估（必须在 held-out set 上 AUC 超过当前线上模型 +1%）
     → Shadow 测试（新模型并行服务，不影响线上，比对结果分布）
     → Canary 发布（5% 流量 → 观察 24h → 25% → 100%）
     → 全量切换 + 旧 index 保留 48h（用于回滚）
```

**回滚条件（任一触发即回滚）：**
- 新模型 P99 latency 超过 SLA（如 >50ms）
- 线上 AUC 在切换后 24h 内下降 >2%
- 异常报错率 >0.1%

---

#### 4.5 在线 Serving API

**问题：** Notebook 只能离线批量跑，生产系统需要实时响应单张图的检索请求。

**方案：FastAPI + FAISS 封装**

```
POST /embed          # 单张图 → 256-dim embedding（base64 or multipart）
POST /search         # 单张图 → top-K 候选列表（含 score、候选 ID）
GET  /health         # 健康检查，返回模型版本 + index 大小
```

**设计要点：**
- 模型在 worker 启动时一次性 load（`@app.on_event("startup")`），避免每次请求重新加载
- CLIP 推理用 `torch.no_grad()` + `torch.cuda.amp.autocast()`，减少显存 + 提速
- FAISS 查询线程安全，可直接多 worker 复用同一 index 对象
- 请求输入做 schema 校验（Pydantic），拒绝非图像格式，防止 crash 穿透到模型层

**API 鉴权：**
- 每个请求 Header 携带 `X-API-Key`，服务端对比环境变量中的密钥
- 未通过鉴权返回 `401 Unauthorized`，防止公网滥用
- Key 通过 Railway 环境变量注入，不硬编码进代码

**结构化请求日志：**
- 每次推理记录：`timestamp / request_id / input_size / embed_latency_ms / search_latency_ms / top1_score`
- 输出为 JSON 格式，方便后续接入 ELK / Datadog 等日志平台
- 异常请求单独记录 `error_type`，用于排查线上问题
- 价值：这是 Drift 监控（4.1）的数据来源，也是 SLA 告警的基础

**对比：REST vs gRPC**

| | REST (FastAPI) | gRPC |
|--|--|--|
| 延迟 | ~2-5ms 协议开销 | ~0.5ms |
| 开发成本 | 低，直接上线 | 高，需定义 proto |
| 适用场景 | 内部服务、demo | 高频调用的微服务 |

本项目选 REST，足够展示 serving 能力，gRPC 留作扩展方向。

---

#### 4.6 Error Analysis（失败案例分析）

**问题：** 指标（AUC=0.945）是全局均值，掩盖了系统性失败模式——哪类图片模型最差，决定了下一步优化的方向。

**实现：**
1. 对 val set 每条 query，记录 ground truth 的排名位置（rank 1 = 最近邻）
2. 按 rank 分桶：rank=1（完美）/ rank 2-5（良好）/ rank 6-10（一般）/ rank>10（失败）
3. 随机抽取 rank>10 的 case，可视化：query 图 + 模型 top-3 预测 + 真实正样本
4. 人工归因：失败原因是颜色/纹理混淆？语义联想缺失？还是负样本噪声？

**输出：**
- 失败 case gallery（图文对，直接贴入报告）
- 失败类型分布饼图（形状混淆 / 颜色混淆 / 语义缺失 / 其他）
- 针对性改进方向（如语义失败多 → 考虑 CLIP text encoder 辅助；颜色混淆多 → 考虑 HSV 颜色直方图融合）

**价值：** 这是大多数同学作品集里没有的内容，体现对模型行为的深度理解，而非只交 metric。

---

#### 4.8 GitHub Actions CI/CD

**目标：** push 到 `main` 分支后自动部署到 Railway，代码即上线，无需手动操作。

**流水线（`.github/workflows/deploy.yml`）：**

```
push to main
    ↓
[CI] 运行测试（pytest serve_test.py）
    ↓
[CI] 构建 Docker 镜像，验证启动成功（health check）
    ↓
[CD] 触发 Railway redeploy（通过 Railway Deploy Hook URL）
    ↓
Railway 拉取最新代码 → 构建镜像 → 滚动更新服务
```

**关键设计：**
- CI 阶段失败（测试不过 / 镜像构建失败）则阻断部署，保护线上服务
- Railway Deploy Hook 存储在 GitHub Secrets，不暴露在代码中
- 每次部署在 GitHub Actions 页面留有完整记录，可追溯任意版本的部署状态

**价值：** 将手动部署变为自动化，体现 DevOps 工程规范；配合 4.4 的回滚机制，形成完整的发布→监控→回滚闭环。

---

#### 4.7 容器化（Dockerfile）

**目标：** 一键复现，任何环境都能运行，而不依赖本地 conda 配置。

**Dockerfile 结构：**
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# 推理服务
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose（完整栈）：**
```yaml
services:
  retrieval-api:
    build: .
    ports: ["8000:8000"]
    volumes:
      - ./result:/app/result   # 挂载 index 和 checkpoint，不打包进镜像
    environment:
      - MODEL_PATH=/app/result/best_clip_deploy.pt
      - INDEX_PATH=/app/result/gallery.index
```

**要点：**
- checkpoint 和 index 通过 volume 挂载，镜像不含模型权重（避免镜像过大）
- `requirements.txt` 固定版本号，保证环境一致性
- CPU-only 镜像用 `pytorch/pytorch:2.1.0-cpu`，大幅减小镜像体积（5GB → 1.5GB）

---

## 7. 文件结构

```
tll-metric-learning/
├── README.md                    # 项目介绍 + live demo 链接 + 实验结果表格 + build badge
├── image_retrieval_algo.ipynb   # 主 notebook（ResNet152 + CLIP+LoRA）
├── serve.py                     # FastAPI serving endpoint（Phase 4.5）
├── retrieval/
│   ├── model.py                 # CLIP+LoRA 加载，singleton pattern
│   ├── index.py                 # FAISS 封装
│   └── pipeline.py              # embed → search 流程
├── analysis/
│   └── error_analysis.ipynb     # 失败案例分析（Phase 4.6）
├── benchmarks/
│   └── latency_qps.py           # P99 + QPS 压测（Phase 3）
├── .github/
│   └── workflows/
│       └── deploy.yml           # GitHub Actions CI/CD（Phase 4.8）
├── Dockerfile                   # 容器化（Phase 4.7）
├── docker-compose.yml
├── requirements.txt
├── data/                        # 数据集
│   ├── train.csv
│   ├── test_candidates 2.csv
│   ├── train/left & right/      # 2000 训练对
│   └── test 2/left & right/     # 2000 测试 query + 候选池
├── result/                      # 实验产出
│   ├── best_clip_deploy.pt      # 最佳模型权重
│   ├── gallery_embeddings.npy   # 全库 embedding (2000×256)
│   ├── gallery.index            # FAISS IVFFlat 索引
│   ├── submission_clip.csv      # CLIP+LoRA 提交文件
│   ├── submission_resnet152.csv # ResNet152 提交文件
│   ├── ablation_runs_final.csv  # 48 组实验原始数据
│   ├── ablation_summary.csv     # 按 experiment 聚合结果
│   └── experiment_log.json      # 完整实验日志
└── PROJECT_SUMMARY.md           # 本文件
```

---

## 8. 关键设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| Loss function | Triplet > Contrastive | 直接优化相对距离，对多维度相似性泛化更好 |
| Backbone | ResNet152 + CLIP | ResNet152 是 paper baseline 对比点；CLIP 语义表示对跨域相似性更强 |
| LoRA vs Full FT | LoRA | 2000 对样本全量微调 CLIP 必然过拟合；LoRA 只训练 2% 参数 |
| Embedding dim | 256 | 平衡检索精度（高维更精确）和 FAISS 速度（低维更快） |
| Distance metric | Cosine | 尺度不变性，对归一化 embedding 内积计算高效 |
| Mining strategy | Batch Hard | 随机负样本多数 loss=0，梯度停滞；Hard Mining 每步梯度有效 |

---

*Last updated: 2026-05-24 (新增 Phase 3 QPS压测 + Phase 4.5 Serving API + 4.6 Error Analysis + 4.7 容器化 + 4.8 GitHub Actions CI/CD + API 鉴权 + 结构化日志 + README 徽章)*
*Author: Yufan Zhang*
*Target: Algorithm Engineering Position*
