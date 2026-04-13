# ClickLens — YouTube Thumbnail Click-Through Predictor

> **Live app:** https://clicklens-6x9m.onrender.com
> **Backend API:** https://clicklens-j2r2.onrender.com/health

ClickLens is a computer-vision tool that predicts which YouTube thumbnail will get the most clicks. Upload 2–4 candidate thumbnails and get ranked predictions with Grad-CAM explainability heatmaps.

## Team
- **Lindsay Gross** — Data Pipeline
- **Arnav Mahale** — Modeling
- **Sharmil Nanjappa** — Application

## Quick Start

### 1. Clone & install Python dependencies
```bash
git clone <repo-url> && cd ClickLens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Add your YouTube Data API key to .env
```

### 3. Collect data
```bash
python scripts/scrape_youtube.py
```

### 4. Build features & labels
```bash
python scripts/build_features.py
```

### 5. Train models
```bash
python scripts/train_baseline.py
python scripts/train_xgboost.py
python scripts/train_efficientnet.py
```

### 6. Evaluate
```bash
python scripts/evaluate.py
```

### 7. Build the thumbnail recommender
Run these in order after `build_features.py` has produced `features.csv`.
If `train_efficientnet.py` has also produced `models/efficientnet_best.pth`,
`extract_embeddings.py` will use the fine-tuned checkpoint (recommended for
better embedding quality); otherwise it falls back to ImageNet-pretrained
weights, so the pipeline can still run without the checkpoint:
```bash
python scripts/extract_embeddings.py          # 1280-d backbone features per thumbnail
python scripts/build_index.py                 # cosine-similarity kNN index
python scripts/train_baseline_recommender.py  # random + niche-mean-CTR baselines
```
Artifacts produced:
- `data/processed/embeddings.npy` — `(N, 1280)` float32 array
- `data/processed/embeddings_mapping.csv` — `video_id, niche, CTR_label, thumbnail_path`
- `models/knn_index.pkl` — pickled `{index, mapping, embed_dim}` loaded by the backend
- `data/outputs/baseline_recommender_metrics.json` — precision@K / recall@K

> `features.csv` schema: `niche, thumbnail_path, views, subscribers, CTR_label`, then the visual feature columns.

### 8. Run the app

**Backend** (port 8000):
```bash
cd app/backend && uvicorn main:app --reload --port 8000
```

**Frontend** (port 5173):
```bash
cd app/frontend && npm install && npm run dev
```

Open http://localhost:5173 in your browser.

## Experiments

### Cross-niche generalization
```bash
python scripts/experiment_cross_niche.py
```

### Grad-CAM visualization
```bash
python scripts/gradcam.py --image path/to/thumbnail.jpg --output heatmap.jpg
```

## Repo Structure
```
ClickLens/
├── scripts/           # Data pipeline, training, evaluation, experiments
├── models/            # Saved model weights & artifacts
├── data/
│   ├── raw/           # Downloaded thumbnails
│   ├── processed/     # CSVs with metadata & features
│   └── outputs/       # Evaluation results & plots
├── app/
│   ├── backend/       # FastAPI inference server
│   └── frontend/      # React (Vite) UI
├── notebooks/         # Exploration only
└── requirements.txt
```

## Tech Stack
- **Models**: EfficientNet-B0 (PyTorch/timm), XGBoost, Majority-class baseline
- **Recommender**: EfficientNet-B0 embeddings + sklearn NearestNeighbors (cosine)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: React + Vite
- **Explainability**: Grad-CAM
- **Data**: YouTube Data API v3
