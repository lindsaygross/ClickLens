.PHONY: install scrape features train evaluate app-backend app-frontend experiment gradcam all

install:
	pip install -r requirements.txt
	cd app/frontend && npm install

scrape:
	python scripts/scrape_youtube.py

features:
	python scripts/build_features.py

train-baseline:
	python scripts/train_baseline.py

train-xgboost:
	python scripts/train_xgboost.py

train-efficientnet:
	python scripts/train_efficientnet.py

train: train-baseline train-xgboost train-efficientnet

evaluate:
	python scripts/evaluate.py

experiment:
	python scripts/experiment_cross_niche.py

gradcam:
	python scripts/gradcam.py

app-backend:
	cd app/backend && uvicorn main:app --reload --port 8000

app-frontend:
	cd app/frontend && npm run dev

all: scrape features train evaluate
