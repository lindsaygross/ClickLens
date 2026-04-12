"""
ClickLens — YouTube Thumbnail Click-Through Predictor

Entry point for running the full pipeline end-to-end.
"""

import subprocess
import sys


def main():
    steps = [
        ("Scraping YouTube data", [sys.executable, "scripts/scrape_youtube.py"]),
        ("Building features & labels", [sys.executable, "scripts/build_features.py"]),
        ("Training baseline", [sys.executable, "scripts/train_baseline.py"]),
        ("Training XGBoost", [sys.executable, "scripts/train_xgboost.py"]),
        ("Training EfficientNet", [sys.executable, "scripts/train_efficientnet.py"]),
        ("Evaluating models", [sys.executable, "scripts/evaluate.py"]),
    ]

    for description, cmd in steps:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print(f"{'='*60}\n")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nFailed at step: {description}")
            sys.exit(1)

    print("\n✓ Pipeline complete. Run the app with:")
    print("  cd app/backend && uvicorn main:app --reload --port 8000")
    print("  cd app/frontend && npm run dev")


if __name__ == "__main__":
    main()
