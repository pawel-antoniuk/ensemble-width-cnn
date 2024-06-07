from pathlib import Path
import pandas as pd
import numpy as np
import joblib

src_models_dir = Path("/app/data/train/out_models")
test_scores = {}
histories = {}
for model_dir in src_models_dir.glob("*"):
    for rep_it_path in model_dir.glob("*"):
        score_path = next(rep_it_path.glob("score_*.csv"))
        score_df = pd.read_csv(score_path)
        test_score = score_df["best_val_mae"]
        test_scores.setdefault(model_dir.name, []).append(float(test_score))

        metadata = joblib.load(rep_it_path / "metadata.pkl")
        histories.setdefault(model_dir.name, []).append(metadata.history)

{key: np.mean(values) * 90 for key, values in test_scores.items()}
