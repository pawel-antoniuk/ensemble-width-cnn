import joblib
import os
from pathlib import Path
from model import Model
import numpy as np
import pandas as pd
from dotmap import DotMap


def load_results(model_dir: Path):
    prediction_results = []
    for path_rep_it in model_dir.glob("*"):
        request = joblib.load(path_rep_it / "request.pkl")
        print(f"loading model[{path_rep_it}] for request {request}")
        model = Model(request).try_load_or_compute_input_data()
        prediction_result = model.predict_test_data()
        errors_width = (
            prediction_result.predicted_width - prediction_result.actual_width
        )
        errors_location = (
            prediction_result.predicted_location - prediction_result.actual_location
        )
        score_test_width = np.mean(np.abs(errors_width))
        score_test_location = np.mean(np.abs(errors_location))

        # assertions
        score_path = next(path_rep_it.glob("score_*.csv"))
        score_df = pd.read_csv(score_path)
        score_file_test_width = score_df["test_score_width"]
        score_file_test_location = score_df["test_score_location"]
        assert np.all(np.abs(score_file_test_width - score_test_width) < 1e-6)
        assert np.all(np.abs(score_file_test_location - score_test_location) < 1e-6)

        test_filenames = prediction_result.filename
        actual_widths_test = [
            float(x.split("_")[-4].replace("width", "")) for x in test_filenames
        ]
        stored_actual_widths_test = np.array(model.store_split.y_test_width * 45.0)
        assert np.all(abs(stored_actual_widths_test - actual_widths_test) < 1e-5)

        actual_recordings = [
            os.path.basename(x.split("_")[-7].replace("width", ""))
            for x in test_filenames
        ]
        assert np.all(model.store_split.actual_recordings_test == actual_recordings)

        prediction_results.append(
            DotMap(
                {
                    "filename": prediction_result.filename,
                    "actual_width": prediction_result.actual_width * 90,
                    "predicted_width": prediction_result.predicted_width * 90,
                    "errors_width": errors_width * 90,
                    "actual_location": prediction_result.actual_location * 45,
                    "predicted_location": prediction_result.predicted_location * 45,
                    "errors_location": errors_location * 45,
                }
            )
        )

    return prediction_results


def get_results(model_name: str):
    results_filename = f"/app/data/analyze/state/analyze_results_{model_name}.pkl"
    try:
        return joblib.load(results_filename)
    except FileNotFoundError:
        model_dir = Path("/app/data/train/out_models") / model_name
        prediction_results = load_results(model_dir)
        joblib.dump(prediction_results, results_filename)
        return prediction_results
