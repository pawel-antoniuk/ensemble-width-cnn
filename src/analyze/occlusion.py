import os
from pathlib import Path
import joblib

import numpy as np
from dotmap import DotMap
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import tensorflow as tf
import colormaps
import matplotlib.colors as mcolors
from model import Model
from results import get_results

model_name = "multi_inout_1_model_final_3_nogcc_nosame"
model_dir = Path("/app/data/train/out_models") / model_name
src_request_path = model_dir / "0" / "request.pkl"
request = joblib.load(src_request_path)

model = Model(request).try_load_or_compute_input_data()

for i_sample in range(3):
    X = model.x_test_mag[i_sample:i_sample+1, :, :, :]
    y_actual_location = model.store_split.y_test_location[i_sample]
    y_actual_width = model.store_split.y_test_width[i_sample]
    y_pred = model.predict(X)
    err_location = abs(y_actual_location - y_pred['predicted_location']) * 45
    err_width = abs(y_actual_width - y_pred['predicted_width']) * 90
    print(f'Score: Location: {err_location}, Width: {err_width}')
