import os
from pathlib import Path
import joblib

# import sys

# sys.path.append('"/app/src/train')

import numpy as np
from dotmap import DotMap
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import tensorflow as tf
import colormaps
import matplotlib.colors as mcolors
from model import Model
from results import get_results
import soundfile as sf


model_name = "final_width_location"
model_dir = Path("/app/data/train/out_models") / model_name
src_request_path = model_dir / "1" / "request.pkl"
request = joblib.load(src_request_path)
data_dir = Path("/app/data/spatialize/spatresults/spat")
model = Model(request).try_load_or_compute_input_data()
errors_width = []
errors_location = []

spectrogram_per_actual_width_l = {}
spectrogram_per_actual_width_r = {}

for sample_path in list(data_dir.glob("*.wav")):
    print(f"[test] sample: {sample_path}")
    metadata = model.extract_metadata_from_sample_name(sample_path)
    audio_data, sample_rate = sf.read(sample_path)
    features = model.extract_features_from_raw_signal(audio_data, sample_rate)
    pred = model.predict_by_features(features)
    error_width = abs(pred.predicted_width[0] - metadata.actual_width * 2)
    error_location = abs(pred.predicted_location[0] - metadata.actual_location)
    errors_width.append(error_width)
    errors_location.append(error_location)
    actual_width = int(metadata.actual_width) // 5 * 5
    spectrogram_per_actual_width_l.setdefault(actual_width, []).append(
        features["in_mag"][0, :, :, 0]
    )
    spectrogram_per_actual_width_r.setdefault(actual_width, []).append(
        features["in_mag"][0, :, :, 1]
    )

    print(
        f"[test] actual_width: {metadata.actual_width * 2}, \
          actual_location: {metadata.actual_location}"
    )
    print(
        f"[test] predicted_width: {pred.predicted_width[0]}, \
          predicted_location: {pred.predicted_location[0]}"
    )
    print(
        f"[test] error_width: {error_width}, \
          error_location: {error_location}"
    )
    print(
        f"[test] cumulative_mae_width: {np.mean(errors_width)}, \
          cumulative_mae_location: {np.mean(errors_location)}"
    )
    print()

# %% Figures

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))

widths = [0, 15, 30, 45]
nrows = len(widths)
ncols = 4

ref_left = np.mean(spectrogram_per_actual_width_l[0], axis=0)
ref_right = np.mean(spectrogram_per_actual_width_r[0], axis=0)
ref_diff = ref_left - ref_right

pltargs = {
    "cmap": colormaps.parula,
    "aspect": "auto",
    "origin": "lower",
    "interpolation": "none",
    "extent": [0, 7, 0, 16],
}
def labels():
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.xticks(np.arange(0, 8))
    plt.yticks(np.arange(0, 17, 4))
    plt.grid() 

for i, actual_width in enumerate(widths):
    left = np.abs(
        np.mean(spectrogram_per_actual_width_l[actual_width], axis=0) - ref_left
    )
    right = np.abs(
        np.mean(spectrogram_per_actual_width_r[actual_width], axis=0) - ref_right
    )
    diff = np.abs(np.abs(left - right) - ref_diff)

    plt.subplot(nrows, ncols, i * ncols + 1, polar=True)
    plt.text(np.pi / 4, 1.1, f'θ = 0°\nω = {actual_width * 2}°')
    actual_location_rad = 0 / 180 * np.pi
    actual_width_rad = actual_width / 180 * np.pi
    theta = np.linspace(
        actual_location_rad - actual_width_rad, actual_location_rad + actual_width_rad
    )
    
    r = np.ones(theta.shape)
    plt.plot(theta, r, "r")
    plt.gca().set_thetamin(-90)
    plt.gca().set_thetamax(90)
    plt.gca().set_theta_direction(1)
    plt.gca().set_yticklabels([])

    plt.subplot(nrows, ncols, i * ncols + 2)
    plt.imshow(left, vmin=0, vmax=0.4, **pltargs)
    labels()

    plt.subplot(nrows, ncols, i * ncols + 3)
    plt.imshow(right, vmin=0, vmax=0.4, **pltargs)
    labels()

    plt.subplot(nrows, ncols, i * ncols + 4)
    plt.imshow(diff, vmin=0, vmax=0.3, **pltargs)
    labels()
    # plt.subplot(nrows, ncols, i * ncols + 4)
    # plt.colorbar()

plt.tight_layout()
plt.show()
plt.savefig(f"/app/data/analyze/figure/samples_width.png")
