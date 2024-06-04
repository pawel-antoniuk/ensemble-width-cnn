import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotmap import DotMap
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import tensorflow as tf
import colormaps
import matplotlib.colors as mcolors

from model import Model

model_name = "multi_inout_1_model_final_3_nogcc"
model_dir = Path("/data/train/out_models") / model_name
out_img_format = "pdf"


def load_results():
    prediction_results = []
    for path_rep_it in model_dir.glob("*"):
        request = joblib.load(path_rep_it / "request.pkl")
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
        assert np.all(np.abs(score_file_test_location -
                      score_test_location) < 1e-6)

        test_filenames = prediction_result.filename
        actual_widths_test = [
            float(x.split("_")[-4].replace("width", "")) for x in test_filenames
        ]
        stored_actual_widths_test = np.array(
            model.store_split.y_test_width * 45.0)
        assert np.all(abs(stored_actual_widths_test -
                      actual_widths_test) < 1e-5)

        actual_recordings = [
            os.path.basename(x.split("_")[-7].replace("width", ""))
            for x in test_filenames
        ]
        assert np.all(model.store_split.actual_recordings_test ==
                      actual_recordings)

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


def get_results():
    results_filename = f"/data/analyze/state/analyze_results_{model_name}.pkl"
    try:
        return joblib.load(results_filename)
    except FileNotFoundError:
        prediction_results = load_results()
        joblib.dump(prediction_results, results_filename)
        return prediction_results


def load_sample_data():
    src_request_path = model_dir / "0" / "request.pkl"
    request = joblib.load(src_request_path)
    model = Model(request).try_load_or_compute_input_data()
    np.random.seed(1)
    n_samples = 100
    random_indices = np.random.choice(
        model.store_split.x_train_mag_shape[0], n_samples, replace=False
    )
    filename_indices_train = model.store_split.idx_dev[model.store_split.idx_train]
    return DotMap(
        {
            "data_mag": model.x_train_mag[random_indices, :, :, :],
            "data_phase": model.x_train_phase[random_indices, :, :, :],
            "data_gcc_phat": model.store_split.x_train_gcc_phat[random_indices, :],
            "filenames": model.store_extract.actual_filenames[
                filename_indices_train[random_indices]
            ],
            "axis_band": np.linspace(
                model.spectrogram_min_freq,
                model.spectrogram_max_freq,
                model.target_bands,
            ),
            "axis_time": np.linspace(0, 7, model.n_time_frames),
        }
    )


def get_sample_data():
    sample_data_filename = "analyze_sample_data.pkl"
    try:
        return joblib.load(sample_data_filename)
    except FileNotFoundError:
        samples = load_sample_data()
        joblib.dump(samples, sample_data_filename)
        return samples


def get_train_history():
    histories = []
    for rep_it_path in model_dir.glob("*"):
        metadata_path = rep_it_path / "metadata.pkl"
        metadata = joblib.load(metadata_path)
        histories.append(metadata.history)
    return histories


# %% Summary statistics
results = get_results()
mean_width = np.mean([np.mean(np.abs(result.errors_width))
                     for result in results])
std_width = np.std([np.mean(np.abs(result.errors_width))
                   for result in results])
print(f"Width MAE: {mean_width:0.2f} (std: {std_width:0.2f})")

mean_location = np.mean(
    [np.mean(np.abs(result.errors_location)) for result in results])
std_location = np.std([np.mean(np.abs(result.errors_location))
                      for result in results])
print(f"Location MAE: {mean_location:0.2f} (std: {std_location:0.2f})")

os.makedirs("figures", exist_ok=True)


# %% Plot input samples (spectrograms)
samples = get_sample_data()
plt.figure(figsize=(8, 6))
for i_sample in range(1, 5):

    def trim_string(s, n):
        return s[:n] + "..." if len(s) > n else s

    plt.subplot(2, 2, i_sample)
    plt.pcolormesh(
        samples.axis_time,
        samples.axis_band / 1000,
        samples.data_mag[i_sample, :, :, 0],
        rasterized=True,
    )
    plt.ylabel("Frequency [kHz]")
    plt.xlabel("Time [s]")
    plt.xticks(np.linspace(0, 7, 8))
    plt.yticks(np.linspace(0, 16, 9))
    title = os.path.basename(samples.filenames[i_sample])
    plt.gca().tick_params(direction="inout", which="both")
    plt.title("Recording: " + title.split('_')[0])
    plt.grid()
plt.tight_layout()
plt.savefig(f"figures/samples.{out_img_format}")
# plt.show()


# %% Plot train history
histories = get_train_history()
plt.figure(figsize=(6, 5))
for i_history, history in enumerate(histories[0:4]):
    # plt.subplot(2, 2, i_history + 1)
    plt.plot(np.array(history["loss"]) * 90)
    plt.plot(np.array(history["val_loss"]) * 90)
    plt.axvline(x=np.argmin(history["val_loss"]), color="r")
    plt.xlabel("Iteration")
    plt.ylabel("Total Mean Absolute Error")
    # plt.title(f'Iteration {i_history}')
    plt.grid()
    plt.xticks(range(0, 100, 5))
    plt.xlim([-2, 55])
    plt.ylim([12, 44])
    plt.legend(["Training Loss", "Validation Loss",
               "Optimal Validation Checkpoint"])
plt.tight_layout()
plt.savefig(f"figures/history.{out_img_format}")
# plt.show()


# %% Draw model
model = tf.keras.models.load_model( model_dir / "0" / "model.keras")
tf.keras.utils.plot_model(
    model,
    to_file=f"figures/model.{out_img_format}",
    show_shapes=True,
    show_layer_names=True,
)


# %% actual vs predicted width
plt.figure(figsize=(4, 3))
plt.scatter(
    results[0].actual_width,
    results[0].predicted_width,
    s=15,
    alpha=0.15,
    edgecolor="none",
    rasterized=True,
)
plt.xlabel("Actual Width")
plt.ylabel("Predicted Width")
plt.grid()
plt.xticks(np.linspace(0, 90, 7))
plt.yticks(np.linspace(0, 90, 7))
plt.xlim([0, 90])
plt.ylim([0, 90])
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x}°"))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x}°"))
plt.tight_layout()
plt.savefig(f"figures/actual_vs_predicted_width.{out_img_format}")
# plt.show()


# %% actual vs predicted location
plt.figure(figsize=(4, 3))
plt.scatter(
    results[0].actual_location,
    results[0].predicted_location,
    s=15,
    alpha=0.15,
    edgecolor="none",
    rasterized=True,
)
plt.xlabel("Actual Location")
plt.ylabel("Predicted Location")
plt.grid()
plt.xticks(np.linspace(-45, 45, 7))
plt.yticks(np.linspace(-45, 45, 7))
plt.xlim([-45, 45])
plt.ylim([-45, 45])
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x}°"))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x}°"))
plt.tight_layout()
plt.savefig(f"figures/actual_vs_predicted_location.{out_img_format}")
# plt.show()


# %% Mean absolute error vs actual width
actual_widths = np.linspace(0, 90, 100)
mae_for_actual_width = []
std_for_actual_width = []
win = 2

for actual_width in actual_widths:
    maes = []
    for result in results:
        widths = result.actual_width
        mask = (widths > actual_width - win /
                2) & (widths < actual_width + win / 2)
        mae = np.mean(np.abs(result.errors_width[mask]))
        maes.append(mae)
    mae = np.mean(maes)
    std = np.std(maes)
    mae_for_actual_width.append(mae)
    std_for_actual_width.append(std)

mae_for_actual_width = np.array(mae_for_actual_width)
std_for_actual_width = np.array(std_for_actual_width)

plt.figure(figsize=(4, 3))
plt.plot(actual_widths, mae_for_actual_width)
plt.fill_between(
    actual_widths,
    mae_for_actual_width - std_for_actual_width,
    mae_for_actual_width + std_for_actual_width,
    alpha=0.3,
    edgecolor="none",
)
plt.xlabel("Actual Width")
plt.ylabel("Mean Absolute Error")
plt.grid()
plt.xticks(np.linspace(0, 90, 7))
plt.yticks(np.linspace(0, 18, 7))
plt.xlim([0, 90])
plt.ylim([0, 18])
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x}°"))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x}°"))
plt.legend(["Mean Absolute Error", "Standard Deviation"])
plt.tight_layout()
plt.savefig(f"figures/mae_width.{out_img_format}")
# plt.show()


# %% Mean absolute error vs actual location
actual_locations = np.linspace(-45, 45, 100)
mae_for_actual_location = []
std_for_actual_location = []
win = 2

for actual_location in actual_locations:
    maes = []
    for result in results:
        locations = result.actual_location
        mask = (locations > actual_location - win / 2) & (
            locations < actual_location + win / 2
        )
        mae = np.mean(np.abs(result.errors_location[mask]))
        maes.append(mae)
    mae = np.mean(maes)
    std = np.std(maes)
    mae_for_actual_location.append(mae)
    std_for_actual_location.append(std)

mae_for_actual_location = np.array(mae_for_actual_location)
std_for_actual_location = np.array(std_for_actual_location)

plt.figure(figsize=(4, 3))
plt.plot(actual_locations, mae_for_actual_location)
plt.fill_between(
    actual_locations,
    mae_for_actual_location - std_for_actual_location,
    mae_for_actual_location + std_for_actual_location,
    alpha=0.3,
    edgecolor="none",
)
plt.xlabel("Actual Location")
plt.ylabel("Mean Absolute Error")
plt.grid()
plt.xticks(np.linspace(-45, 45, 7))
plt.yticks(np.linspace(0, 18, 7))
plt.xlim([-45, 45])
plt.ylim([0, 18])
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x}°"))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x}°"))
plt.legend(["Mean Absolute Error", "Standard Deviation"])
plt.tight_layout()
plt.savefig(f"figures/mae_location.{out_img_format}")
# plt.show()


# %% Calculate width vs location vs error
actual_widths = np.linspace(0, 90, 91)
actual_locations = np.linspace(-45, 45, 91)
mae_map_width = np.zeros((len(actual_widths), len(actual_locations)))
std_map_width = np.zeros((len(actual_widths), len(actual_locations)))
mae_map_location = np.zeros((len(actual_widths), len(actual_locations)))
std_map_location = np.zeros((len(actual_widths), len(actual_locations)))
win = 3

for i_actual_wdith, actual_width in enumerate(actual_widths):
    for i_actual_location, actual_location in enumerate(actual_locations):
        maes_width = []
        maes_location = []

        for result in results:
            locations = result.actual_location
            widths = result.actual_width
            mask = (locations > actual_location - win / 2) \
                & (locations < actual_location + win / 2) \
                & (widths > actual_width - win / 2) \
                & (widths < actual_width + win / 2)
            mae_location = np.mean(np.abs(result.errors_location[mask]))
            mae_width = np.mean(np.abs(result.errors_width[mask]))
            maes_location.append(mae_location)
            maes_location.append(mae_width)

        mae_map_location[i_actual_wdith,
                         i_actual_location] = np.mean(mae_location)
        std_map_location[i_actual_wdith,
                         i_actual_location] = np.std(mae_location)
        mae_map_width[i_actual_wdith, i_actual_location] = np.mean(mae_width)
        std_map_width[i_actual_wdith, i_actual_location] = np.std(mae_width)


# %% Plot width vs location vs error
# Location MAE
cm = colormaps.parula
norm = mcolors.Normalize(vmin=0, vmax=15)
plt.figure(figsize=(8, 4))
plt.imshow(mae_map_location, 
           cmap=cm,
           interpolation='nearest',
           aspect='auto',
           norm=norm)
plt.gca().invert_yaxis()

cbar = plt.colorbar()
cbar.set_label('Ensemble Location Error')
tick_labels = [f'{int(tick)}°' for tick in cbar.get_ticks()]
tick_labels[-1] = f'>{tick_labels[-1]}'
cbar.set_ticks(cbar.get_ticks())
cbar.set_ticklabels(tick_labels)

plt.xlabel('Ensemble Location')
plt.ylabel('Ensemble Width')
xticks = np.linspace(0, mae_map_location.shape[1] - 1, 7)
yticks = np.linspace(0, mae_map_location.shape[0] - 1, 4)
xlabels = [f'{int(label)}°' for label in np.linspace(-45, 45, 7)]
ylabels = [f'{int(label)}°' for label in np.linspace(0, 90, 4)]
plt.xticks(xticks, xlabels)
plt.yticks(yticks, ylabels)
plt.grid(color='black')
plt.xlim([0, 90])
plt.ylim([0, 90])
plt.savefig(f"figures/map_mae_location.{out_img_format}")

# Width MAE
norm = mcolors.Normalize(vmin=0, vmax=30)
plt.figure(figsize=(8, 4))
plt.imshow(mae_map_width, 
           cmap=cm,
           interpolation='nearest',
           aspect='auto',
           norm=norm)
plt.gca().invert_yaxis()

cbar = plt.colorbar()
cbar.set_label('Ensemble Width Error')
tick_labels = [f'{int(tick)}°' for tick in cbar.get_ticks()]
tick_labels[-1] = f'>{tick_labels[-1]}'
cbar.set_ticks(cbar.get_ticks())
cbar.set_ticklabels(tick_labels)

plt.xlabel('Ensemble Location')
plt.ylabel('Ensemble Width')
xticks = np.linspace(0, mae_map_width.shape[1] - 1, 7)
yticks = np.linspace(0, mae_map_width.shape[0] - 1, 4)
xlabels = [f'{int(label)}°' for label in np.linspace(-45, 45, 7)]
ylabels = [f'{int(label)}°' for label in np.linspace(0, 90, 4)]
plt.xticks(xticks, xlabels)
plt.yticks(yticks, ylabels)
plt.grid(color='black')
plt.xlim([0, 90])
plt.ylim([0, 90])
plt.savefig(f"figures/map_mae_width.{out_img_format}")

# %%
