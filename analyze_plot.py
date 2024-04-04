import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotmap import DotMap
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import tensorflow as tf

from model import Model


def load_results():
    prediction_results = []
    src_model_path = Path('models') / 'simple2'
    for path_rep_it in src_model_path.glob('*'):
        request = joblib.load(path_rep_it / 'request.pkl')
        model = Model(request).try_load_or_compute_input_data()
        prediction_result = model.predict_test_data()
        errors = prediction_result.predicted_width - prediction_result.actual_width
        score_test = np.mean(np.abs(errors))

        # assertions
        score_path = next(path_rep_it.glob('score_*.csv'))
        score_df = pd.read_csv(score_path)
        score_file_test = score_df['test_score']
        assert np.all(np.abs(score_file_test - score_test) < 1e-6)

        test_filenames = prediction_result.filename
        actual_widths_test = [float(x.split('_')[-4].replace('width', '')) for x in test_filenames]
        stored_actual_widths_test = np.array(model.store_split.y_test * 45.0)
        assert np.all(abs(stored_actual_widths_test - actual_widths_test) < 1e-6)

        actual_recordings = [os.path.basename(x.split('_')[-7].replace('width', '')) for x in test_filenames]
        assert np.all(model.store_split.actual_recordings_test == actual_recordings)

        prediction_results.append(DotMap({
            'filename': prediction_result.filename,
            'actual_width': prediction_result.actual_width * 90,
            'predicted_width': prediction_result.predicted_width * 90,
            'errors': errors * 90
        }))

    return prediction_results


def get_results():
    results_filename = 'analyze_results.pkl'
    try:
        return joblib.load(results_filename)
    except FileNotFoundError:
        prediction_results = load_results()
        joblib.dump(results, results_filename)
        return prediction_results


def load_sample_data():
    src_request_path = Path('models') / 'simple2' / '0' / 'request.pkl'
    request = joblib.load(src_request_path)
    model = Model(request).try_load_or_compute_input_data()
    np.random.seed(1)
    n_samples = 100
    random_indices = np.random.choice(model.store_split.x_train.shape[0], n_samples, replace=False)
    filename_indices_train = model.store_split.idx_dev[model.store_split.idx_train]
    return DotMap({
        'data': model.store_split.x_train[random_indices, :, :, :],
        'filenames': model.store_extract.actual_filenames[filename_indices_train[random_indices]],
        'axis_band': np.linspace(model.spectrogram_min_freq, model.spectrogram_max_freq, model.target_bands),
        'axis_time': np.linspace(0, 7, model.n_time_frames)
    })


def get_sample_data():
    sample_data_filename = 'analyze_sample_data.pkl'
    try:
        return joblib.load(sample_data_filename)
    except FileNotFoundError:
        samples = load_sample_data()
        joblib.dump(samples, sample_data_filename)
        return samples


def get_train_history():
    histories = []
    src_model_path = Path('models') / 'simple2'
    for rep_it_path in src_model_path.glob('*'):
        metadata_path = rep_it_path / 'metadata.pkl'
        metadata = joblib.load(metadata_path)
        histories.append(metadata.history)
    return histories


#%% Summary statistics
results = get_results()
mean = np.mean([np.mean(np.abs(result.errors)) for result in results])
std = np.std([np.mean(np.abs(result.errors)) for result in results])
print(f'MAE: {mean:0.2f} (std: {std:0.2f})')

os.makedirs('figures', exist_ok=True)

#%% Plot input samples
samples = get_sample_data()
plt.figure(figsize=(12, 8))
for i_sample in range(1, 5):
    def trim_string(s, n):
        return s[:n] + '...' if len(s) > n else s


    plt.subplot(2, 2, i_sample)
    plt.pcolormesh(samples.axis_time,
                   samples.axis_band / 1000,
                   samples.data[i_sample, :, :, 0])
    plt.ylabel('Frequency [kHz]')
    plt.xlabel('Time [s]')
    plt.xticks(np.linspace(0, 7, 8))
    plt.yticks(np.linspace(0, 16, 9))
    title = os.path.basename(samples.filenames[i_sample])
    plt.gca().tick_params(direction='inout', which='both')
    plt.title('Sample: ' + trim_string(title, 50))
    plt.grid()
plt.tight_layout()
plt.savefig('figures/samples.png')
plt.show()

#%% Plot train history
histories = get_train_history()
plt.figure(figsize=(12, 10))
for i_history, history in enumerate(histories[0:4]):
    plt.subplot(2, 2, i_history + 1)
    plt.plot(np.array(history['loss']) * 90)
    plt.plot(np.array(history['val_loss']) * 90)
    plt.axvline(x=np.argmin(history['val_loss']), color='r')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Absolute Error')
    plt.title(f'Iteration {i_history}')
    plt.grid()
    plt.xticks(range(0, 100, 5))
    plt.xlim([-2, 55])
    plt.ylim([6, 22])
    plt.legend(['Training Loss', 'Validation Loss', 'Optimal Validation Checkpoint'])
plt.tight_layout()
plt.savefig('figures/history.png')
plt.show()

#%% Plot model
model_path = Path('models') / 'simple2' / '0' / 'model.h5'
model = tf.keras.models.load_model(model_path)
tf.keras.utils.plot_model(model,
                          to_file='figures/model.png',
                          show_shapes=True,
                          show_layer_names=True)

#%% Scatter plot
plt.figure(figsize=(6, 5))
plt.scatter(results[0].actual_width, results[0].predicted_width, s=20, alpha=0.25, edgecolor='none')
plt.xlabel('Actual Width')
plt.ylabel('Predicted Width')
plt.grid()
plt.xticks(np.linspace(0, 90, 7))
plt.yticks(np.linspace(0, 90, 7))
plt.xlim([0, 90])
plt.ylim([0, 90])
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x}°'))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x}°'))
plt.tight_layout()
plt.savefig('figures/scatter.png')
plt.show()

actual_widths = np.linspace(0, 90, 100)
mae_for_actual_width = []
std_for_actual_width = []
win = 2

for actual_width in actual_widths:
    maes = []
    for result in results:
        widths = result.actual_width
        mask = ((widths > actual_width - win / 2)
                & (widths < actual_width + win / 2))
        mae = np.mean(np.abs(result.errors[mask]))
        maes.append(mae)
    mae = np.mean(maes)
    std = np.std(maes)
    mae_for_actual_width.append(mae)
    std_for_actual_width.append(std)

mae_for_actual_width = np.array(mae_for_actual_width)
std_for_actual_width = np.array(std_for_actual_width)

#%% Mean absolute error
plt.figure(figsize=(6, 5))
plt.plot(actual_widths, mae_for_actual_width)
plt.fill_between(actual_widths,
                 mae_for_actual_width - std_for_actual_width,
                 mae_for_actual_width + std_for_actual_width,
                 alpha=0.3,
                 edgecolor='none')
plt.xlabel('Actual Width')
plt.ylabel('Mean Absolute Error')
plt.grid()
plt.xticks(np.linspace(0, 90, 7))
plt.yticks(np.linspace(0, 18, 7))
plt.xlim([0, 90])
plt.ylim([0, 18])
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x}°'))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x}°'))
plt.legend(['Mean Absolute Error', 'Standard Deviation'])
plt.tight_layout()
plt.savefig('figures/mae.png')
plt.show()
