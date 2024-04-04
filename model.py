from __future__ import annotations

import hashlib
import importlib
import os
import random
import sys
import timeit
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks
import yaml
from dotmap import DotMap
from scipy.signal import spectrogram
from scipy.signal.windows import hamming
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class Request:
    name: str = 'unknown_model'
    input_dir: str = 'input'
    num_samples: int = 23040
    target_bands: int = 300
    time_window_len: float = 0.04
    time_window_overlap: float = 0.5
    spectrogram_min_freq: int = 100
    spectrogram_max_freq: int = 16000
    recording_time_length: int = 7
    model_architecture: str = 'default'
    train_batch_size: int = 8
    train_epochs: int = 128
    train_patience: int = 15
    train_loss: str = 'mean_squared_error'
    train_optimizer: str = 'adam'
    dev_test_ratio: float = 1 / 3
    train_val_ratio: float = 1 / 8
    split_seed: int = 2
    train_seed: int = 3
    repetition_iteration: int = 1
    spectrogram_type: str = 'MAGNITUDE'


class Model:
    def __init__(self, request: Request):
        self.result_best_val_mae: Optional[float] = None
        self.result_best_epoch: Optional[int] = None
        self.result_test_mae: Optional[float] = None
        self.model_trainable_params: Optional[int] = None
        self.model_non_trainable_params: Optional[int] = None
        self.result_history: Optional[tensorflow.keras.callbacks.History] = None
        self.time_extract: Optional[float] = None
        self.time_split: Optional[float] = None
        self.time_train: Optional[float] = None

        self.store_split: DotMap = DotMap()
        self.store_extract: DotMap = DotMap()
        self.request: Request = request
        self.name = request.name
        self.input_dir = Path(request.input_dir)
        self.num_samples = request.num_samples
        self.filepaths = [p for p in self.input_dir.iterdir()][0: self.num_samples]
        self.target_bands = request.target_bands
        self.time_window_len = request.time_window_len
        self.time_window_overlap = request.time_window_overlap
        self.spectrogram_min_freq = request.spectrogram_min_freq
        self.spectrogram_max_freq = request.spectrogram_max_freq
        self.recording_time_length = request.recording_time_length
        self.n_time_frames = int(
            self.recording_time_length / self.time_window_len * 2 - 1
        )
        self.model_architecture = request.model_architecture
        self.train_batch_size = request.train_batch_size
        self.train_epochs = request.train_epochs
        self.train_patience = request.train_patience
        self.train_loss = request.train_loss
        self.train_optimizer = request.train_optimizer
        self.store_extract.number_of_spectrograms = 4 if request.spectrogram_type == 'MAGNITUDE_PHASE' else 2
        self.store_extract.input_features_shape = (
            self.num_samples,
            self.target_bands,
            self.n_time_frames,
            self.store_extract.number_of_spectrograms,
        )
        self.dev_test_ratio = request.dev_test_ratio
        self.train_val_ratio = request.train_val_ratio
        self.split_seed = request.split_seed
        self.train_seed = request.train_seed
        self.repetition_iteration = request.repetition_iteration
        self.spectrogram_type = request.spectrogram_type

        # self.log_buffer = io.StringIO()

        self.__print_properties_trimmed()

        # Configure GPU VRAM
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                config = [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=20 * 1024
                    )
                ]
                tf.config.experimental.set_virtual_device_configuration(gpus[0], config)
            except RuntimeError as e:
                print(e)

    def extract(self) -> Model:
        start_time = timeit.default_timer()

        print('[extract] Initializing memory')
        self.store_extract.input_features = np.memmap(
            '/run/media/pawel/alpha/input_features',
            dtype='float32',
            mode='w+',
            shape=self.store_extract.input_features_shape,
        )
        self.store_extract.actual_widths = np.zeros(self.num_samples)
        self.store_extract.actual_locations = np.zeros(self.num_samples)
        self.store_extract.actual_recordings = ['' for _ in range(self.num_samples)]
        self.store_extract.actual_filenames = ['' for _ in range(self.num_samples)]

        print('[extract] Memory initialized')

        with ThreadPoolExecutor(max_workers=2) as executor:
            try:
                futures = [
                    executor.submit(self.__load_spectrogram, filepath)
                    for filepath in self.filepaths
                ]

                last_results_elapsed = []

                for i, future in enumerate(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise e

                    self.store_extract.input_features[i] = result.spectrogram
                    self.store_extract.actual_widths[i] = result.actual_width
                    self.store_extract.actual_locations[i] = result.actual_location
                    self.store_extract.actual_recordings[i] = result.actual_recording
                    self.store_extract.actual_filenames[i] = result.actual_filename
                    last_results_elapsed.append(result.elapsed)

                    # if i % 10 == 0:
                    mean_elapsed = sum(last_results_elapsed) / len(last_results_elapsed)
                    print(f'[extract] [{i}] Average spectrogram extraction time per sample: {mean_elapsed:.3f} s')
                    last_results_elapsed.clear()

            except KeyboardInterrupt as e:
                executor.shutdown(wait=False, cancel_futures=True)
                raise e

        self.store_extract.actual_recordings = np.array(self.store_extract.actual_recordings)
        self.store_extract.actual_filenames = np.array(self.store_extract.actual_filenames)

        elapsed = timeit.default_timer() - start_time
        print(f'[extract] Loaded all spectrograms in {elapsed:.2f} s')
        self.time_extract = elapsed

        return self

    def try_load_or_compute_input_data(self) -> Model:
        try:
            print('[load] Attempting to load the cached input features')
            self.load_extracted()
        except FileNotFoundError:
            print('[load] Cache does not contain the requested data. Extracting data from scratch')
            self.extract()
            self.save_extracted()
        try:
            print('[load] Attempting to load the cached split data')
            self.load_split()
        except FileNotFoundError:
            print('[load] Cache does not contain the requested split data. Splitting data from scratch')
            self.split()
            self.save_split()
        return self

    def load_split(self) -> Model:
        start_time = timeit.default_timer()
        data_hash = self.__get_split_hash()
        print(f'[load] Split hash: {data_hash}')
        self.store_split = joblib.load(f'state/{data_hash}_store_split.pkl')
        elapsed = timeit.default_timer() - start_time
        print(f'[load] Loaded cached splits in {elapsed:.2f} s')
        return self

    def save_split(self) -> Model:
        start_time = timeit.default_timer()
        data_hash = self.__get_split_hash()
        print(f'[save] Split hash: {data_hash}')
        os.makedirs(f'state', exist_ok=True)
        joblib.dump(self.store_split, f'state/{data_hash}_store_split.pkl')
        elapsed = timeit.default_timer() - start_time
        print(f'[save] Saved cached splits in {elapsed:.2f} s')
        return self

    def load_extracted(self) -> Model:
        start_time = timeit.default_timer()
        data_hash = self.__get_extract_hash()
        print(f'[load] Extraction hash: {data_hash}')
        self.store_extract = joblib.load(f'state/{data_hash}_store_extract.pkl')
        elapsed = timeit.default_timer() - start_time
        print(f'[load] Loaded cached spectrograms in {elapsed:.2f} s')
        return self

    def save_extracted(self) -> Model:
        start_time = timeit.default_timer()
        data_hash = self.__get_extract_hash()
        print(f'[save] Extraction hash: {data_hash}')
        os.makedirs(f'state/', exist_ok=True)
        joblib.dump(self.store_extract, f'state/{data_hash}_store_extract.pkl')
        elapsed = timeit.default_timer() - start_time
        print(f'[save] Saved spectrograms in {elapsed:.2f} s')
        return self

    def plot_features(self, sample=50) -> Model:
        plt.figure(figsize=(20, 5))
        n_spectrograms = self.store_split.x_train.shape[-1]

        for i_spectrogram in range(n_spectrograms):
            plt.subplot(2, n_spectrograms // 2, i_spectrogram)
            data = self.store_split.x_train[sample, :, :, i_spectrogram]
            plt.pcolormesh(data, shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
            # plt.title('Left Channel Spectrogram')

        plt.rcParams.update({'font.size': 20})
        plt.tight_layout()
        plt.show()
        plt.savefig('figures/features.png')

        return self

    # Split data into training, validation and testing
    def split(self) -> Model:
        start_time_all = timeit.default_timer()
        print('[split] [1/3] Splitting data into development and testing set')
        start_time = timeit.default_timer()
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=self.dev_test_ratio,
            random_state=self.split_seed
        )
        idx_dev, idx_test = next(
            gss.split(
                self.store_extract.input_features, 
                self.store_extract.actual_widths,
                groups=self.store_extract.actual_recordings
            )
        )
        x_dev, x_test = self.store_extract.input_features[idx_dev], self.store_extract.input_features[idx_test]
        y_dev, y_test = self.store_extract.actual_widths[idx_dev], self.store_extract.actual_widths[idx_test]
        self.store_split.actual_recordings_dev = self.store_extract.actual_recordings[idx_dev]
        self.store_split.actual_recordings_test = self.store_extract.actual_recordings[idx_test]
        self.store_split.idx_dev = idx_dev
        self.store_split.idx_test = idx_test
        elapsed = timeit.default_timer() - start_time
        print(f'[split] [1/3] Done ({elapsed:.2f} s)')

        print('[split] [2/3] Splitting development set into training and validation set')
        start_time = timeit.default_timer()
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=self.train_val_ratio,
            random_state=self.split_seed + 1
        )
        idx_train, idx_val = next(gss.split(x_dev, y_dev, groups=self.store_split.actual_recordings_dev))
        x_train, x_val = x_dev[idx_train], x_dev[idx_val]
        y_train, y_val = y_dev[idx_train], y_dev[idx_val]
        self.store_split.actual_recordings_train = self.store_extract.actual_recordings[idx_train]
        self.store_split.actual_recordings_val = self.store_extract.actual_recordings[idx_val]
        self.store_split.idx_train = idx_train
        self.store_split.idx_val = idx_val
        elapsed = timeit.default_timer() - start_time
        print(f'[split] [2/3] Done ({elapsed:.2f} s)')

        del self.store_extract.input_features
        del self.store_extract.actual_widths
        del self.store_extract.actual_locations
        del self.store_extract.actual_recordings

        print('[split] [3/3] Normalizing data')
        start_time = timeit.default_timer()
        x_train_mean = np.mean(x_train)
        x_train_std = np.std(x_train)
        x_train = (x_train - x_train_mean) / x_train_std
        x_val = (x_val - x_train_mean) / x_train_std
        x_test = (x_test - x_train_mean) / x_train_std
        y_train = y_train / 45
        y_val = y_val / 45
        y_test = y_test / 45
        elapsed = timeit.default_timer() - start_time
        print(f'[split] [3/3] Done ({elapsed:.2f} s)')

        self.store_split.x_train, self.store_split.y_train = x_train, y_train
        self.store_split.x_val, self.store_split.y_val = x_val, y_val
        self.store_split.x_test, self.store_split.y_test = x_test, y_test

        print(f'[split] x_train shape: {x_train.shape}, y_train shape {y_train.shape}')
        print(f'[split] x_val shape: {x_val.shape}, y_val shape {y_val.shape}')
        print(f'[split] x_test shape: {x_test.shape}, y_test shape {y_test.shape}')

        self.time_split = timeit.default_timer() - start_time_all

        return self

    # Train
    def train(self) -> Model:
        start_time = timeit.default_timer()

        tf.random.set_seed(self.train_seed)
        np.random.seed(self.train_seed)
        random.seed(self.train_seed)

        dest_model_path = self.__get_model_output_dir() / 'model.h5'
        os.makedirs(os.path.dirname(dest_model_path), exist_ok=True)

        model = self.__load_topology()
        model.summary()

        self.__save_model_params(model)

        early_stop = tensorflow.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.train_patience,
            restore_best_weights=False
        )
        checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(
            dest_model_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1,
        )
        model.compile(
            loss=self.train_loss,
            optimizer=self.train_optimizer
        )
        self.result_history = model.fit(
            self.store_split.x_train,
            self.store_split.y_train,
            batch_size=self.train_batch_size,
            epochs=self.train_epochs,
            validation_data=(self.store_split.x_val, self.store_split.y_val),
            callbacks=[early_stop, checkpoint],
        )

        self.result_best_val_mae = min(self.result_history.history['val_loss'])
        self.result_best_epoch = (
                self.result_history.history['val_loss'].index(self.result_best_val_mae) + 1
        )

        print('[train] The training procedure ended with validation loss:', self.result_best_val_mae)

        model.load_weights(dest_model_path)

        self.result_test_mae = model.evaluate(self.store_split.x_test, self.store_split.y_test, verbose=0)

        print('[train] Test loss:', self.result_test_mae)

        self.time_train = timeit.default_timer() - start_time

        return self

    # Plot Training History

    def plot_history(self) -> Model:
        loss = self.result_history.history['loss']
        val_loss = self.result_history.history['val_loss']
        # mae = self.result_history.history['mae']
        # val_mae = self.result_history.history['val_mae']
        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(12, 5))

        # Plotting training and validation loss
        # plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting training and validation MAE
        # plt.subplot(1, 2, 2)
        # plt.plot(epochs, mae, 'ro', label='Training MAE')
        # plt.plot(epochs, val_mae, 'r', label='Validation MAE')
        # plt.title('Training and Validation MAE')
        # plt.xlabel('Epochs')
        # plt.ylabel('MAE')
        # plt.legend()

        plt.tight_layout()
        plt.show()

        return self

    def save_results(self) -> Model:
        dest_dir = self.__get_model_output_dir()
        os.makedirs(dest_dir, exist_ok=True)

        # save request
        joblib.dump(self.request, dest_dir / 'request.pkl')
        with open(dest_dir / 'request.yaml', 'w') as f:
            yaml.dump(asdict(self.request), f)

        # save source
        current_script_path = os.path.abspath(__file__)
        with open(current_script_path, 'r') as current_script_source:
            with open(dest_dir / 'source.pyarch', 'w') as dest_script_source:
                dest_script_source.write(current_script_source.read())

        # save splits
        pd.DataFrame({
            'recordings': np.unique(self.store_split.actual_recordings_dev)
        }).to_csv(dest_dir / 'recordings_dev.csv')
        pd.DataFrame({
            'recordings': np.unique(self.store_split.actual_recordings_train)
        }).to_csv(dest_dir / 'recordings_train.csv')
        pd.DataFrame({
            'recordings': np.unique(self.store_split.actual_recordings_val)
        }).to_csv(dest_dir / 'recordings_val.csv')
        pd.DataFrame({
            'recordings': np.unique(self.store_split.actual_recordings_test)
        }).to_csv(dest_dir / 'recordings_test.csv')

        # save architecture source
        architecture_script_path = (
                Path('architecture') / f'{self.model_architecture}.py'
        )
        with open(architecture_script_path, 'r') as architecture_script_source:
            with open(dest_dir / 'architecture.pyarch', 'w') as dest_script_source:
                dest_script_source.write(architecture_script_source.read())

        # save metadata
        metadata = DotMap()
        metadata.history = self.result_history.history
        joblib.dump(metadata, dest_dir / 'metadata.pkl')

        # delete old score if exists and save actual scores
        for score_filename in dest_dir.glob('score_*.csv'):
            score_filename.unlink()
        pd.DataFrame(
            {
                'best_val_mae': [self.result_best_val_mae],
                'test_score': [self.result_test_mae],
                'best_epoch': [self.result_best_epoch],
                'params': [self.model_trainable_params],
            }
        ).to_csv(dest_dir / f'score_{self.result_best_val_mae:0.4f}.csv')

        # save properties
        max_value_len = 100
        properties = []
        values = []
        for prop, value in vars(self).items():
            value = str(value)
            if len(value) > max_value_len:
                value = value[:max_value_len] + ' ...'
            properties.append(prop)
            values.append(value)
        pd.DataFrame({'property': properties, 'value': values}).to_csv(
            dest_dir / 'properties.csv'
        )

        return self

    def predict_test_data(self):
        output_dir = self.__get_model_output_dir()
        model_path = output_dir / 'model.h5'
        model = tensorflow.keras.models.load_model(model_path)
        predictions = model.predict(self.store_split.x_test)
        test_filenames = self.store_extract.actual_filenames[self.store_split.idx_test]

        return pd.DataFrame({
            'filename': test_filenames,
            'actual_width': self.store_split.y_test,
            'predicted_width': predictions[:, 0]
        })

    def __get_extract_hash(self) -> str:
        args = [self.input_dir,
                self.target_bands,
                self.time_window_len,
                self.time_window_overlap,
                self.spectrogram_type]
        combined_string = '|'.join(str(arg) for arg in args)
        encoded_string = combined_string.encode('utf-8')
        hash_obj = hashlib.sha256(encoded_string)
        return hash_obj.hexdigest()[:8]

    def __get_split_hash(self) -> str:
        args = [self.__get_extract_hash(),
                self.split_seed]
        combined_string = '|'.join(str(arg) for arg in args)
        encoded_string = combined_string.encode('utf-8')
        hash_obj = hashlib.sha256(encoded_string)
        return hash_obj.hexdigest()[:8]

    def __save_model_params(self, model):
        self.model_trainable_params = np.sum(
            [K.count_params(w) for w in model.trainable_weights]
        )
        self.model_non_trainable_params = np.sum(
            [K.count_params(w) for w in model.non_trainable_weights]
        )

    def __get_model_output_dir(self) -> Path:
        return Path('models') / self.name / str(self.repetition_iteration)

    def __print_properties_trimmed(self) -> None:
        n = 100
        for prop, value in vars(self).items():
            if value is None:
                continue
            value = str(value)
            if isinstance(value, str) and len(value) > n:
                value = value[:n] + ' ...'
            print(f'\t{prop}: {value}')

    def __aggregate_spectrogram_to_n_bands(self, frequencies: np.ndarray, sxx: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray]:
        mask = (frequencies >= self.spectrogram_min_freq) & (frequencies <= self.spectrogram_max_freq)
        valid_freq_indices = np.where(mask)[0]
        frequencies = frequencies[valid_freq_indices]
        sxx = sxx[valid_freq_indices, :]
        total_freq_range = frequencies[-1] - frequencies[0]
        band_freq_range = total_freq_range / self.target_bands
        aggregated_sxx = np.zeros((self.target_bands, sxx.shape[1]), dtype=np.complex128)
        new_freq = np.linspace(self.spectrogram_min_freq,
                               self.spectrogram_max_freq,
                               num=self.target_bands)

        for i in range(self.target_bands):
            band_start_freq = self.spectrogram_min_freq + i * band_freq_range
            band_end_freq = band_start_freq + band_freq_range
            mask = (frequencies >= band_start_freq) & (frequencies < band_end_freq)
            freq_indices = np.where(mask)[0]

            if len(freq_indices) > 0:
                aggregated_sxx[i, :] = np.mean(sxx[freq_indices, :], axis=0)

        return new_freq, aggregated_sxx

    def __load_spectrogram(self, filepath: Path) -> DotMap:
        if filepath.is_file():
            start_time = timeit.default_timer()
            actual_width = float(str(filepath).split('_')[-4].replace('width', ''))
            actual_location = float(
                str(filepath).split('_')[-2].replace('azoffset', '')
            )
            actual_recording = str(filepath).split('_')[0].split(os.sep)[-1]

            audio_data, sample_rate = sf.read(filepath)
            _, _, features = self.__compute_spectrogram(audio_data, sample_rate)
            elapsed = timeit.default_timer() - start_time

            result = DotMap()
            result.spectrogram = features
            result.actual_width = actual_width
            result.actual_location = actual_location
            result.actual_recording = actual_recording
            result.actual_filename = str(filepath)
            result.elapsed = elapsed

            return result

    def __compute_spectrogram(
            self,
            audio_data: np.ndarray,
            sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nperseg = int(self.time_window_len * sample_rate)
        noverlap = int(nperseg * self.time_window_overlap)

        def __spectrogram(channel):
            freq, time, zxx = spectrogram(
                channel,
                fs=sample_rate,
                window=hamming(nperseg),
                noverlap=noverlap,
                mode='complex'
            )

            freq, zxx = self.__aggregate_spectrogram_to_n_bands(freq, zxx)

            if self.spectrogram_type == 'MAGNITUDE':
                spectrogram_data = 10 * np.log10(np.abs(zxx))
            elif self.spectrogram_type == 'PHASE':
                spectrogram_data = np.unwrap(np.angle(zxx), axis=1)
            elif self.spectrogram_type == 'MAGNITUDE_PHASE':
                spectrogram_data = np.stack([
                    10 * np.log10(np.abs(zxx)),
                    np.unwrap(np.angle(zxx), axis=1)
                ], axis=-1)
            else:
                raise ValueError(f'Unknown spectrogram type: {self.spectrogram_type}')

            if spectrogram_data.ndim == 2:
                spectrogram_data = np.expand_dims(spectrogram_data, axis=2)

            return freq, time, spectrogram_data

        freq_left, times_left, spectrogram_left = __spectrogram(audio_data[:, 0])
        _, _, spectrogram_right = __spectrogram(audio_data[:, 1])
        spectrogram_stack = np.concatenate([spectrogram_left, spectrogram_right], axis=2)

        return freq_left, times_left, spectrogram_stack

    def __load_topology(self) -> tensorflow.keras.Sequential:
        input_shape = (self.target_bands, self.n_time_frames, self.store_extract.number_of_spectrograms)
        sys.path.append('architecture')
        module = importlib.import_module(self.model_architecture)
        return module.architecture(input_shape)
