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
import keras
import yaml
from dotmap import DotMap
from scipy.signal import spectrogram
from scipy.signal.windows import hamming
from sklearn.model_selection import GroupShuffleSplit
from keras import mixed_precision

from gcc_phat import gcc_phat_feature, gcc_phat_feature_nsamples


@dataclass
class Request:
    name: str = "unknown_model"
    input_dir: str = "input"
    memmap_dir: str = "memmap"
    num_samples: int = 23040
    sample_rate: int = 48000
    target_bands: int = 300
    time_window_len: float = 0.04
    time_window_overlap: float = 0.5
    spectrogram_min_freq: int = 100
    spectrogram_max_freq: int = 16000
    recording_time_length: int = 7
    model_architecture: str = "default"
    learn_rate: float = 0.001
    learn_b1: float = 0.9
    learn_b2: float = 0.999
    learn_decay: float = 0
    train_batch_size: int = 8
    train_epochs: int = 128
    train_patience: int = 15
    train_loss: str = "mae"
    dev_test_ratio: float = 1 / 3
    train_val_ratio: float = 1 / 8
    split_seed: int = 2
    train_seed: int = 3
    repetition_iteration: int = 1
    spectrogram_type: str = "MAGNITUDE"
    dtype: str = "float32"
    gcc_phat_len: float = 0.0007


class Model:
    def __init__(self, request: Request):
        self.result_best_val_mae: Optional[float] = None
        self.result_best_epoch: Optional[int] = None
        self.result_test_mae: Optional[dict] = None
        self.model_n_params: Optional[int] = None
        self.result_history: Optional[keras.callbacks.History] = None
        self.time_extract: Optional[float] = None
        self.time_split: Optional[float] = None
        self.time_train: Optional[float] = None

        self.input_spectrogram_magnitude: Optional[np.memmap] = None
        self.input_spectrogram_phase: Optional[np.memmap] = None
        self.x_dev_mag: Optional[np.memmap] = None
        self.x_dev_phase: Optional[np.memmap] = None
        self.x_train_mag: Optional[np.memmap] = None
        self.x_train_phase: Optional[np.memmap] = None
        self.x_val_mag: Optional[np.memmap] = None
        self.x_val_phase: Optional[np.memmap] = None
        self.x_test_mag: Optional[np.memmap] = None
        self.x_test_phase: Optional[np.memmap] = None

        self.store_split: DotMap = DotMap()
        self.store_extract: DotMap = DotMap()
        self.request: Request = request
        self.name = request.name
        self.input_dir = Path(request.input_dir)
        self.memmap_dir = Path(request.memmap_dir)
        self.num_samples = request.num_samples
        self.sample_rate = request.sample_rate
        self.filepaths = [p for p in self.input_dir.iterdir()
                          ][0: self.num_samples]
        self.target_bands = request.target_bands
        self.time_window_len = request.time_window_len
        self.time_window_overlap = request.time_window_overlap
        self.spectrogram_min_freq = request.spectrogram_min_freq
        self.spectrogram_max_freq = request.spectrogram_max_freq
        self.recording_time_length = request.recording_time_length
        self.n_time_frames = int(
            (self.recording_time_length - self.time_window_len)
            / (self.time_window_len * (1 - self.time_window_overlap))
            + 1
        )
        self.model_architecture = request.model_architecture
        self.train_batch_size = request.train_batch_size
        self.train_epochs = request.train_epochs
        self.train_patience = request.train_patience
        self.train_loss = request.train_loss
        # self.train_learn_schedule = keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=request.learn_rate,
        #     decay_steps=10000,
        #     decay_rate=request.learn_decay)
        self.train_optimizer = keras.optimizers.Adam(
            learning_rate=request.learn_rate,
            beta_1=request.learn_b1,
            beta_2=request.learn_b2,
            decay=request.learn_decay,
        )
        self.store_extract.number_of_spectrograms = 2
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
        self.dtype = request.dtype
        self.gcc_phat_len = request.gcc_phat_len

        # self.log_buffer = io.StringIO()

        self.__print_properties_trimmed()

        keras.config.set_dtype_policy(self.dtype)

        # Configure GPU VRAM
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     try:
        #         config = [
        #             tf.config.experimental.VirtualDeviceConfiguration(
        #                 memory_limit=20 * 1024
        #             )
        #         ]
        #         tf.config.experimental.set_virtual_device_configuration(
        #             gpus[0], config)
        #     except RuntimeError as e:
        #         print(e)
        #         traceback.print_exc()

    def extract(self) -> Model:
        start_time = timeit.default_timer()

        print("[extract] Initializing memory")
        self.__initialize_extract_memory()
        self.store_extract.actual_widths = np.zeros(
            self.num_samples, dtype=self.dtype)
        self.store_extract.actual_locations = np.zeros(
            self.num_samples, dtype=self.dtype)
        self.store_extract.actual_recordings = [
            "" for _ in range(self.num_samples)]
        self.store_extract.actual_filenames = [
            "" for _ in range(self.num_samples)]
        self.store_extract.gcc_phat_fvec = np.zeros(
            (self.num_samples, gcc_phat_feature_nsamples(
                self.sample_rate, self.gcc_phat_len)),
            dtype=self.dtype
        )

        print("[extract] Memory initialized")

        last_results_elapsed = []

        def extraction(i, result):
            self.input_spectrogram_magnitude[i] = result.input_spectrogram_magnitude
            self.input_spectrogram_phase[i] = result.input_spectrogram_phase
            self.store_extract.actual_widths[i] = result.actual_width
            self.store_extract.actual_locations[i] = result.actual_location
            self.store_extract.actual_recordings[i] = result.actual_recording
            self.store_extract.actual_filenames[i] = result.actual_filename
            self.store_extract.gcc_phat_fvec[i, :] = result.gcc_phat_fvec
            last_results_elapsed.append(result.elapsed)

            # if i % 10 == 0:
            mean_elapsed = sum(last_results_elapsed) / \
                len(last_results_elapsed)
            print(f"[extract] [{i}] Average spectrogram extraction time per sample: {mean_elapsed:.3f} s [samples: {len(last_results_elapsed)}]")
            last_results_elapsed.clear()

        self.__execute_results_extraction(extraction, multithread=True)
        self.__flush_extract_memory()
        self.store_extract.actual_recordings = np.array(
            self.store_extract.actual_recordings
        )
        self.store_extract.actual_filenames = np.array(
            self.store_extract.actual_filenames
        )

        elapsed = timeit.default_timer() - start_time
        print(f"[extract] Loaded all spectrograms in {elapsed:.2f} s")
        self.time_extract = elapsed

        return self

    def __execute_results_extraction(self, to_execute, multithread=False):
        if multithread:
            with ThreadPoolExecutor(max_workers=2) as executor:
                try:
                    futures = [
                        executor.submit(self._extract_features, filepath)
                        for filepath in self.filepaths
                    ]
                    for i, future in enumerate(futures):
                        try:
                            result = future.result()
                        except Exception as e:
                            print(e)
                            traceback.print_exc()
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise e
                        to_execute(i, result)
                except KeyboardInterrupt as e:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise e
        else:
            for i, filepath in enumerate(self.filepaths):
                result = self._extract_features(filepath)
                to_execute(i, result)

    def try_load_or_compute_input_data(self) -> Model:
        try:
            print("[load] Attempting to load the cached input features")
            self.load_extracted()
        except FileNotFoundError:
            print(
                "[load] Cache does not contain the requested data. Extracting data from scratch"
            )
            self.extract()
            print("[load] Spectrograms extracted, saving them")
            self.save_extracted()
        try:
            print("[load] Attempting to load the cached split data")
            self.load_split()
        except FileNotFoundError:
            print(
                "[load] Cache does not contain the requested split data. Splitting data from scratch"
            )
            self.split()
            self.save_split()
        return self

    def load_split(self) -> Model:
        start_time = timeit.default_timer()
        data_hash = self.__get_split_hash()
        print(f"[load] Split hash: {data_hash}")
        self.store_split = joblib.load(f"/app/data/train/state/{data_hash}_store_split.pkl")
        self.__initialize_split_memory()
        elapsed = timeit.default_timer() - start_time
        print(f"[load] Loaded cached splits in {elapsed:.2f} s")
        return self

    def save_split(self) -> Model:
        start_time = timeit.default_timer()
        data_hash = self.__get_split_hash()
        print(f"[save] Split hash: {data_hash}")
        os.makedirs("/app/data/train/state", exist_ok=True)
        joblib.dump(self.store_split, f"/app/data/train/state/{data_hash}_store_split.pkl")
        elapsed = timeit.default_timer() - start_time
        print(f"[save] Saved cached splits in {elapsed:.2f} s")
        return self

    def load_extracted(self) -> Model:
        start_time = timeit.default_timer()
        data_hash = self.__get_extract_hash()
        print(f"[load] Extraction hash: {data_hash}")
        self.store_extract = joblib.load(
            f"/app/data/train/state/{data_hash}_store_extract.pkl")
        self.__initialize_extract_memory()
        elapsed = timeit.default_timer() - start_time
        print(f"[load] Loaded cached spectrograms in {elapsed:.2f} s")
        return self

    def save_extracted(self) -> Model:
        start_time = timeit.default_timer()
        data_hash = self.__get_extract_hash()
        print(f"[save] Extraction hash: {data_hash}")
        os.makedirs("/app/data/train/state/", exist_ok=True)
        joblib.dump(self.store_extract, f"/app/data/train/state/{data_hash}_store_extract.pkl")
        elapsed = timeit.default_timer() - start_time
        print(f"[save] Saved spectrograms in {elapsed:.2f} s")
        return self

    def plot_features(self) -> Model:
        output_dir = self.__get_model_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        start_time = timeit.default_timer()
        n_samples = self.x_train_mag.shape[0]
        n_spectrograms_mag = self.x_train_mag.shape[-1]
        n_spectrograms_phase = self.x_train_phase.shape[-1]
        n_spectrograms_all = n_spectrograms_mag + n_spectrograms_phase
        np.random.seed(345)
        random_samples = np.random.choice(n_samples, size=5, replace=False)

        for sample in random_samples:
            plt.figure(figsize=(20, 10))

            for i_spectrogram in range(n_spectrograms_mag):
                plt.subplot(2, int(np.ceil(n_spectrograms_all / 2)),
                            i_spectrogram + 1)
                data = self.x_train_mag[sample, :, :, i_spectrogram]
                plt.pcolormesh(data, shading="gouraud")
                plt.title(f"Magnitude Spectrogram {i_spectrogram}")

            for i_spectrogram in range(n_spectrograms_phase):
                plt.subplot(
                    2,
                    int(np.ceil(n_spectrograms_all / 2)),
                    n_spectrograms_mag + i_spectrogram + 1,
                )
                data = self.x_train_phase[sample, :, :, i_spectrogram]
                plt.pcolormesh(data, shading="gouraud")
                plt.title(f"Phase Spectrogram {i_spectrogram}")

            plt.tight_layout()
            plt.savefig(output_dir / f"features_{sample}.png")

        elapsed = timeit.default_timer() - start_time
        print(f"[draw] Plotted features in {elapsed} s")

        return self

    # Split data into training, validation and testing
    def split(self) -> Model:
        start_time_all = timeit.default_timer()
        print("[split] [1/3] Splitting data into development and testing set")
        start_time = timeit.default_timer()

        x_all = self.input_spectrogram_magnitude
        gss = GroupShuffleSplit(
            n_splits=1, test_size=self.dev_test_ratio, random_state=self.split_seed
        )
        idx_dev, idx_test = next(
            gss.split(x_all, groups=self.store_extract.actual_recordings)
        )

        y_dev_width = self.store_extract.actual_widths[idx_dev]
        y_dev_location = self.store_extract.actual_locations[idx_dev]
        self.store_split.y_test_width = self.store_extract.actual_widths[idx_test]
        self.store_split.y_test_location = self.store_extract.actual_locations[idx_test]
        self.store_split.actual_recordings_dev = self.store_extract.actual_recordings[
            idx_dev
        ]
        self.store_split.actual_recordings_test = self.store_extract.actual_recordings[
            idx_test
        ]

        gss = GroupShuffleSplit(
            n_splits=1, test_size=self.train_val_ratio, random_state=self.split_seed + 1
        )
        idx_train, idx_val = next(
            gss.split(idx_dev, groups=self.store_split.actual_recordings_dev)
        )

        def __shape(samples):
            return (
                samples,
                self.target_bands,
                self.n_time_frames,
                self.store_extract.number_of_spectrograms,
            )

        print("[split] [2/5] Initializing memory maps")
        self.store_split.x_dev_mag_shape = __shape(len(idx_dev))
        self.store_split.x_dev_phase_shape = __shape(len(idx_dev))
        self.store_split.x_train_mag_shape = __shape(len(idx_train))
        self.store_split.x_train_phase_shape = __shape(len(idx_train))
        self.store_split.x_val_mag_shape = __shape(len(idx_val))
        self.store_split.x_val_phase_shape = __shape(len(idx_val))
        self.store_split.x_test_mag_shape = __shape(len(idx_test))
        self.store_split.x_test_phase_shape = __shape(len(idx_test))
        self.__initialize_split_memory()

        elapsed = timeit.default_timer() - start_time
        print(f"[split] [2/5] Done ({elapsed:.2f} s)")
        print("[split] [3/5] Assigning allocated mmap")
        start_time = timeit.default_timer()

        self.x_dev_mag[:] = self.input_spectrogram_magnitude[idx_dev]
        self.x_dev_phase[:] = self.input_spectrogram_phase[idx_dev]
        self.store_split.x_dev_gcc_phat = self.store_extract.gcc_phat_fvec[idx_dev, :]

        self.x_test_mag[:] = self.input_spectrogram_magnitude[idx_test]
        self.x_test_phase[:] = self.input_spectrogram_phase[idx_test]
        self.store_split.x_test_gcc_phat = self.store_extract.gcc_phat_fvec[idx_test, :]

        self.store_split.idx_dev = idx_dev
        self.store_split.idx_test = idx_test

        elapsed = timeit.default_timer() - start_time
        print(f"[split] [3/5] Done ({elapsed:.2f} s)")
        print(
            "[split] [4/5] Splitting development set into training and validation set"
        )
        start_time = timeit.default_timer()

        self.x_train_mag[:] = self.x_dev_mag[idx_train]
        self.x_train_phase[:] = self.x_dev_phase[idx_train]
        self.store_split.x_train_gcc_phat = self.store_split.x_dev_gcc_phat[
            idx_train, :
        ]

        self.x_val_mag[:] = self.x_dev_mag[idx_val]
        self.x_val_phase[:] = self.x_dev_phase[idx_val]
        self.store_split.x_val_gcc_phat = self.store_split.x_dev_gcc_phat[idx_val, :]

        self.store_split.y_train_width = y_dev_width[idx_train]
        self.store_split.y_val_width = y_dev_width[idx_val]

        self.store_split.y_train_location = y_dev_location[idx_train]
        self.store_split.y_val_location = y_dev_location[idx_val]

        self.store_split.actual_recordings_train = self.store_extract.actual_recordings[
            idx_train
        ]
        self.store_split.actual_recordings_val = self.store_extract.actual_recordings[
            idx_val
        ]
        self.store_split.idx_train = idx_train
        self.store_split.idx_val = idx_val

        elapsed = timeit.default_timer() - start_time
        print(f"[split] [4/5] Done ({elapsed:.2f} s)")
        print("[split] [5/5] Normalizing data")
        start_time = timeit.default_timer()

        x_train_mag_min = np.min(self.x_train_mag)
        x_train_mag_max = np.max(self.x_train_mag)
        x_train_mag_mean = np.mean(self.x_train_mag)
        x_train_mag_std = np.std(self.x_train_mag)
        x_train_phase_min = np.min(self.x_train_phase)
        x_train_phase_max = np.max(self.x_train_phase)
        x_train_gcc_phat_min = np.min(self.store_split.x_train_gcc_phat)
        x_train_gcc_phat_max = np.max(self.store_split.x_train_gcc_phat)
        x_train_gcc_phat_mean = np.mean(self.store_split.x_train_gcc_phat)
        x_train_gcc_phat_std = np.std(self.store_split.x_train_gcc_phat)

        self.x_train_mag[:] = (
            self.x_train_mag - x_train_mag_mean) / x_train_mag_std
        self.x_val_mag[:] = (
            self.x_val_mag - x_train_mag_mean) / x_train_mag_std
        self.x_test_mag[:] = (
            self.x_test_mag - x_train_mag_mean) / x_train_mag_std

        self.x_train_phase[:] = (self.x_train_phase - x_train_phase_min) / (
            x_train_phase_max - x_train_phase_min
        )
        self.x_val_phase[:] = (self.x_val_phase - x_train_phase_min) / (
            x_train_phase_max - x_train_phase_min
        )
        self.x_test_phase[:] = (self.x_test_phase - x_train_phase_min) / (
            x_train_phase_max - x_train_phase_min
        )

        self.store_split.x_train_gcc_phat = (
            self.store_split.x_train_gcc_phat - x_train_gcc_phat_mean
        ) / x_train_gcc_phat_std
        self.store_split.x_val_gcc_phat = (
            self.store_split.x_val_gcc_phat - x_train_gcc_phat_mean
        ) / x_train_gcc_phat_std
        self.store_split.x_test_gcc_phat = (
            self.store_split.x_test_gcc_phat - x_train_gcc_phat_mean
        ) / x_train_gcc_phat_std

        self.store_split.y_train_width = self.store_split.y_train_width / 45
        self.store_split.y_val_width = self.store_split.y_val_width / 45
        self.store_split.y_test_width = self.store_split.y_test_width / 45

        self.store_split.y_train_location = self.store_split.y_train_location / 45
        self.store_split.y_val_location = self.store_split.y_val_location / 45
        self.store_split.y_test_location = self.store_split.y_test_location / 45

        elapsed = timeit.default_timer() - start_time

        print(f"[split] [5/5] Done ({elapsed:.2f} s)")

        print(
            f"[split] x_train_mag shape: {self.x_train_mag.shape}, "
            f"x_train_phase shape: {self.x_train_phase.shape}, "
            f"y_train_width shape {self.store_split.y_train_width.shape}"
            f"y_train_location shape {self.store_split.y_train_location.shape}"
        )
        print(
            f"[split] x_val_mag shape: {self.x_val_mag.shape}, "
            f"x_val_phase shape: {self.x_val_phase.shape}, "
            f"y_val_width shape {self.store_split.y_val_width.shape}"
            f"y_val_location shape {self.store_split.y_val_location.shape}"
        )
        print(
            f"[split] x_test_mag shape: {self.x_test_mag.shape}, "
            f"x_test_phase shape: {self.x_test_phase.shape}, "
            f"y_test_width shape {self.store_split.y_test_width.shape}"
            f"y_test_location shape {self.store_split.y_test_location.shape}"
        )

        self.time_split = timeit.default_timer() - start_time_all

        self.__flush_split_memory()

        return self

    # Train
    def train(self) -> Model:
        start_time = timeit.default_timer()

        tf.random.set_seed(self.train_seed)
        np.random.seed(self.train_seed)
        random.seed(self.train_seed)

        dest_model_path = self.__get_model_output_dir() / "model.keras"
        os.makedirs(os.path.dirname(dest_model_path), exist_ok=True)

        model = self.__load_topology()
        model.summary()

        self.__save_model_params(model)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.train_patience, restore_best_weights=False
        )
        checkpoint = keras.callbacks.ModelCheckpoint(
            dest_model_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        )
        model.compile(
            optimizer=self.train_optimizer,
            loss={"out_width": self.train_loss,
                  "out_location": self.train_loss},
            metrics={"out_width": ["mae"], "out_location": ["mae"]},
        )
        model_inout = self.__get_model_inout()
        self.result_history = model.fit(
            model_inout.in_train,
            model_inout.out_train,
            batch_size=self.train_batch_size,
            epochs=self.train_epochs,
            validation_data=(model_inout.in_val, model_inout.out_val),
            validation_batch_size=self.train_batch_size,
            callbacks=[early_stop, checkpoint],
        )

        self.result_best_val_mae = min(self.result_history.history["val_loss"])
        self.result_best_epoch = (
            self.result_history.history["val_loss"].index(
                self.result_best_val_mae) + 1
        )

        print(
            "[train] The training procedure ended with validation loss:",
            self.result_best_val_mae,
        )

        model.load_weights(dest_model_path)

        self.result_test_mae = model.evaluate(
            model_inout.in_test, model_inout.out_test, verbose=0, return_dict=True
        )
        print("[train] Test loss:", self.result_test_mae)
        print(
            "[train] Width average error (in degree):",
            self.result_test_mae["out_width_mae"] * 90,
        )
        print(
            "[train] Location average error (in degree):",
            self.result_test_mae["out_location_mae"] * 45,
        )

        self.time_train = timeit.default_timer() - start_time

        return self

    def __get_model_inout(self) -> DotMap:
        model_inout = DotMap()
        model_inout.in_train = {
            "in_mag": self.x_train_mag,
            # "in_phase": self.x_train_mag,
            "in_gcc": self.store_split.x_train_gcc_phat,
        }
        model_inout.in_val = {
            "in_mag": self.x_val_mag,
            # "in_phase": self.x_val_mag,
            "in_gcc": self.store_split.x_val_gcc_phat,
        }
        model_inout.in_test = {
            "in_mag": self.x_test_mag,
            # "in_phase": self.x_test_phase,
            "in_gcc": self.store_split.x_test_gcc_phat,
        }
        model_inout.out_train = {
            "out_width": self.store_split.y_train_width,
            "out_location": self.store_split.y_train_location,
        }
        model_inout.out_val = {
            "out_width": self.store_split.y_val_width,
            "out_location": self.store_split.y_val_location,
        }
        model_inout.out_test = {
            "out_width": self.store_split.y_test_width,
            "out_location": self.store_split.y_test_location,
        }
        return model_inout

    # Plot Training History

    def plot_history(self) -> Model:
        loss = self.result_history.history["loss"]
        val_loss = self.result_history.history["val_loss"]
        # mae = self.result_history.history['mae']
        # val_mae = self.result_history.history['val_mae']
        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(12, 5))

        # Plotting training and validation loss
        # plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
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
        joblib.dump(self.request, dest_dir / "request.pkl")
        with open(dest_dir / "request.yaml", "w") as f:
            yaml.dump(asdict(self.request), f)

        # save source
        current_script_path = os.path.abspath(__file__)
        with open(current_script_path, "r") as current_script_source:
            with open(dest_dir / "source.pyarch", "w") as dest_script_source:
                dest_script_source.write(current_script_source.read())

        # save splits
        pd.DataFrame(
            {"recordings": np.unique(self.store_split.actual_recordings_dev)}
        ).to_csv(dest_dir / "recordings_dev.csv")
        pd.DataFrame(
            {"recordings": np.unique(self.store_split.actual_recordings_train)}
        ).to_csv(dest_dir / "recordings_train.csv")
        pd.DataFrame(
            {"recordings": np.unique(self.store_split.actual_recordings_val)}
        ).to_csv(dest_dir / "recordings_val.csv")
        pd.DataFrame(
            {"recordings": np.unique(self.store_split.actual_recordings_test)}
        ).to_csv(dest_dir / "recordings_test.csv")

        # save architecture source
        architecture_script_path = (
            Path("/app/src/train/architecture") / f"{self.model_architecture}.py"
        )
        with open(architecture_script_path, "r") as architecture_script_source:
            with open(dest_dir / "architecture.pyarch", "w") as dest_script_source:
                dest_script_source.write(architecture_script_source.read())

        # save metadata
        metadata = DotMap()
        metadata.history = self.result_history.history
        joblib.dump(metadata, dest_dir / "metadata.pkl")

        # delete old score if exists and save actual scores
        for score_filename in dest_dir.glob("score_*.csv"):
            score_filename.unlink()
        pd.DataFrame(
            {
                "best_val_mae": [self.result_best_val_mae],
                "test_score": [self.result_test_mae["loss"]],
                "test_score_width": [self.result_test_mae["out_width_mae"]],
                "test_score_location": [self.result_test_mae["out_location_mae"]],
                "best_epoch": [self.result_best_epoch],
                "params": [self.model_n_params],
            }
        ).to_csv(dest_dir / f"score_{self.result_best_val_mae:0.4f}.csv")

        # save properties
        max_value_len = 100
        properties = []
        values = []
        for prop, value in vars(self).items():
            value = str(value)
            if len(value) > max_value_len:
                value = value[:max_value_len] + " ..."
            properties.append(prop)
            values.append(value)
        pd.DataFrame({"property": properties, "value": values}).to_csv(
            dest_dir / "properties.csv"
        )

        return self

    def predict_test_data(self) -> pd.DataFrame:
        output_dir = self.__get_model_output_dir()
        model_path = output_dir / "model.keras"
        model = keras.models.load_model(model_path)
        model_inout = self.__get_model_inout()
        predictions = model.predict(model_inout.in_test)
        test_filenames = self.store_extract.actual_filenames[self.store_split.idx_test]

        return pd.DataFrame(
            {
                "filename": test_filenames,
                "actual_location": model_inout.out_test["out_location"],
                "actual_width": model_inout.out_test["out_width"],
                "predicted_width": predictions[0][:, 0],
                "predicted_location": predictions[1][:, 0],
            }
        )

    def __get_extract_hash(self) -> str:
        args = [
            self.input_dir,
            self.target_bands,
            self.time_window_len,
            self.time_window_overlap,
            self.spectrogram_type,
            self.dtype,
            self.gcc_phat_len
        ]
        combined_string = "|".join(str(arg) for arg in args)
        encoded_string = combined_string.encode("utf-8")
        hash_obj = hashlib.sha256(encoded_string)
        return hash_obj.hexdigest()[:8]

    def __get_split_hash(self) -> str:
        args = [self.__get_extract_hash(), self.split_seed]
        combined_string = "|".join(str(arg) for arg in args)
        encoded_string = combined_string.encode("utf-8")
        hash_obj = hashlib.sha256(encoded_string)
        return hash_obj.hexdigest()[:8]

    def __save_model_params(self, model: keras.Model):
        model.count_params()
        self.model_n_params = model.count_params()

    def __get_model_output_dir(self) -> Path:
        return Path("/app/data/train/out_models") / self.name / str(self.repetition_iteration)

    def __print_properties_trimmed(self) -> None:
        n = 100
        for prop, value in vars(self).items():
            if value is None:
                continue
            value = str(value)
            if isinstance(value, str) and len(value) > n:
                value = value[:n] + " ..."
            print(f"\t{prop}: {value}")

    def __aggregate_spectrogram_to_n_bands(
        self, frequencies: np.ndarray, sxx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = (frequencies >= self.spectrogram_min_freq) & (
            frequencies <= self.spectrogram_max_freq
        )
        valid_freq_indices = np.where(mask)[0]
        frequencies = frequencies[valid_freq_indices]
        sxx = sxx[valid_freq_indices, :]
        total_freq_range = frequencies[-1] - frequencies[0]
        band_freq_range = total_freq_range / self.target_bands
        aggregated_sxx = np.zeros(
            (self.target_bands, sxx.shape[1]), dtype=np.complex128
        )
        new_freq = np.linspace(
            self.spectrogram_min_freq, self.spectrogram_max_freq, num=self.target_bands
        )

        for i in range(self.target_bands):
            band_start_freq = self.spectrogram_min_freq + i * band_freq_range
            band_end_freq = band_start_freq + band_freq_range
            mask = (frequencies >= band_start_freq) & (
                frequencies < band_end_freq)
            freq_indices = np.where(mask)[0]

            if len(freq_indices) > 0:
                aggregated_sxx[i, :] = np.mean(sxx[freq_indices, :], axis=0)

        return new_freq, aggregated_sxx

    def _extract_features(self, filepath: Path) -> DotMap:
        if not filepath.is_file():
            raise ValueError("Invalid file path")

        start_time = timeit.default_timer()

        actual_width = float(str(filepath).split("_")[-4].replace("width", ""))
        actual_location = float(str(filepath).split(
            "_")[-2].replace("azoffset", ""))
        actual_recording = str(filepath).split("_")[0].split(os.sep)[-1]

        audio_data, sample_rate = sf.read(filepath)
        assert self.sample_rate == sample_rate
        _, _, features_left, features_right = self.__compute_spectrogram(
            audio_data, sample_rate
        )

        gcc_phat_fvec = gcc_phat_feature(
            audio_data, sample_rate, self.gcc_phat_len)

        elapsed = timeit.default_timer() - start_time

        result = DotMap()
        result.input_spectrogram_magnitude = np.stack(
            [
                np.log10(features_left * np.conjugate(features_left)),
                np.log10(features_right * np.conjugate(features_right)),
            ],
            axis=-1,
        )
        result.input_spectrogram_phase = np.stack(
            [np.angle(features_left), np.angle(features_right)], axis=-1
        )

        result.gcc_phat_fvec = gcc_phat_fvec
        result.actual_width = actual_width
        result.actual_location = actual_location
        result.actual_recording = actual_recording
        result.actual_filename = str(filepath)
        result.elapsed = elapsed

        return result

    def __compute_spectrogram(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nperseg = int(self.time_window_len * sample_rate)
        noverlap = int(nperseg * self.time_window_overlap)

        def __spectrogram(channel):
            freq, time, zxx = spectrogram(
                channel,
                fs=sample_rate,
                window=hamming(nperseg),
                noverlap=noverlap,
                mode="complex",
            )

            freq, zxx = self.__aggregate_spectrogram_to_n_bands(freq, zxx)

            return freq, time, zxx

        freq_left, times_left, spectrogram_left = __spectrogram(
            audio_data[:, 0])
        _, _, spectrogram_right = __spectrogram(audio_data[:, 1])

        return freq_left, times_left, spectrogram_left, spectrogram_right

    def __load_topology(self):
        input_shape = (
            self.target_bands,
            self.n_time_frames,
            self.store_extract.number_of_spectrograms,
        )
        sys.path.append("/app/src/train/architecture")
        module = importlib.import_module(self.model_architecture)
        return module.architecture(input_shape)

    def __initialize_extract_memory(self):
        self.input_spectrogram_magnitude = self.__get_extract_memory(
            "input_spectrogram_magnitude", self.store_extract.input_features_shape
        )
        self.input_spectrogram_phase = self.__get_extract_memory(
            "input_spectrogram_phase", self.store_extract.input_features_shape
        )

    def __initialize_split_memory(self):
        self.x_dev_mag = self.__get_split_memory(
            "x_dev_mag", self.store_split.x_dev_mag_shape
        )
        self.x_dev_phase = self.__get_split_memory(
            "x_dev_phase", self.store_split.x_dev_phase_shape
        )
        self.x_train_mag = self.__get_split_memory(
            "x_train_mag", self.store_split.x_train_mag_shape
        )
        self.x_train_phase = self.__get_split_memory(
            "x_train_phase", self.store_split.x_train_phase_shape
        )
        self.x_val_mag = self.__get_split_memory(
            "x_val_mag", self.store_split.x_val_mag_shape
        )
        self.x_val_phase = self.__get_split_memory(
            "x_val_phase", self.store_split.x_val_phase_shape
        )
        self.x_test_mag = self.__get_split_memory(
            "x_test_mag", self.store_split.x_test_mag_shape
        )
        self.x_test_phase = self.__get_split_memory(
            "x_test_phase", self.store_split.x_test_phase_shape
        )

    def __get_extract_memory(self, name: str, shape: tuple | np.ndarray) -> np.ndarray:
        extract_data_hash = self.__get_extract_hash()
        filename = f"{self.memmap_dir}/{extract_data_hash}_{name}"
        memory_data = np.memmap(
            filename,
            dtype=self.dtype,
            mode="r+" if os.path.exists(filename) else "w+",
            shape=shape,
        )
        return memory_data

    def __get_split_memory(self, name: str, shape: tuple | np.ndarray) -> np.ndarray:
        split_data_hash = self.__get_split_hash()
        filename = f"{self.memmap_dir}/{split_data_hash}_{name}"
        memory_data = np.memmap(
            filename,
            dtype=self.dtype,
            mode="r+" if os.path.exists(filename) else "w+",
            shape=shape,
        )
        return memory_data

    def __flush_extract_memory(self):
        self.input_spectrogram_magnitude.flush()
        self.input_spectrogram_phase.flush()

    def __flush_split_memory(self):
        self.x_dev_mag.flush()
        self.x_dev_phase.flush()
        self.x_train_mag.flush()
        self.x_train_phase.flush()
        self.x_val_mag.flush()
        self.x_val_phase.flush()
        self.x_test_mag.flush()
        self.x_test_phase.flush()
