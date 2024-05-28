import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import soundfile as sf
import os
import random


def gcc_phat(x1: np.ndarray, x2: np.ndarray, fs=1, interp=1):
    n = x1.shape[0] + x2.shape[0] - 1
    if n % 2 != 0:
        n += 1
    X1 = fft.rfft(x1, n)
    X2 = fft.rfft(x2, n)
    X1 /= np.abs(X1)
    X2 /= np.abs(X2)
    cc = fft.irfft(X1 * np.conj(X2), n=interp * n)
    t_max = n // 2 + 1
    cc = np.concatenate((cc[-t_max:], cc[:t_max]))
    tau = np.argmax(np.abs(cc))
    tau -= t_max
    return tau / (fs * interp), cc


def gcc_phat_feature(xy: np.ndarray, fs: int, len: float) -> np.ndarray:
    _, cc = gcc_phat(xy[:, 0], xy[:, 1], fs=fs)
    c = cc.shape[0]//2
    o = int(len * fs)
    ccs = cc[c-o:c+o]
    assert ccs.shape[0] == gcc_phat_feature_nsamples(fs, len)
    return ccs


def gcc_phat_feature_nsamples(fs: int, len: float) -> int:
    o = int(len * fs)
    return int(o * 2)


def plot_gcc_phat_feature(filename: str, base_path: str, len: float = 0.0007):
    full_filename = os.path.join(base_path, filename)
    xy, fs = sf.read(full_filename)
    fvec = gcc_phat_feature(xy, fs)
    num_samples = len(fvec)
    time_lags = np.linspace(-num_samples // 2,
                            num_samples // 2, num_samples) / fs * 1e6

    plt.figure(figsize=(6, 4))
    plt.plot(time_lags, fvec)
    plt.title(filename)
    plt.xlabel('Time Lag [µs]')
    plt.ylabel('Normalized Correlation Values')
    plt.xlim(-700, 700)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.savefig(f'gcc_phat/gcc_phat_{filename}.png')
    plt.close()


if __name__ == "__main__":
    base_path = '/run/media/pawel/alpha/spatresults/spat'

    # List all files in the directory
    all_files = os.listdir(base_path)

    # Filter to get only .wav files
    wav_files = [f for f in all_files if f.endswith('.wav')]

    # Select 5 random files
    random_files = random.sample(wav_files, 10)

    for filename in random_files:
        plot_gcc_phat_feature(filename, base_path)
