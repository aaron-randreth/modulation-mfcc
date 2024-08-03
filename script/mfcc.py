from typing import override

from math import sqrt

from scipy import signal
from scipy.signal import argrelextrema, find_peaks

import librosa
import numpy as np
import numpy
import numpy.typing as npt

import pyqtgraph as pg


def load_channel(
    file_path: str, signal_sample_rate: float = 10_000, channel_nb: int = 0
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Return the audio data of single channel of a file using librosa.

    Parameters
    ----------

    signal_sample_data: float, optional
        The sample rate of the file.

    channel_nb: int, optional
        The channel to return. (the default is the first one)

    Return
    ------

    audio_data: ndarray
        The data of the chosen channel.

    """
    audio_data, _ = librosa.load(file_path, sr=signal_sample_rate, mono=False)

    # if audio_data.ndim > 1:
        # return audio_data[channel_nb, :]

    return audio_data

#scrip leonardo mettre la source !!!
def get_MFCCS_change(
    audio_data: np.ndarray,
    signal_sample_rate=10_000,
    tStep=0.005,
    winLen=0.025,
    n_mfcc=13,
    n_fft=512,
    removeFirst=1,
    filtCutoff=12,
    filtOrd=6,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    @Returns: mfcc_times, mfcc_values
    """

    win_length = int(np.rint(winLen * signal_sample_rate))
    hop_length = int(np.rint(tStep * signal_sample_rate))

    myMfccs = librosa.feature.mfcc(
        y=audio_data,
        sr=signal_sample_rate,
        n_mfcc=n_mfcc,
        win_length=win_length,
        hop_length=hop_length,
        n_fft=n_fft,
    )

    if removeFirst:
        myMfccs = myMfccs[1:, :]

    cutOffNorm = filtCutoff / ((1 / tStep) / 2)
    b1, a1 = signal.butter(filtOrd, cutOffNorm, "lowpass")

    filtMffcs = signal.filtfilt(b1, a1, myMfccs, axis=1)
    myAbsDiff = np.sqrt(np.gradient(filtMffcs, axis=1) ** 2)

    totChange = np.sum(myAbsDiff, axis=0)
    totChange = signal.filtfilt(b1, a1, totChange)

    values = totChange
    times = np.arange(len(totChange)) * tStep
    print(times)

    return times, values
