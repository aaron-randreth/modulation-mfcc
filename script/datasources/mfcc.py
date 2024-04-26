from typing import override

from math import sqrt

from scipy import signal
from scipy.signal import argrelextrema, find_peaks

import librosa

import numpy
import numpy.typing as npt

import pyqtgraph as pg

from .datasource import DataSource, Plotter


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

    if audio_data.ndim > 1:
        return audio_data[channel_nb, :]

    return audio_data


def get_MFCCS_change(
    audio_data: numpy.ndarray,
    signal_sample_rate=10_000,
    tStep=0.005,
    winLen=0.025,
    n_mfcc=13,
    n_fft=512,
    removeFirst=1,
    filtCutoff=12,
    filtOrd=6,
) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
    """
    @Returns: mfcc_times, mfcc_values
    """

    win_length = int(numpy.rint(winLen * signal_sample_rate))
    hop_length = int(numpy.rint(tStep * signal_sample_rate))

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
    myAbsDiff = numpy.sqrt(numpy.gradient(filtMffcs, axis=1) ** 2)

    totChange = numpy.sum(myAbsDiff, axis=0)
    totChange = signal.filtfilt(b1, a1, totChange)

    values = totChange
    times = numpy.arange(len(totChange)) / 200.0

    return times, values


class MfccSource(DataSource):
    file_path: str

    start_idx: int
    end_idx: int

    audio_data: numpy.typing.NDArray[numpy.float64]
    values: numpy.typing.NDArray[numpy.float64]
    times: numpy.typing.NDArray[numpy.float64]

    def __init__(self, file_path: str, start_idx: int = 0, end_idx: int = -1) -> None:
        self.file_path = file_path

        self.start_idx = start_idx
        self.end_idx = end_idx

        self.times = numpy.asarray([])
        self.values = numpy.asarray([])
        self.update()

    @override
    def change_file(self, file_path: str) -> None:
        self.file_path = file_path

        self.start_idx = 0
        self.end_idx = -1
        
        self.update()

    @override
    def get_data(self) -> tuple[npt.NDArray[numpy.float64], npt.NDArray[numpy.float64]]:
        return self.times, self.values

    @override
    def update(self) -> None:
        self.audio_data = load_channel(self.file_path)
        self.times, self.values = get_MFCCS_change(self.audio_data)

    @override
    def get_subset(self, start_idx: int = 0, end_idx: int = -1) -> DataSource:
        return type(self)(self.file_path, start_idx, end_idx)


