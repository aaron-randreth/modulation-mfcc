from typing import override

import numpy
import numpy.typing as npt

import pyqtgraph as pg

import parselmouth

from .datasource import DataSource, Plotter


class AudioSource(DataSource):
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
        return self.audio_data.xs(), self.audio_data.values[0]

    @override
    def update(self) -> None:
        self.audio_data = parselmouth.Sound(self.file_path)

    @override
    def get_subset(self, start_idx: int = 0, end_idx: int = -1) -> DataSource:
        return type(self)(self.file_path, start_idx, end_idx)
