from typing import override

from math import sqrt

from scipy import signal
from scipy.signal import argrelextrema, find_peaks

import librosa

import numpy
import numpy.typing as npt

import pyqtgraph as pg

from .datasource import DataSource, Plotter


class TwoDimPlotter(Plotter):
    """
        Plots a source with data inside two nparray.
        The format of the source returned data should be
        (x coordiantes, y coordinates).
    """

    @override
    def plot_from(
        self, name: str, source: DataSource, pen: str = "#FF5733",
        **kargs
    ) -> pg.PlotDataItem:
        return pg.PlotDataItem(*source.get_data(), name=name, pen=pen, **kargs)

    @override
    def update_plot(self, plot: pg.PlotDataItem, source: DataSource) -> None:
        plot.setData(*source.get_data())
