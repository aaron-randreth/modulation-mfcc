from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

import pyqtgraph as pg

import praat_py_ui.tiers as ui_tiers


class DataSource(ABC):
    def get_data(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pass


    @abstractmethod
    def change_file(self, file_path: str) -> None:
        pass

    @abstractmethod
    def get_subset(self, start_idx: int, end_idx: int) -> "DataSource":
        pass

    @abstractmethod
    def update(self) -> None:
        pass


class Plotter(ABC):

    @abstractmethod
    def plot_from(self, name: str, source: DataSource, pen: str = "#000000",
                  **kargs) -> pg.PlotDataItem | pg.ImageItem:
        pass

    @abstractmethod
    def update_plot(self, plot: pg.PlotDataItem, source: DataSource) -> None:
        pass


class DataDisplay:
    id: str
    name: str
    _data_source: DataSource
    plot: pg.PlotDataItem | pg.ImageItem
    _plotter: Plotter

    def __init__(
        self, id: str, name: str, data_source: DataSource, plotter: Plotter,
        **kargs
    ) -> None:
        self.id = id
        self.name = name
        self._data_source = data_source
        self._plotter = plotter
        self.plot = self._plotter.plot_from(self.name, self._data_source,
                                            **kargs)

    def change_file(self, file_path: str) -> None:
        self._data_source.change_file(file_path)
        self.update()

    def update(self):
        self._data_source.update()
        self._plotter.update_plot(self.plot, self._data_source)


class PlotDisplay:
    file_path: str
    principal_plot: pg.PlotItem
    plots: dict[str, DataDisplay]

    def __init__(self, title: str) -> None:
        self.principal_plot = pg.plot()

        self.principal_plot.setWindowTitle(title)
        self.principal_plot.addLegend(brush=pg.mkBrush(255, 255, 255, 255))
        self.principal_plot.showGrid(x=True, y=True)

        self.plots = {}

    def add_source(self, displayed_source: DataDisplay) -> None:
        self.plots[displayed_source.id] = displayed_source
        self.principal_plot.addItem(displayed_source.plot)

    def get_sources(self) -> list[tuple[str, str, bool]]:
        """
        Return a human readable list of available sources.

        Return
        ------

        sources : list[tuple[str, str, bool]
            A list of (source_id, source_name, is_displayed) for all sources

        """
        return [
            (id, disp.name, disp.plot.isVisible()) for id, disp in self.plots.items()
        ]

    def set_source_visible(self, source_id: str, is_visible: bool) -> None:
        if source_id not in self.plots:
            raise ValueError("Source id not found")

        self.plots[source_id].plot.setVisible(is_visible)

    def change_file(self, file_path: str) -> None:
        self.file_path = file_path

        for p in self.plots.values():
            p.change_file(self.file_path)

    def update_sources(self) -> None:
        for s in self.plots.values():
            s.update()
