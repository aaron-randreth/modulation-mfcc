import sys
from scipy.signal import argrelextrema
from math import sqrt

from scipy import signal
from librosa import feature as lf
from scipy.signal import find_peaks
import numpy as np
from PyQt5.QtWidgets import QToolBar, QAction
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QGridLayout,
    QLabel,
    QScrollArea,
)

import pyqtgraph as pg

import parselmouth

pg.setConfigOptions(foreground="black", background="w")

import tgt
from PyQt5.QtCore import Qt
from praat_py_ui import (
    tiers as ui_tiers,
    textgridtools as ui_tgt,
    spectrogram as specto,
    parselmouth_calc as calc,
)

from datasources.mfcc import load_channel, get_MFCCS_change

from scrollable_window import Info, InfoBox, Output


def create_plot_widget(x, y):
    plot = pg.PlotWidget()
    plot.plot(x=x, y=y)

    # plot.setLimits(xMin=xlim[0], xMax=xlim[1], yMin=ylim[0], yMax=ylim[1])
    plot.setMouseEnabled(x=True, y=False)
    # plot.setLabel("bottom", "Temps", units="s")
    # plot.getAxis("left").setWidth(60)

    return plot


class MinMaxFinder:

    def find_subsection(self, x, y, selected_interval):
        start_time = float(selected_interval.start_time)
        end_time = float(selected_interval.end_time)
        fs = 200
        start_index = int(start_time * fs)
        end_index = int(end_time * fs)

        start_index = max(start_index, 0)
        end_index = min(end_index, len(y))

        interval_times = x[start_index:end_index]
        interval_values = y[start_index:end_index]

        return interval_times, interval_values

    def analyse_minimum(self, x, y, selected_interval):
        if selected_interval is not None:
            interval_times, interval_values = self.find_subsection(
                x, y, selected_interval
            )
        else:
            interval_times, interval_values = x, y

        # Trouver les minimums dans l'intervalle
        min_peaks, _ = find_peaks(
            -interval_values
        )  # Utiliser -interval_values pour trouver les minimums

        if len(min_peaks) <= 1:  # TODO verif if 1 is correct
            return [], []

        min_times = interval_times[min_peaks]
        min_values = interval_values[min_peaks]

        return min_times, min_values

    def analyse_maximum(self, x, y, selected_interval):
        if selected_interval is not None:
            interval_times, interval_values = self.find_subsection(
                x, y, selected_interval
            )
        else:
            interval_times, interval_values = x, y

        min_peaks, _ = find_peaks(-interval_values)

        initial_peaks, _ = find_peaks(interval_values)

        if len(initial_peaks) == 1:
            peaks = initial_peaks
            peak_times_final = interval_times[peaks] if len(peaks) > 0 else []
            peak_values_final = interval_values[peaks] if len(peaks) > 0 else []
            return peak_times_final, peak_values_final

        peak_times = interval_times[initial_peaks]
        time_gaps = np.diff(peak_times)

        q75, q25 = np.percentile(time_gaps, [100, 10])
        iqr = q75 - q25
        min_distance_time = iqr - 0.03
        min_distance_samples = max(
            int(
                min_distance_time
                * len(interval_values)
                / (interval_times[-1] - interval_times[0])
            ),
            1,
        )

        peaks, _ = find_peaks(
            interval_values, distance=min_distance_samples, prominence=1
        )
        peak_times_final = interval_times[peaks]
        peak_values_final = interval_values[peaks]

        # Vérifier si le dernier maximum dépasse le dernier minimum
        if len(min_peaks) == 0 or len(peak_values_final) == 0:
            return peak_times_final, peak_values_final

        last_max_index = peaks[-1]
        last_min_index = min_peaks[-1]

        if last_min_index <= last_min_index:
            return peak_times_final, peak_values_final

        # Le dernier maximum dépasse le dernier minimum.
        peak_times_final = np.delete(peak_times_final, -1)
        peak_values_final = np.delete(peak_values_final, -1)

        return peak_times_final, peak_values_final


class MinMaxAnalyser(QWidget):
    name: str

    extremum: MinMaxFinder
    toolbar: QToolBar
    plot_widget: pg.PlotWidget

    max_points: pg.ScatterPlotItem
    min_points: pg.ScatterPlotItem

    def __init__(self, name: str, x, y, extremum: MinMaxFinder, get_interval) -> None:
        super().__init__()

        self.name = name
        self.x = x
        self.y = y
        self.extremum = extremum
        self.get_interval = get_interval

        self.__init_ui()

    def __init_ui(self) -> None:
        layout = QVBoxLayout()

        self.toolbar = QToolBar()
        self.config_toolbar(self.toolbar)

        self.plot_widget = pg.PlotWidget()

        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMaximumHeight(400)

        self.curve = self.plot_widget.plot(self.x, self.y, pen="red", name=self.name)
        self.curve.setCurveClickable(True)
        self.curve.sigClicked.connect(self.add_point_on_click)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)

    def config_toolbar(self, toolbar: QToolBar) -> None:

        self.do_maximum_analysis = QAction("Analyse max", self)
        self.do_minimum_analysis = QAction("Analyse min", self)

        self.do_maximum_analysis.triggered.connect(self.compute_max)
        self.do_minimum_analysis.triggered.connect(self.compute_min)

        self.manual_peak_removal = self.toggleable_action("Toggle Manual Peak Removal")
        self.manual_peak_maximum_addition = self.toggleable_action(
            "Toggle Manual Maximum peak Addition"
        )
        self.manual_peak_minimum_addition = self.toggleable_action(
            "Toggle Manual Minimum peak Addition"
        )

        toolbar.addAction(self.do_maximum_analysis)
        toolbar.addAction(self.do_minimum_analysis)

        toolbar.addSeparator()

        toolbar.addAction(self.manual_peak_removal)
        toolbar.addAction(self.manual_peak_maximum_addition)
        toolbar.addAction(self.manual_peak_minimum_addition)

    def toggleable_action(
        self,
        label: str,
    ) -> QAction:

        action = QAction(label, self)
        action.setCheckable(True)
        action.setChecked(False)

        return action

    def add_point_on_click(
        self,
        clicked_curve: pg.PlotDataItem,
        event: "MouseClickEvent",  # TODO Find where to import this from
    ) -> None:

        if (
            not self.manual_peak_maximum_addition.isChecked()
            and not self.manual_peak_minimum_addition.isChecked()
        ):
            return

        mouse_point = event.pos()
        x, y = mouse_point.x(), mouse_point.y()

        points_x, points_y = self.max_points.getData()

        self.max_points.setData(np.append(points_x, x), np.append(points_y, y))

    def remove_peak_on_click(self, clicked_scatter_plot, clicked_points):
        if not self.manual_peak_removal.isChecked():
            return

        self.plot_widget.setCursor(Qt.PointingHandCursor)

        points_data = list(clicked_scatter_plot.getData())
        points_data[0] = list(points_data[0])
        points_data[1] = list(points_data[1])

        to_remove = []

        for point in clicked_points:
            pos = point.viewPos()

            x_idx = points_data[0].index(pos.x())

            if points_data[1][x_idx] != pos.y():
                continue

            to_remove.append(x_idx)

        for idx in sorted(to_remove, reverse=True):
            points_data[0].pop(idx)
            points_data[1].pop(idx)

        clicked_scatter_plot.setData(*points_data)

    def compute_min(self, interval) -> None:
        if self.get_interval is not None:
            interval = self.get_interval()
        else:
            interval = None

        x_min, y_min = self.extremum.analyse_minimum(self.x, self.y, interval)
        # No minimum found
        if len(x_min) == 0 or len(y_min) == 0:
            return

        self.min_points = pg.ScatterPlotItem(
            name="min",
            x=x_min,
            y=y_min,
            symbol="o",
            size=10,
            pen=pg.mkPen("r"),
            brush=pg.mkBrush("r"),
        )

        self.min_points.sigClicked.connect(self.remove_peak_on_click)
        self.plot_widget.getPlotItem().addItem(self.min_points)

    def compute_max(self, interval) -> None:
        if self.get_interval is not None:
            interval = self.get_interval()
        else:
            interval = None

        x_max, y_max = self.extremum.analyse_maximum(self.x, self.y, interval)

        # No maximum found
        if len(x_max) == 0 or len(y_max) == 0:
            return

        self.max_points = pg.ScatterPlotItem(
            name="max",
            x=x_max,
            y=y_max,
            symbol="x",
            size=10,
            pen=pg.mkPen("g"),
            brush=pg.mkBrush("b"),
        )

        self.max_points.sigClicked.connect(self.remove_peak_on_click)
        self.plot_widget.getPlotItem().addItem(self.max_points)


class AudioAnalyzer(QMainWindow):
    textgrid: ui_tiers.TextGrid | None = None

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Analyzer")

        main_widget = QWidget(self)
        layout = QHBoxLayout()

        # self.audio_file_label = QLabel("Aucun fichier audio chargé")
        # layout.addWidget(self.audio_file_label, 0, 0, 1, 3)

        curve_area, self.curve_layout = self.curves_container()

        layout.addWidget(curve_area)
        layout.addWidget(self.button_container())

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def curves_container(self) -> QWidget:
        curve_parent = QWidget()
        scroll_area = QScrollArea()
        scroll_layout = QVBoxLayout()

        curve_parent.setLayout(scroll_layout)

        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(curve_parent)

        return scroll_area, scroll_layout

    def button_container(self) -> QWidget:
        button_parent = QWidget()
        layout = QVBoxLayout()

        self.button = QPushButton("Load Audio File")
        self.eva_button = QPushButton("Load EVA File")
        self.annotation_button = QPushButton("Load Textgrid Annotation")
        self.annotation_save_button = QPushButton("Save TextGrid Annotation")

        layout.addWidget(self.button)
        layout.addWidget(self.eva_button)
        layout.addWidget(self.annotation_button)
        layout.addWidget(self.annotation_save_button)

        self.button.clicked.connect(self.load_audio)
        self.eva_button.clicked.connect(self.load_eva)
        self.annotation_button.clicked.connect(self.load_annotations)
        self.annotation_save_button.clicked.connect(self.save_annotations)

        button_parent.setLayout(layout)

        return button_parent

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav)"
        )
        if not file_path:
            return

        # self.audio_file_label.setText(f"Fichier audio chargé : {file_path}")

        sound_data = calc.Parselmouth(file_path)
        snd = sound_data.get_sound()
        spc = sound_data.get_spectrogram()

        self.curve_layout.addWidget(
            create_plot_widget(snd.timestamps, snd.amplitudes[0])
        )

        self.curve_layout.addWidget(
            specto.create_spectrogram_plot(
                spc.frequencies, spc.timestamps, spc.data_matrix
            )
        )

        audio_data = load_channel(file_path)
        x_mfccs, y_mfccs = get_MFCCS_change(audio_data)
        a = MinMaxAnalyser(
            "Mfcc", x_mfccs, y_mfccs, MinMaxFinder(), self.get_selected_tier_interval
        )
        self.a = a
        self.curve_layout.addWidget(a)

    def load_eva(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav)"
        )
        if not file_path:
            return

        audio_data = load_channel(file_path)
        print(audio_data.ndim)
        times = np.arange(len(audio_data[0, :])) / 200

        for i, channel in enumerate(audio_data):
            self.curve_layout.addWidget(
                MinMaxAnalyser(
                    f"EVA-{1}",
                    times,
                    channel,
                    MinMaxFinder(),
                    self.get_selected_tier_interval,
                )
            )

    def get_selected_tier_interval(self) -> None:
        if self.textgrid is None:
            return None

        first_tier = self.textgrid.get_tier_by_index(0)
        for t in first_tier.get_elements():
            if not t.get_name():
                continue

            return t

        return None

    def tier_plot_clicked(self, tier_plot, event):
        if not tier_plot.sceneBoundingRect().contains(event.scenePos()):
            return

        mouse_point = tier_plot.vb.mapSceneToView(event.scenePos())

        x, y = mouse_point.x(), mouse_point.y()
        self.add_interval(x, 1, "test", tier_plot, "phones")

    def save_annotations(self):
        if self.textgrid is None:
            # TODO Display pop up error
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save TextGrid File", "", "TextGrid Files (*.TextGrid)"
        )

        if not filepath:
            # TODO Display pop up error
            return

        tgt_textgrid = self.textgrid.to_textgrid()
        tgt.io.write_to_file(tgt_textgrid, filepath, format="long")

    def load_annotations(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open TextGrid File", "", "TextGrid Files (*.TextGrid)"
        )
        if not filepath:
            # TODO Display pop up error
            return

        tgt_textgrid = tgt.io.read_textgrid(filepath)
        self.textgrid = ui_tgt.TextgridTGTConvert().from_textgrid(
            tgt_textgrid, self.a.plot_widget
        )

        self.curve_layout.addWidget(self.textgrid)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = AudioAnalyzer()
    ex.show()
    sys.exit(app.exec_())
