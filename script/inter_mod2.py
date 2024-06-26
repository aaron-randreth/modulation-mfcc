import sys
from math import sqrt

import scipy
import numpy as np
from librosa import feature as lf

from PyQt5.QtWidgets import (
    QMenu,
    QStackedWidget,
    QMenuBar,
    QToolBar,
    QAction,
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
    QListWidget,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox
)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QWindow

import pyqtgraph as pg
import parselmouth
import tgt

from praat_py_ui import (
    tiers as ui_tiers,
    textgridtools as ui_tgt,
    spectrogram as specto,
    parselmouth_calc as calc,
)
from datasources.mfcc import load_channel, get_MFCCS_change
from scrollable_window import Info, InfoBox, Output

pg.setConfigOptions(foreground="black", background="w")

def calc_formant(sound: parselmouth.Sound, start_time: float, end_time: float, formant_number: int) -> tuple[list[float], list[float]]:
    formants = sound.to_formant_burg()

    time_values = formants.ts()
    formant1_dict = {time:
                     formants.get_value_at_time(formant_number=formant_number, time=time) for time in time_values}

    preserved_formant1_dict = {time: formant1_dict[time] for time in time_values if start_time <= time <= end_time}

    interp_func1 = scipy.interpolate.interp1d(list(preserved_formant1_dict.keys()), list(preserved_formant1_dict.values()), kind='linear')
    time_values_interp1 = np.linspace(min(preserved_formant1_dict.keys()), max(preserved_formant1_dict.keys()), num=1000)
    interpolated_formants = interp_func1(time_values_interp1)

    smoothed_curve1 = scipy.signal.savgol_filter(interpolated_formants, window_length=101, polyorder=1)

    return time_values_interp1, smoothed_curve1

def create_plot_widget(x, y):
    plot = pg.PlotWidget()
    plot.plot(x=x, y=y)
    return plot

class SelectableListDialog(QDialog):

    def __init__(self, num_items: int, format_string: str):
        super().__init__()

        self.setWindowTitle('Selectable List')

        self.item_labels = [format_string.format(i) for i in range(num_items)]

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)

        self.list_widget.addItems(self.item_labels)

        self.dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.dialog_buttons.accepted.connect(self.accept)
        self.dialog_buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.list_widget)
        layout.addWidget(self.dialog_buttons)
        self.setLayout(layout)

    def get_selected_indices(self) -> list[int]:
        selected_texts = [item.text() for item in self.list_widget.selectedItems()]
        return [self.item_labels.index(text) for text in selected_texts]

class Crosshair:

    def __init__(self, central_plots) -> None:
        self.central_plots = []
        self.display_plots = []
        self.crosshair_lines = []

        for plot in central_plots:
            self.add_central_plot(plot)

        self.link_plots()

    @property
    def plots(self):
        return [*self.central_plots, *self.display_plots]

    def link_plots(self):
        for p in self.plots:
            p.setXLink(self.central_plots[0])

    def add_central_plot(self, central_plot) -> None:
        line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(style=Qt.DashLine, color="r")
        )

        self.crosshair_lines.append(line)
        self.central_plots.append(central_plot)

        central_plot.addItem(line, ignoreBounds=True)
        central_plot.scene().sigMouseMoved.connect(self.move_crosshair)

        self.link_plots()

    def add_display_plot(self, display_plot) -> None:
        line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(style=Qt.DashLine, color="b")
        )

        self.crosshair_lines.append(line)
        self.display_plots.append(display_plot)

        display_plot.addItem(line, ignoreBounds=True)

        self.link_plots()

    def move_crosshair(self, event):
        mousePoint = None
        pos = event

        for p in self.central_plots:
            if p.sceneBoundingRect().contains(pos):
                mousePoint = p.getPlotItem().vb.mapSceneToView(pos)

        if mousePoint is None:
            return

        for l in self.crosshair_lines:
            l.setPos(mousePoint.x())

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
        min_peaks, _ = scipy.signal.find_peaks(
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

        min_peaks, _ = scipy.signal.find_peaks(-interval_values)

        initial_peaks, _ = scipy.signal.find_peaks(interval_values)

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

        peaks, _ = scipy.signal.find_peaks(
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
    spectrogram_stacked_widget: QStackedWidget
    spectrogram_loaded: bool = False

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Analyzer")
        self._createMenuBar()
        main_widget = QWidget(self)
        layout = QHBoxLayout()

        self.spectrogram_widget = None
        self.spectrogram_stacked_widget = QStackedWidget()
        curve_area, self.curve_layout = self.curves_container()
        self.spectrogram_stacked_widget.addWidget(curve_area)
        self.file_path = ""

        layout.addWidget(self.spectrogram_stacked_widget)
        layout.addWidget(self.button_container())

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
    def _createMenuBar(self):
        menuBar = QMenuBar(self)
        self.setMenuBar(menuBar)
        
        # Menu "File"
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        
        # Actions
        loadAudioAction = QAction("Load Audio File", self)
        loadAudioAction.triggered.connect(self.load_audio)
        fileMenu.addAction(loadAudioAction)
        
        loadEVAAction = QAction("Load EVA File", self)
        loadEVAAction.triggered.connect(self.load_eva)
        fileMenu.addAction(loadEVAAction)
        
        loadAnnotationAction = QAction("Load Textgrid Annotation", self)
        loadAnnotationAction.triggered.connect(self.load_annotations)
        fileMenu.addAction(loadAnnotationAction)
        
        saveAnnotationAction = QAction("Save TextGrid Annotation", self)
        saveAnnotationAction.triggered.connect(self.save_annotations)
        fileMenu.addAction(saveAnnotationAction)

        # Menu "Affichage"
        editMenu = menuBar.addMenu("&Affichage")
        
        # Action Toggle Spectrogram
        self.toggle_spectrogram_action = QAction("Toggle Spectrogram", self)
        self.toggle_spectrogram_action.setChecked(False)
        self.toggle_spectrogram_action.setCheckable(True)

        self.toggle_spectrogram_action.triggered.connect(self.toggle_spectrogram)
        editMenu.addAction(self.toggle_spectrogram_action)
        
        # Menu "Help"
        helpMenu = menuBar.addMenu("&Help")

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


        button_parent.setLayout(layout)

        return button_parent


    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav)"
        )

        if not file_path:
            return

        self.file_path = file_path
        sound_data = calc.Parselmouth(file_path)
        snd = sound_data.get_sound()
        self.spc = sound_data.get_spectrogram()

        # Ajouter le tracé audio au QStackedWidget
        sound_widget = create_plot_widget(snd.timestamps, snd.amplitudes[0])
        self.curve_layout.addWidget(sound_widget)

        spectrogram_widget = None
        # Charger initialement le spectrogramme si l'état est activé
        if self.spectrogram_loaded:
            self.spectrogram_widget = specto.create_spectrogram_plot(
                self.spc.frequencies, self.spc.timestamps, self.spc.data_matrix
            )
            self.spectrogram_stacked_widget.addWidget(self.spectrogram_widget)
            self.spectrogram_stacked_widget.setCurrentWidget(self.spectrogram_widget)


        audio_data = load_channel(file_path)
        x_mfccs, y_mfccs = get_MFCCS_change(audio_data)

        a = MinMaxAnalyser(
            "Mfcc", x_mfccs, y_mfccs, MinMaxFinder(), self.get_selected_tier_interval
        )

        self.crosshair = Crosshair([sound_widget])

        if self.spectrogram_widget is not None:
            self.crosshair.add_central_plot(self.spectrogram_widget)

        self.a = a
        self.crosshair.add_display_plot(a.plot_widget)
        self.curve_layout.addWidget(a)

    def toggle_spectrogram(self):
        if self.spectrogram_loaded:
            # Si le spectrogramme est déjà chargé, le masquer
            self.spectrogram_widget.setParent(None)
            self.spectrogram_widget = None
            self.spectrogram_loaded = False
            return

        if not hasattr(self, 'spc') or not self.spc:
            return

        self.spectrogram_widget = specto.create_spectrogram_plot(
            self.spc.frequencies, self.spc.timestamps, self.spc.data_matrix
        )
        self.curve_layout.insertWidget(1, self.spectrogram_widget)

        self.spectrogram_loaded = True
        self.crosshair.add_central_plot(self.spectrogram_widget)

        selected_tier_interval = self.get_selected_tier_interval()
        if selected_tier_interval is None:
            return

        start = float(selected_tier_interval.start_time)
        end = float(selected_tier_interval.end_time)

        f1_times, f1_values = calc_formant(parselmouth.Sound(self.file_path), start, end, 1)
        self.spectrogram_widget.plot(f1_times, f1_values)

        f2_times, f2_values = calc_formant(parselmouth.Sound(self.file_path), start, end, 2)
        self.spectrogram_widget.plot(f2_times, f2_values)

        f3_times, f3_values = calc_formant(parselmouth.Sound(self.file_path), start, end, 3)
        self.spectrogram_widget.plot(f3_times, f3_values)

    def load_eva(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav)"
        )
        if not file_path:
            return

        audio_data = load_channel(file_path)
        times = np.arange(len(audio_data[0, :])) / 200

        channel_nb = len(audio_data)

        selection = SelectableListDialog(channel_nb, "Channel {}")

        if selection.exec_() != QDialog.Accepted:
            return

        for i in selection.get_selected_indices():
            channel = audio_data[i]

            a = MinMaxAnalyser(
                    f"EVA-{1}",
                    times,
                    channel,
                    MinMaxFinder(),
                    self.get_selected_tier_interval,
            )

            self.curve_layout.addWidget(a)
            self.crosshair.add_display_plot(a.plot_widget)

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

        for tier in self.textgrid.get_tiers():
            self.crosshair.add_display_plot(tier)

        self.curve_layout.addWidget(self.textgrid)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = AudioAnalyzer()
    ex.show()
    sys.exit(app.exec_())
