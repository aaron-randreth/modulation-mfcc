import sys
from math import sqrt

import scipy
import numpy as np
from librosa import feature as lf
from PyQt5.QtWidgets import QCheckBox, QTableWidget, QTableWidgetItem, QComboBox
from PyQt5.QtWidgets import QCheckBox
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
def calc_formants(sound: parselmouth.Sound, start_time: float, end_time: float) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    formants = sound.to_formant_burg()

    time_values = formants.ts()
    formant_values = {i: {time: formants.get_value_at_time(formant_number=i, time=time) for time in time_values} for i in range(1, 4)}

    preserved_formants = {i: {time: formant_values[i][time] for time in time_values if start_time <= time <= end_time} for i in range(1, 4)}

    interpolated_formants = {}
    smoothed_formants = {}
    resampled_formants = {}

    for i in range(1, 4):
        interp_func = scipy.interpolate.interp1d(list(preserved_formants[i].keys()), list(preserved_formants[i].values()), kind='linear')
        time_values_interp = np.linspace(min(preserved_formants[i].keys()), max(preserved_formants[i].keys()), num=1000)
        interpolated_formants[i] = interp_func(time_values_interp)
        smoothed_formants[i] = scipy.signal.savgol_filter(interpolated_formants[i], window_length=101, polyorder=1)
        new_time_values = np.arange(start_time, end_time, 1/200.0)
        interp_func_resampled = scipy.interpolate.interp1d(time_values_interp, smoothed_formants[i], kind='linear', fill_value="extrapolate")
        resampled_formants[i] = interp_func_resampled(new_time_values)

    return new_time_values, resampled_formants[1], resampled_formants[2], resampled_formants[3]

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

    def find_in_interval(
        self,
        times: list[float],
        values: list[float],
        interval: tuple[float, float]
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:
        start, end = interval

        interval_times = []
        interval_values = []

        for time, value in zip(times, values):
            in_interval: bool = start <= time and time <= end

            if not in_interval:
                continue

            interval_times.append(time)
            interval_values.append(value)

        return np.array(interval_times), np.array(interval_values)

    def analyse_minimum(self, x, y, interval):
        if interval is None:
            print("No interval specified.")
            return [], []

        interval_times, interval_values = self.find_in_interval(x, y, interval)
        
        min_peaks, _ = scipy.signal.find_peaks(-interval_values)
        if len(min_peaks) == 0:
            return [], []
        
        min_times = interval_times[min_peaks]
        min_values = interval_values[min_peaks]
        
        return min_times, min_values

    def analyse_maximum(self, x, y, interval):
        if interval is None:
            print("No interval specified.")
            return [], []

        interval_times, interval_values = self.find_in_interval(x, y, interval)
        
        max_peaks, _ = scipy.signal.find_peaks(interval_values)
        if len(max_peaks) == 0:
            return [], []
        
        max_times = interval_times[max_peaks]
        max_values = interval_values[max_peaks]
        
        return max_times, max_values



class MinMaxAnalyser(QWidget):
    name: str

    extremum: MinMaxFinder
    toolbar: QToolBar
    plot_widget: pg.PlotWidget

    max_points: pg.ScatterPlotItem
    min_points: pg.ScatterPlotItem

    def __init__(self, name: str, x, y, extremum: MinMaxFinder, get_interval_func, secondary_viewbox=None) -> None:
        super().__init__()
        self.name = name
        self.x = x
        self.y = y
        self.extremum = extremum
        self.get_interval = get_interval_func  
        self.secondary_viewbox = secondary_viewbox

        self.plot_widget = None  
        
        self.visibility_checkbox = QCheckBox(f"Toggle visibility for {name}")
        self.visibility_checkbox.setChecked(True) 

        self.__init_ui()

        self.max_points = pg.ScatterPlotItem(pen=pg.mkPen("g"), brush=pg.mkBrush("b"))
        self.min_points = pg.ScatterPlotItem(pen=pg.mkPen("r"), brush=pg.mkBrush("r"))

        self.plot_widget.addItem(self.max_points)
        self.plot_widget.addItem(self.min_points)
        self.max_points.hide()
        self.min_points.hide()

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

    def update_plot(self, x, y):
        self.curve.setData(x, y)

    def toggle_formant(self, formant_number):
        if not hasattr(self, 'formant_analyzers') or len(self.formant_analyzers) < formant_number:
            return

        analyzer = self.formant_analyzers[formant_number - 1]
        if analyzer.isVisible():
            analyzer.setParent(None)
            self.curve_layout.removeWidget(analyzer)
            analyzer.visibility_checkbox.setParent(None)
            self.curve_layout.removeWidget(analyzer.visibility_checkbox)
        else:
            self.curve_layout.addWidget(analyzer.visibility_checkbox)
            self.curve_layout.addWidget(analyzer)
            self.crosshair.add_display_plot(analyzer.plot_widget)

    def setup_formant_buttons(self):
        self.formant_buttons = []
        for formant_number in [1, 2, 3]:
            button = QPushButton(f'Toggle Formant {formant_number}')
            button.clicked.connect(lambda _, fn=formant_number: self.toggle_formant(fn))
            self.formant_buttons.append(button)
            self.curve_layout.addWidget(button)



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

    def toggleable_action(self, label: str) -> QAction:
        action = QAction(label, self)
        action.setCheckable(True)
        action.setChecked(False)
        return action

    def add_point_on_click(self, clicked_curve: pg.PlotDataItem, event: "MouseClickEvent"):
        if not self.manual_peak_maximum_addition.isChecked() and not self.manual_peak_minimum_addition.isChecked():
            return

        mouse_point = event.pos()
        x, y = mouse_point.x(), mouse_point.y()

        if self.manual_peak_maximum_addition.isChecked():
            points_x, points_y = self.max_points.getData()
            self.max_points.setData(np.append(points_x, x), np.append(points_y, y))
            self.max_points.show()  

        elif self.manual_peak_minimum_addition.isChecked():
            points_x, points_y = self.min_points.getData()
            self.min_points.setData(np.append(points_x, x), np.append(points_y, y))
            self.min_points.show() 

    def remove_peak_on_click(self, clicked_scatter_plot, clicked_points):
        if not self.manual_peak_removal.isChecked():
            return

        points_data = clicked_scatter_plot.getData()
        points_x, points_y = points_data[0], points_data[1]

        to_remove = []
        for point in clicked_points:
            pos = point.pos()

            for i, (px, py) in enumerate(zip(points_x, points_y)):
                if (px == pos.x()) and (py == pos.y()):
                    to_remove.append(i)

        points_x = np.delete(points_x, to_remove)
        points_y = np.delete(points_y, to_remove)

        clicked_scatter_plot.setData(points_x, points_y)

        self.plot_widget.unsetCursor()

    def compute_min(self):
        interval = self.get_interval()
        if interval is None:
            print("No region selected.")
            return

        start, end = interval
        x_min, y_min = self.extremum.analyse_minimum(self.x, self.y, (start, end))  

        if len(x_min) == 0 or len(y_min) == 0:
            print(f"No minimums found within the selected region ({start}, {end}).")
            return

        min_points = pg.ScatterPlotItem(
            name="min",
            x=x_min,
            y=y_min,
            symbol="o",
            size=10,
            pen=pg.mkPen("r"),
            brush=pg.mkBrush("r"),
        )

        self.min_points.addPoints(x=x_min, y=y_min)
        self.min_points.sigClicked.connect(self.remove_peak_on_click)
        if self.secondary_viewbox:
            self.secondary_viewbox.addItem(min_points)
        else:
            self.plot_widget.addItem(min_points)

    def compute_max(self):
        interval = self.get_interval()
        if interval is None:
            print("No region selected.")
            return

        start, end = interval
        x_max, y_max = self.extremum.analyse_maximum(self.x, self.y, (start, end)) 

        if len(x_max) == 0 or len(y_max) == 0:
            print("No maximums found within the selected region.")
            return

        max_points = pg.ScatterPlotItem(
            name="max",
            x=x_max,
            y=y_max,
            symbol="x",
            size=10,
            pen=pg.mkPen("g"),
            brush=pg.mkBrush("b"),
        )

        self.max_points.addPoints(x=x_max, y=y_max)
        self.max_points.sigClicked.connect(self.remove_peak_on_click)
        if self.secondary_viewbox:
            self.secondary_viewbox.addItem(max_points)
        else:
            self.plot_widget.addItem(max_points)

class AudioAnalyzer(QMainWindow):
    textgrid: ui_tiers.TextGrid | None = None
    spectrogram_stacked_widget: QStackedWidget
    spectrogram_loaded: bool = False

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Analyzer")
        self._createMenuBar()
        main_widget = QWidget(self)
        main_layout = QHBoxLayout()  
        left_layout = QVBoxLayout()  
        right_layout = QVBoxLayout() 
        
        self.textgrid_visibility_checkboxes = {}
        self.spectrogram_widget = None
        self.spectrogram_stacked_widget = QStackedWidget()
        curve_area, self.curve_layout = self.curves_container()
        self.spectrogram_stacked_widget.addWidget(curve_area)
        self.file_path = ""

        left_layout.addWidget(self.spectrogram_stacked_widget)
        left_layout.addWidget(self.button_container())
        self.central_plots = []

        self.selected_region = pg.LinearRegionItem()
        self.selected_region.setZValue(10)
        self.selected_region.hide()
        self.textgrid_visibility_checkboxes = {}

        self.panels = []
        for i in range(1, 5):
            panel = QWidget(self)
            panel_layout = QVBoxLayout(panel)
            panel_title = QLabel(f"Panel {i}")
            panel_layout.addWidget(panel_title)
            empty_plot = pg.PlotWidget()
            panel_layout.addWidget(empty_plot)
            left_layout.addWidget(panel)
            self.panels.append((panel, empty_plot))

        for i in range(1, 4):
            self.panels[i][1].setXLink(self.panels[0][1])


        self.dashboard = QTableWidget(4, 5, self)   
        self.dashboard.setHorizontalHeaderLabels(["Acoustique", "EMA", "Couleur", "Panel", "Visibility"])

        for row in range(4):
            combo_box = QComboBox()
            combo_box.addItems(["Option 1", "Mfcc 2", "formant "])
            combo_box.currentIndexChanged.connect(lambda index, row=row: self.update_panel(row, index))
            self.dashboard.setCellWidget(row, 0, combo_box)

            for col in range(1, 3):
                button = QPushButton(f"Button {row+1},{col+1}")
                self.dashboard.setCellWidget(row, col, button)

            panel_combo_box = QComboBox()
            panel_combo_box.addItems(["1", "2", "3", "4"])
            self.dashboard.setCellWidget(row, 3, panel_combo_box)

            visibility_checkbox = QCheckBox()
            visibility_checkbox.setChecked(True) 
            visibility_checkbox.stateChanged.connect(lambda state, row=row: self.toggle_visibility(row, state))
            self.dashboard.setCellWidget(row, 4, visibility_checkbox)
        
        right_layout.addWidget(self.dashboard)

        self.analysis_toolbar = QToolBar(self)
        self.analysis_panel_combo_box = QComboBox()
        self.analysis_panel_combo_box.addItems(["1", "2", "3", "4"])
        self.analysis_toolbar.addWidget(QLabel("Select Panel:"))
        self.analysis_toolbar.addWidget(self.analysis_panel_combo_box)

        self.do_maximum_analysis = QAction("Analyse max", self)
        self.do_minimum_analysis = QAction("Analyse min", self)
        self.do_maximum_analysis.triggered.connect(self.compute_max)
        self.do_minimum_analysis.triggered.connect(self.compute_min)
        self.analysis_toolbar.addAction(self.do_maximum_analysis)
        self.analysis_toolbar.addAction(self.do_minimum_analysis)
        
        right_layout.addWidget(self.analysis_toolbar)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.selected_region = pg.LinearRegionItem()
        self.selected_region.setZValue(10)
        self.selected_region.hide()
        self.textgrid_visibility_checkboxes = {}

    def fix_y_axis_limits(self, plot_widget):
        view_box = plot_widget.getPlotItem().getViewBox()
        y_range = view_box.viewRange()[1]
        view_box.setLimits(yMin=y_range[0], yMax=y_range[1])

        if hasattr(plot_widget, 'secondary_viewbox'):
            secondary_y_range = plot_widget.secondary_viewbox.viewRange()[1]
            plot_widget.secondary_viewbox.setLimits(yMin=secondary_y_range[0], yMax=secondary_y_range[1])

    def add_selection_tool(self, plot_widget):
        plot_widget.addItem(self.selected_region)
        self.selected_region.show()
        self.selected_region.sigRegionChanged.connect(self.on_region_changed)
        
    def save_y_ranges(self, panel):
        self.y_range_main_before = panel.viewRange()[1]
        if hasattr(panel, 'secondary_viewbox'):
            self.y_range_secondary_before = panel.secondary_viewbox.viewRange()[1]

    def restore_y_ranges(self, panel):
        if hasattr(self, 'y_range_main_before'):
            panel.setYRange(self.y_range_main_before[0], self.y_range_main_before[1])
        if hasattr(self, 'y_range_secondary_before') and hasattr(panel, 'secondary_viewbox'):
            panel.secondary_viewbox.setYRange(self.y_range_secondary_before[0], self.y_range_secondary_before[1])

    def on_region_changed(self):
        region = self.selected_region.getRegion()
        print("Selected region from", region[0], "to", region[1])

    def update_panel(self, row, index):
        panel_combo_box = self.dashboard.cellWidget(row, 3)
        selected_panel = int(panel_combo_box.currentText()) - 1

        if index == 1:  # mfcc
            audio_data = load_channel(self.file_path)
            x_mfccs, y_mfccs = get_MFCCS_change(audio_data)

            panel = self.panels[selected_panel][1]
            mfcc_curve = panel.plot(x_mfccs, y_mfccs, pen='r', clear=False)

            if not hasattr(panel, 'plot_items'):
                panel.plot_items = {}
            panel.plot_items[row] = [mfcc_curve]

        elif index == 2:  # Formant
            start, end = self.get_selected_region_interval()
            if start is None or end is None:
                return  

            f_times, f1_values, f2_values, f3_values = calc_formants(parselmouth.Sound(self.file_path), start, end)

            panel = self.panels[selected_panel][1]

            if not hasattr(panel, 'secondary_viewbox'):
                panel.secondary_viewbox = pg.ViewBox()
                panel.getPlotItem().scene().addItem(panel.secondary_viewbox)
                panel.getPlotItem().getAxis('right').linkToView(panel.secondary_viewbox)
                panel.secondary_viewbox.setXLink(panel)
                panel.getPlotItem().getViewBox().sigResized.connect(
                    lambda: panel.secondary_viewbox.setGeometry(panel.getPlotItem().getViewBox().sceneBoundingRect()))

            formant_curve_f1 = pg.PlotDataItem(f_times, f1_values, pen='b')
            formant_curve_f2 = pg.PlotDataItem(f_times, f2_values, pen='g')
            formant_curve_f3 = pg.PlotDataItem(f_times, f3_values, pen='m')

            panel.secondary_viewbox.addItem(formant_curve_f1)
            panel.secondary_viewbox.addItem(formant_curve_f2)
            panel.secondary_viewbox.addItem(formant_curve_f3)

            if not hasattr(panel, 'plot_items'):
                panel.plot_items = {}
            panel.plot_items[row] = [formant_curve_f1, formant_curve_f2, formant_curve_f3]

            panel.getPlotItem().showAxis('right')
            panel.getPlotItem().getAxis('right').setLabel('Formants')

    def compute_max(self):
        selected_panel = int(self.analysis_panel_combo_box.currentText()) - 1
        panel = self.panels[selected_panel][1]

        if not hasattr(panel, 'plot_items'):
            return

        for plot_item in panel.plot_items.values():
            for curve in plot_item:
                x, y = curve.xData, curve.yData
                max_finder = MinMaxFinder()
                x_max, y_max = max_finder.analyse_maximum(x, y, self.get_selected_region_interval())

                if len(x_max) == 0 or len(y_max) == 0:
                    print("No maximums found within the selected region.")
                    continue

                max_points = pg.ScatterPlotItem(
                    x=x_max,
                    y=y_max,
                    symbol="x",
                    size=10,
                    pen=pg.mkPen("g"),
                    brush=pg.mkBrush("b"),
                )
                if hasattr(panel, 'secondary_viewbox') and curve.getViewBox() is panel.secondary_viewbox:
                    panel.secondary_viewbox.addItem(max_points)
                else:
                    panel.addItem(max_points)

    def compute_min(self):
        selected_panel = int(self.analysis_panel_combo_box.currentText()) - 1
        panel = self.panels[selected_panel][1]

        if not hasattr(panel, 'plot_items'):
            return

        for plot_item in panel.plot_items.values():
            for curve in plot_item:
                x, y = curve.xData, curve.yData
                min_finder = MinMaxFinder()
                x_min, y_min = min_finder.analyse_minimum(x, y, self.get_selected_region_interval())

                if len(x_min) == 0 or len(y_min) == 0:
                    print(f"No minimums found within the selected region.")
                    continue

                min_points = pg.ScatterPlotItem(
                    x=x_min,
                    y=y_min,
                    symbol="o",
                    size=10,
                    pen=pg.mkPen("r"),
                    brush=pg.mkBrush("r"),
                )
                if hasattr(panel, 'secondary_viewbox') and curve.getViewBox() is panel.secondary_viewbox:
                    panel.secondary_viewbox.addItem(min_points)
                else:
                    panel.addItem(min_points)

    def toggle_visibility(self, row, state):
        panel_combo_box = self.dashboard.cellWidget(row, 3)
        selected_panel = int(panel_combo_box.currentText()) - 1
        panel = self.panels[selected_panel][1]

        if hasattr(panel, 'plot_items') and row in panel.plot_items:
            for plot_item in panel.plot_items[row]:
                if state == Qt.Checked:
                    plot_item.show()
                else:
                    plot_item.hide()

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

        sound_widget = create_plot_widget(snd.timestamps, snd.amplitudes[0])
        self.add_selection_tool(sound_widget)  
        self.curve_layout.addWidget(sound_widget)

        spectrogram_widget = None
        if self.spectrogram_loaded:
            self.spectrogram_widget = specto.create_spectrogram_plot(
                self.spc.frequencies, self.spc.timestamps, self.spc.data_matrix
            )
            self.spectrogram_stacked_widget.addWidget(self.spectrogram_widget)
            self.spectrogram_stacked_widget.setCurrentWidget(self.spectrogram_widget)

        audio_data = load_channel(file_path)
        x_mfccs, y_mfccs = get_MFCCS_change(audio_data)

        self.add_selection_tool(sound_widget) 
        a = MinMaxAnalyser(
            "Mfcc", x_mfccs, y_mfccs, MinMaxFinder(), self.get_selected_region_interval
        )
        self.crosshair = Crosshair([sound_widget])

        if self.spectrogram_widget is not None:
            self.crosshair.add_central_plot(self.spectrogram_widget)

        self.a = a
        self.crosshair.add_display_plot(a.plot_widget)

        self.curve_layout.addWidget(a.visibility_checkbox)  
        self.curve_layout.addWidget(a)

    def toggle_spectrogram(self):
        if self.spectrogram_loaded:
            self.spectrogram_widget.setParent(None)
            self.spectrogram_widget = None
            self.spectrogram_loaded = False
            for analyzer in self.formant_analyzers:
                analyzer.setParent(None)
                self.curve_layout.removeWidget(analyzer)
                analyzer.visibility_checkbox.setParent(None)
                self.curve_layout.removeWidget(analyzer.visibility_checkbox)
            return

        if not hasattr(self, 'spc') or not self.spc:
            return

        self.spectrogram_widget = specto.create_spectrogram_plot(
            self.spc.frequencies, self.spc.timestamps, self.spc.data_matrix
        )
        self.curve_layout.insertWidget(1, self.spectrogram_widget)
        self.spectrogram_loaded = True
        self.crosshair.add_central_plot(self.spectrogram_widget)



    def link_plots(self):
        if not self.central_plots:
            return
        for p in self.central_plots:
            p.setXLink(self.central_plots[0])

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
                self.get_selected_region_interval,
                secondary_viewbox=None  
            )

            self.curve_layout.addWidget(a)
            self.curve_layout.addWidget(a.visibility_checkbox)
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

    def get_selected_region_interval(self):
        if self.selected_region.isVisible():
            start, end = self.selected_region.getRegion()
            return (start, end)
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
            # TODO:
            return

        tgt_textgrid = tgt.io.read_textgrid(filepath)
        self.textgrid = ui_tgt.TextgridTGTConvert().from_textgrid(
            tgt_textgrid, self.a.plot_widget
        )

        for tier in self.textgrid.get_tiers():
            self.crosshair.add_display_plot(tier)

        self.curve_layout.addWidget(self.textgrid)

        self.display_textgrid_checkbox = QCheckBox("Afficher TextGrid", self)
        self.display_textgrid_checkbox.setChecked(True)
        self.display_textgrid_checkbox.stateChanged.connect(self.toggle_textgrid_display)

        self.curve_layout.addWidget(self.display_textgrid_checkbox)

    def toggle_textgrid_display(self, state):
        if state == Qt.Checked:
            self.textgrid.show()
        else:
            self.textgrid.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = AudioAnalyzer()
    ex.show()
    sys.exit(app.exec_())
