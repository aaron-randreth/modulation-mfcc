import sys
from math import sqrt
import scipy
import numpy as np
import csv
from librosa import feature as lf
from PyQt5.QtWidgets import QCheckBox, QTableWidget, QTableWidgetItem, QComboBox, QMenu, QStackedWidget, QMenuBar, QToolBar, QAction, QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QGridLayout, QLabel, QScrollArea, QListWidget, QAbstractItemView, QDialog, QDialogButtonBox, QColorDialog, QInputDialog
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import parselmouth
import tgt
import xarray as xr
from praat_py_ui import tiers as ui_tiers, textgridtools as ui_tgt, spectrogram as specto, parselmouth_calc as calc
from datasources.mfcc import load_channel, get_MFCCS_change
from scrollable_window import Info, InfoBox, Output

pg.setConfigOptions(foreground="black", background="w")

def calc_formants(sound: parselmouth.Sound, start_time: float, end_time: float):
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

def create_plot_widget(x, y, color='r'):
    plot = pg.PlotWidget()
    plot.plot(x=x, y=y, pen=color)
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

    def add_panel_plot(self, panel_plot):
        line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(style=Qt.DashLine, color="g")
        )
        self.crosshair_lines.append(line)
        self.central_plots.append(panel_plot)
        panel_plot.addItem(line, ignoreBounds=True)
        panel_plot.scene().sigMouseMoved.connect(self.move_crosshair)
        self.link_plots()

class MinMaxFinder:
    def find_in_interval(self, times: list[float], values: list[float], interval: tuple[float, float]) -> tuple[np.ndarray[float], np.ndarray[float]]:
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
    def __init__(self, name: str, x, y, extremum: MinMaxFinder, get_interval_func, color='r', secondary_viewbox=None, tertiary_viewbox=None) -> None:
        super().__init__()
        self.name = name
        self.x = x
        self.y = y
        self.extremum = extremum
        self.get_interval = get_interval_func
        self.color = color
        self.secondary_viewbox = secondary_viewbox
        self.tertiary_viewbox = tertiary_viewbox
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
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMaximumHeight(400)
        
        # Utilisez ScatterPlotItem pour rendre les points cliquables
        self.curve = pg.ScatterPlotItem(x=self.x, y=self.y, pen=self.color, brush=pg.mkBrush(self.color))
        self.curve.sigClicked.connect(self.add_point_on_click)
        self.plot_widget.addItem(self.curve)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def update_plot(self, x, y):
        self.curve.setData(x=x, y=y)

    def add_point_on_click(self, plot_item, event):
        pos = event[0].scenePos()
        if not plot_item.getViewBox().sceneBoundingRect().contains(pos):
            return

        mouse_point = plot_item.getViewBox().mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        print(f"Clicked at x: {x}, y: {y}")  # Affiche les coordonnées du clic

        if self.parent().manual_peak_maximum_addition.isChecked():
            points_x, points_y = self.max_points.getData()
            closest_index = np.argmin(np.abs(points_x - x))
            points_x = np.insert(points_x, closest_index, x)
            points_y = np.insert(points_y, closest_index, y)
            self.max_points.setData(points_x, points_y)
            self.max_points.show()
            print("Added max point")
        elif self.parent().manual_peak_minimum_addition.isChecked():
            points_x, points_y = self.min_points.getData()
            closest_index = np.argmin(np.abs(points_x - x))
            points_x = np.insert(points_x, closest_index, x)
            points_y = np.insert(points_y, closest_index, y)
            self.min_points.setData(points_x, points_y)
            self.min_points.show()
            print("Added min point")
        elif self.parent().manual_peak_removal.isChecked():
            points_x, points_y = self.max_points.getData()
            distances = np.sqrt((points_x - x) ** 2 + (points_y - y) ** 2)
            closest_index = np.argmin(distances)
            points_x = np.delete(points_x, closest_index)
            points_y = np.delete(points_y, closest_index)
            self.max_points.setData(points_x, points_y)
            self.max_points.show()
            print("Removed point")


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
        panels_layout = QVBoxLayout()
        self.panels = []
        for i in range(1, 5):
            panel = QWidget(self)
            panel_layout = QVBoxLayout(panel)
            panel_title = QLabel(f"Panel {i}")
            panel_layout.addWidget(panel_title)
            empty_plot = pg.PlotWidget()
            empty_plot.setMaximumHeight(165)
            panel_layout.addWidget(empty_plot)
            panels_layout.addWidget(panel)
            self.panels.append((panel, empty_plot))
        for i in range(1, 4):
            self.panels[i][1].setXLink(self.panels[0][1])
        left_layout.addWidget(curve_area)
        left_layout.addLayout(panels_layout)
        self.file_path = ""
        left_layout.addWidget(self.button_container())
        self.central_plots = []
        self.selected_region = pg.LinearRegionItem()
        self.selected_region.setZValue(10)
        self.selected_region.hide()
        self.textgrid_visibility_checkboxes = {}
        self.dashboard = QTableWidget(4, 5, self)
        self.dashboard.setHorizontalHeaderLabels(["Acoustique", "EMA", "Couleur", "Panel", "Visibility"])
        for row in range(4):
            combo_box = QComboBox()
            combo_box.addItems(["Choose", "Modulation cepstrale", "Formant 1", "Formant 2", "Formant 3", "Courbes ema"])
            combo_box.currentIndexChanged.connect(lambda index, row=row: self.update_panel(row, index))
            self.dashboard.setCellWidget(row, 0, combo_box)
            for col in range(1, 2):
                button = QPushButton(f"Button {row+1},{col+1}")
                button.setStyleSheet("QPushButton { background-color: lightblue; border: 1px solid black; padding: 5px; }")
                self.dashboard.setCellWidget(row, col, button)
            panel_combo_box = QComboBox()
            panel_combo_box.addItems(["1", "2", "3", "4"])
            self.dashboard.setCellWidget(row, 3, panel_combo_box)
            visibility_checkbox = QCheckBox()
            visibility_checkbox.setChecked(True) 
            visibility_checkbox.stateChanged.connect(lambda state, row=row: self.toggle_visibility(row, state))
            self.dashboard.setCellWidget(row, 4, visibility_checkbox)
            color_buttons_layout = QHBoxLayout()
            color_names = ["brown", "red", "green", "blue", "orange"]
            for color in color_names:
                color_button = QPushButton()
                color_button.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
                color_button.setFixedSize(20, 20)
                color_button.clicked.connect(lambda _, row=row, color=color: self.change_curve_color(row, color))
                color_buttons_layout.addWidget(color_button)
            color_buttons_widget = QWidget()
            color_buttons_widget.setLayout(color_buttons_layout)
            self.dashboard.setCellWidget(row, 2, color_buttons_widget)
        self.add_row_button = QPushButton("+")
        self.add_row_button.setStyleSheet("QPushButton { background-color: lightgreen; border: 1px solid black; padding: 5px; }")
        self.add_row_button.clicked.connect(self.add_dashboard_row)
        right_layout.addWidget(self.dashboard)
        right_layout.addWidget(self.add_row_button)
        self.analysis_toolbar = QToolBar(self)
        self.analysis_panel_combo_box = QComboBox()
        self.analysis_panel_combo_box.addItems(["1", "2", "3", "4"])
        self.analysis_toolbar.addWidget(QLabel("Select Panel:"))
        self.analysis_toolbar.addWidget(self.analysis_panel_combo_box)
        self.do_maximum_analysis = QAction("Analyse max", self)
        self.do_minimum_analysis = QAction("Analyse min", self)
        self.do_maximum_analysis.triggered.connect(self.compute_max)
        self.do_minimum_analysis.triggered.connect(self.compute_min)
        export_csv_action = QAction("Export to CSV", self)
        export_csv_action.triggered.connect(self.export_to_csv)
        self.analysis_toolbar.addAction(self.do_maximum_analysis)
        self.analysis_toolbar.addAction(self.do_minimum_analysis)
        self.analysis_toolbar.addAction(export_csv_action)
        self.manual_peak_maximum_addition = QAction("Add Max Peak", self)
        self.manual_peak_minimum_addition = QAction("Add Min Peak", self)
        self.manual_peak_removal = QAction("Remove Peak", self)
        self.manual_peak_maximum_addition.setCheckable(True)
        self.manual_peak_minimum_addition.setCheckable(True)
        self.manual_peak_removal.setCheckable(True)
        self.analysis_toolbar.addAction(self.manual_peak_maximum_addition)
        self.analysis_toolbar.addAction(self.manual_peak_minimum_addition)
        self.analysis_toolbar.addAction(self.manual_peak_removal)
        right_layout.addWidget(self.analysis_toolbar)
        right_layout.addStretch(1)  # Ajout d'un espace pour pousser les éléments vers le haut
        right_layout.addWidget(self.create_spectrogram_toggle_button())  # Ajouter le bouton Toggle Spectrogram en bas
        
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.selected_region = pg.LinearRegionItem()
        self.selected_region.setZValue(10)
        self.selected_region.hide()
        self.textgrid_visibility_checkboxes = {}
        self.crosshair = Crosshair([])
        self.connect_panel_clicks()

    def connect_panel_clicks(self):
        for i, (_, plot_widget) in enumerate(self.panels):
            plot_widget.max_points = pg.ScatterPlotItem(pen=pg.mkPen("g"), brush=pg.mkBrush("b"))
            plot_widget.min_points = pg.ScatterPlotItem(pen=pg.mkPen("r"), brush=pg.mkBrush("r"))
            plot_widget.addItem(plot_widget.max_points)
            plot_widget.addItem(plot_widget.min_points)
            plot_widget.scene().sigMouseClicked.connect(lambda event, panel_index=i: self.panel_clicked(event, panel_index))

    def panel_clicked(self, event, panel_index):
        pos = event.scenePos()
        plot_widget = self.panels[panel_index][1]
        if not plot_widget.sceneBoundingRect().contains(pos):
            return

        mouse_point = plot_widget.getPlotItem().vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        print(f"Clicked on panel {panel_index + 1} at x: {x}, y: {y}")

    def add_dashboard_row(self):
        row = self.dashboard.rowCount()
        self.dashboard.insertRow(row)
        combo_box = QComboBox()
        combo_box.addItems(["Choose", "Modulation cepstrale", "Formant 1", "Formant 2", "Formant 3", "Courbes ema"])
        combo_box.currentIndexChanged.connect(lambda index, row=row: self.update_panel(row, index))
        self.dashboard.setCellWidget(row, 0, combo_box)
        button = QPushButton(f"Button {row+1},1")
        button.setStyleSheet("QPushButton { background-color: lightblue; border: 1px solid black; padding: 5px; }")
        self.dashboard.setCellWidget(row, 1, button)
        panel_combo_box = QComboBox()
        panel_combo_box.addItems(["1", "2", "3", "4"])
        self.dashboard.setCellWidget(row, 3, panel_combo_box)
        visibility_checkbox = QCheckBox()
        visibility_checkbox.setChecked(True)
        visibility_checkbox.stateChanged.connect(lambda state, row=row: self.toggle_visibility(row, state))
        self.dashboard.setCellWidget(row, 4, visibility_checkbox)
        color_buttons_layout = QHBoxLayout()
        color_names = ["brown", "red", "green", "blue", "orange"]
        for color in color_names:
            color_button = QPushButton()
            color_button.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
            color_button.setFixedSize(20, 20)
            color_button.clicked.connect(lambda _, row=row, color=color: self.change_curve_color(row, color))
            color_buttons_layout.addWidget(color_button)
        color_buttons_widget = QWidget()
        color_buttons_widget.setLayout(color_buttons_layout)
        self.dashboard.setCellWidget(row, 2, color_buttons_widget)

    def fix_y_axis_limits(self, plot_widget):
        view_box = plot_widget.getPlotItem().getViewBox()
        y_range = view_box.viewRange()[1]
        view_box.setLimits(yMin=y_range[0], yMax=y_range[1])
        if hasattr(plot_widget, 'secondary_viewbox'):
            secondary_y_range = plot_widget.secondary_viewbox.viewRange()[1]
            plot_widget.secondary_viewbox.setLimits(yMin=secondary_y_range[0], yMax=y_range[1])
        if hasattr(plot_widget, 'tertiary_viewbox'):
            tertiary_y_range = plot_widget.tertiary_viewbox.viewRange()[1]
            plot_widget.tertiary_viewbox.setLimits(yMin=tertiary_y_range[0], yMax=y_range[1])

    def add_selection_tool(self, plot_widget):
        plot_widget.addItem(self.selected_region)
        self.selected_region.show()
        self.selected_region.sigRegionChanged.connect(self.on_region_changed)
        
    def save_y_ranges(self, panel):
        self.y_range_main_before = panel.viewRange()[1]
        if hasattr(panel, 'secondary_viewbox'):
            self.y_range_secondary_before = panel.secondary_viewbox.viewRange()[1]
        if hasattr(panel, 'tertiary_viewbox'):
            self.y_range_tertiary_before = panel.tertiary_viewbox.viewRange()[1]

    def restore_y_ranges(self, panel):
        if hasattr(self, 'y_range_main_before'):
            panel.setYRange(self.y_range_main_before[0], self.y_range_main_before[1])
        if hasattr(self, 'y_range_secondary_before') and hasattr(panel, 'secondary_viewbox'):
            panel.secondary_viewbox.setYRange(self.y_range_secondary_before[0], self.y_range_secondary_before[1])
        if hasattr(self, 'y_range_tertiary_before') and hasattr(panel, 'tertiary_viewbox'):
            panel.tertiary_viewbox.setYRange(self.y_range_tertiary_before[0], self.y_range_tertiary_before[1])

    def on_region_changed(self):
        region = self.selected_region.getRegion()
        print("Selected region from", region[0], "to", region[1])
    def update_panel(self, row, index):
        panel_combo_box = self.dashboard.cellWidget(row, 3)
        selected_panel = int(panel_combo_box.currentText()) - 1
        panel = self.panels[selected_panel][1]

        right_axis = None

        if index == 1:  # MFCC
            audio_data = load_channel(self.file_path)
            x_mfccs, y_mfccs = get_MFCCS_change(audio_data)
            mfcc_curve = panel.plot(x_mfccs, y_mfccs, pen='r', clear=False)
            mfcc_curve.sigClicked.connect(lambda plot_item, points, event: self.add_point_on_click(plot_item, event))
            print("MFCC curve connected to click event")
            if not hasattr(panel, 'plot_items'):
                panel.plot_items = {}
            panel.plot_items[row] = [mfcc_curve]
            panel.getPlotItem().getViewBox().setMouseEnabled(y=False)
            panel.getPlotItem().getAxis('left').setLabel('MFCC')
            
        elif index in [2, 3, 4]:  # Formants
            formant_num = index - 1
            start, end = self.get_selected_region_interval()
            if start is None or end is None:
                return
            f_times, f1_values, f2_values, f3_values = calc_formants(parselmouth.Sound(self.file_path), start, end)
            if formant_num == 1:
                formant_values = f1_values
                formant_label = 'Formant 1'
            elif formant_num == 2:
                formant_values = f2_values
                formant_label = 'Formant 2'
            else:
                formant_values = f3_values
                formant_label = 'Formant 3'

            if not hasattr(panel, 'secondary_viewbox'):
                panel.secondary_viewbox = pg.ViewBox()
                panel.getPlotItem().scene().addItem(panel.secondary_viewbox)
                right_axis = pg.AxisItem('right')
                panel.getPlotItem().layout.addItem(right_axis, 2, 3)
                right_axis.linkToView(panel.secondary_viewbox)
                panel.secondary_viewbox.setXLink(panel)
                panel.getPlotItem().getViewBox().sigResized.connect(lambda: panel.secondary_viewbox.setGeometry(panel.getPlotItem().getViewBox().sceneBoundingRect()))

            formant_curve = pg.PlotDataItem(f_times, formant_values, pen='b')
            formant_curve.sigClicked.connect(lambda plot_item, points, event: self.add_point_on_click(plot_item, event))
            print(f"Formant {formant_num} curve connected to click event")
            panel.secondary_viewbox.addItem(formant_curve)
            if not hasattr(panel, 'plot_items'):
                panel.plot_items = {}
            panel.plot_items[row] = [formant_curve]
            panel.getPlotItem().showAxis('right')
            panel.getPlotItem().getAxis('right').setLabel(formant_label)

        elif index == 5:  # courbes ema
            file_path, _ = QFileDialog.getOpenFileName(self, "Open EMA File", "", "EMA Files (*.pos)")
            if not file_path:
                return
            ema_data = read_AG50x(file_path)
            num_channels = len(ema_data.channels)
            channel_selection_dialog = SelectableListDialog(num_channels, "Channel {}")
            if channel_selection_dialog.exec_() != QDialog.Accepted:
                return
            selected_channels = channel_selection_dialog.get_selected_indices()
            time = ema_data.time.values
            if not hasattr(panel, 'tertiary_viewbox'):
                panel.tertiary_viewbox = pg.ViewBox()
                panel.getPlotItem().scene().addItem(panel.tertiary_viewbox)
                right_axis = pg.AxisItem('right')
                panel.getPlotItem().layout.addItem(right_axis, 2, 3)
                right_axis.linkToView(panel.tertiary_viewbox)
                panel.tertiary_viewbox.setXLink(panel)
                panel.getPlotItem().getViewBox().sigResized.connect(lambda: panel.tertiary_viewbox.setGeometry(panel.getPlotItem().getViewBox().sceneBoundingRect()))
            for channel in selected_channels:
                channel_data = ema_data.sel(channels=channel).sel(dimensions="y").ema.values
                channel_label, ok = QInputDialog.getText(self, "Channel Label", f"Enter label for Channel {channel}")
                if not ok or not channel_label:
                    channel_label = f"EMA Channel {channel}"
                ema_curve = pg.PlotDataItem(time, channel_data, pen=pg.mkPen(width=2))
                panel.tertiary_viewbox.addItem(ema_curve)
                if not hasattr(panel, 'plot_items'):
                    panel.plot_items = {}
                if row not in panel.plot_items:
                    panel.plot_items[row] = []
                panel.plot_items[row].append(ema_curve)

            panel.getPlotItem().showAxis('left')
            panel.getPlotItem().getAxis('left').setLabel(channel_label)
            if right_axis is not None:
                right_axis.setLabel(channel_label)

            # Mise à jour du label dans le tableau
            table_item = self.dashboard.cellWidget(row, 0)
            table_item.setItemText(5, channel_label)

        self.crosshair.add_panel_plot(panel)

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
                max_points = pg.ScatterPlotItem(x=x_max, y=y_max, symbol="x", size=10, pen=pg.mkPen("g"), brush=pg.mkBrush("b"))
                if hasattr(panel, 'secondary_viewbox') and curve.getViewBox() is panel.secondary_viewbox:
                    panel.secondary_viewbox.addItem(max_points)
                elif hasattr(panel, 'tertiary_viewbox') and curve.getViewBox() is panel.tertiary_viewbox:
                    panel.tertiary_viewbox.addItem(max_points)
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
                min_points = pg.ScatterPlotItem(x=x_min, y=y_min, symbol="o", size=10, pen=pg.mkPen("r"), brush=pg.mkBrush("r"))
                if hasattr(panel, 'secondary_viewbox') and curve.getViewBox() is panel.secondary_viewbox:
                    panel.secondary_viewbox.addItem(min_points)
                elif hasattr(panel, 'tertiary_viewbox') and curve.getViewBox() is panel.tertiary_viewbox:
                    panel.tertiary_viewbox.addItem(min_points)
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

    def change_curve_color(self, row, color):
        panel_combo_box = self.dashboard.cellWidget(row, 3)
        selected_panel = int(panel_combo_box.currentText()) - 1
        panel = self.panels[selected_panel][1]
        if hasattr(panel, 'plot_items') and row in panel.plot_items:
            for plot_item in panel.plot_items[row]:
                plot_item.setPen(pg.mkPen(color=color, width=2))
        color_widget = self.dashboard.cellWidget(row, 2)
        for i in range(color_widget.layout().count()):
            button = color_widget.layout().itemAt(i).widget()
            button.setStyleSheet(f"background-color: {color}; border: 1px solid black;")

    def _createMenuBar(self):
        menuBar = QMenuBar(self)
        self.setMenuBar(menuBar)
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
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
        helpMenu = menuBar.addMenu("&Help")

    def create_spectrogram_toggle_button(self):
        self.toggle_spectrogram_button = QPushButton("Mask Spectrogram")
        self.toggle_spectrogram_button.setCheckable(True)
        self.toggle_spectrogram_button.setStyleSheet("QPushButton { background-color: lightcoral; border: 1px solid black; padding: 5px; }")
        self.toggle_spectrogram_button.clicked.connect(self.toggle_spectrogram)
        return self.toggle_spectrogram_button

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
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav)")
        if not file_path:
            return
        self.file_path = file_path
        sound_data = calc.Parselmouth(file_path)
        snd = sound_data.get_sound()
        self.spc = sound_data.get_spectrogram()
        sound_widget = create_plot_widget(snd.timestamps, snd.amplitudes[0], color='k')
        self.add_selection_tool(sound_widget)
        self.curve_layout.addWidget(sound_widget)
        spectrogram_widget = None
        if self.spectrogram_loaded:
            self.spectrogram_widget = specto.create_spectrogram_plot(self.spc.frequencies, self.spc.timestamps, self.spc.data_matrix)
            self.spectrogram_stacked_widget.addWidget(self.spectrogram_widget)
            self.spectrogram_stacked_widget.setCurrentWidget(self.spectrogram_widget)
        audio_data = load_channel(file_path)
        x_mfccs, y_mfccs = get_MFCCS_change(audio_data)
        self.add_selection_tool(sound_widget)
        a = MinMaxAnalyser("Mfcc", x_mfccs, y_mfccs, MinMaxFinder(), self.get_selected_region_interval)
        self.crosshair = Crosshair([sound_widget])
        if self.spectrogram_widget is not None:
            self.crosshair.add_central_plot(self.spectrogram_widget)
        self.a = a

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
        self.spectrogram_widget = specto.create_spectrogram_plot(self.spc.frequencies, self.spc.timestamps, self.spc.data_matrix)
        self.curve_layout.insertWidget(1, self.spectrogram_widget)
        self.spectrogram_loaded = True
        self.crosshair.add_central_plot(self.spectrogram_widget)

    def link_plots(self):
        if not self.central_plots:
            return
        for p in self.central_plots:
            p.setXLink(self.central_plots[0])

    def load_eva(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav)")
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
            a = MinMaxAnalyser(f"EVA-{1}", times, channel, MinMaxFinder(), self.get_selected_region_interval, secondary_viewbox=None)
            self.curve_layout.addWidget(a)
            self.curve_layout.addWidget(a.visibility_checkbox)
            self.curve_layout.addWidget(a)
            self.crosshair.add_display_plot(a.plot_widget)
            self.a=a

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
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Save TextGrid File", "", "TextGrid Files (*.TextGrid)")
        if not filepath:
            return
        tgt_textgrid = self.textgrid.to_textgrid()
        tgt.io.write_to_file(tgt_textgrid, filepath, format="long")

    def load_annotations(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open TextGrid File", "", "TextGrid Files (*.TextGrid)")
        if not filepath:
            return
        tgt_textgrid = tgt.io.read_textgrid(filepath)
        self.textgrid = ui_tgt.TextgridTGTConvert().from_textgrid(tgt_textgrid, self.a.plot_widget)
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

    def export_to_csv(self):
        selected_panel = int(self.analysis_panel_combo_box.currentText()) - 1
        panel = self.panels[selected_panel][1]

        if not hasattr(panel, 'plot_items'):
            return

        filepath, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv)")
        if not filepath:
            return

        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['Curve Name', 'Average Min Peaks', 'Average Max Peaks', 'Average All Values']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for row, plot_items in panel.plot_items.items():
                combo_box = self.dashboard.cellWidget(row, 0)
                curve_name = combo_box.currentText()
                for curve in plot_items:
                    interval = self.get_selected_region_interval()
                    min_finder = MinMaxFinder()
                    max_finder = MinMaxFinder()
                    x_min, y_min = min_finder.analyse_minimum(curve.xData, curve.yData, interval)
                    x_max, y_max = max_finder.analyse_maximum(curve.xData, curve.yData, interval)
                    interval_times, interval_values = min_finder.find_in_interval(curve.xData, curve.yData, interval)
                    avg_min_peaks = np.mean(y_min) if len(y_min) > 0 else None
                    avg_max_peaks = np.mean(y_max) if len(y_max) > 0 else None
                    avg_all_values = np.mean(interval_values) if len(interval_values) > 0 else None
                    writer.writerow({
                        'Curve Name': curve_name,
                        'Average Min Peaks': avg_min_peaks,
                        'Average Max Peaks': avg_max_peaks,
                        'Average All Values': avg_all_values
                    })

def read_AG50x(path_to_pos_file):
    dims = ["x","z","y","phi","theta","rms","extra"]
    channel_sample_size = {
        8 : 56,
        16 : 112,
        32 : 256
    }
    pos_file = open(path_to_pos_file, mode="rb")
    file_content = pos_file.read()
    pos_file.seek(0)
    pos_file.readline()
    header_size = int(pos_file.readline().decode("utf8"))
    header_section = file_content[0:header_size]
    header = header_section.decode("utf8").split("\n")
    num_of_channels = int(header[2].split("=")[1])
    ema_samplerate = int(header[3].split("=")[1])
    data = file_content[header_size:]
    data = np.frombuffer(data, np.float32)
    data = np.reshape(data, newshape=(-1, channel_sample_size[num_of_channels]))
    pos = data.reshape(len(data), -1, 7)
    time = np.linspace(0, pos.shape[0] / ema_samplerate, pos.shape[0])
    ema_data = xr.Dataset(
        data_vars=dict(ema=(["time", "channels", "dimensions"], pos)),
        coords=dict(
            time=(["time"], time),
            channels=(["channels"], np.arange(pos.shape[1])),
            dimensions=(["dimensions"], dims)
        ),
        attrs=dict(
            device="AG50x",
            duration=time[-1],
            samplerate=ema_samplerate
        )
    )
    return ema_data

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = AudioAnalyzer()
    ex.show()
    sys.exit(app.exec_())
