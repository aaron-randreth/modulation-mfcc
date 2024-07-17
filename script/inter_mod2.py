import sys
from math import sqrt
import scipy
import numpy as np
import csv
from librosa import feature as lf
from pydub import AudioSegment
from PyQt5.QtCore import QTimer
import threading
import time
from pydub import AudioSegment
from pydub.playback import play

from PyQt5.QtWidgets import QCheckBox, QTableWidget, QTableWidgetItem, QComboBox, QMenu, QStackedWidget, QMenuBar, QToolBar, QAction, QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QGridLayout, QLabel, QScrollArea, QListWidget, QAbstractItemView, QDialog, QDialogButtonBox, QColorDialog, QInputDialog

from PyQt5.QtCore import Qt
import pyqtgraph as pg
import parselmouth
import tgt
import xarray as xr

from praat_py_ui import tiers as ui_tiers, textgridtools as ui_tgt, spectrogram as specto, parselmouth_calc as calc

from datasources.mfcc import load_channel, get_MFCCS_change
from scrollable_window import Info, InfoBox, Output

from calc import calc_formants, MinMaxFinder
from ui import create_plot_widget, SelectableListDialog, Crosshair, MinMaxAnalyser

pg.setConfigOptions(foreground="black", background="w")

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
       # Create Zoom Toolbar
        self.zoom_toolbar = QToolBar(self)
        self.zoom_toolbar.setStyleSheet("background-color: lightgray;")
        self.audio_cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=2))
        self.audio_cursor = pg.LinearRegionItem([0, 0.01], movable=False, brush=pg.mkBrush(0, 255, 0, 50))
        self.audio_cursor.hide()
        self.playing = False
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_cursor)

        self.audio_cursor.hide()
        self.playing = False
        self.player = None
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_cursor)
        right_layout.addWidget(self.button_container())

        self.zoom_in_action = QAction("Zoom In", self)
        self.zoom_in_action.triggered.connect(self.zoom_in)
        self.zoom_out_action = QAction("Zoom Out", self)
        self.zoom_out_action.triggered.connect(self.zoom_out)
        self.zoom_to_region_action = QAction("Zoom to Region", self)
        self.zoom_to_region_action.triggered.connect(self.zoom_to_region)
        self.zoom_toolbar.addAction(self.zoom_in_action)
        self.zoom_toolbar.addAction(self.zoom_out_action)
        self.zoom_toolbar.addAction(self.zoom_to_region_action)

        right_layout.addWidget(self.zoom_toolbar)  # Add zoom toolbar above analysis toolbar

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
        # self.analysis_toolbar.addAction(self.manual_peak_removal)
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
    def play_selected_region(self):
        if not self.file_path:
            return

        region = self.get_selected_region_interval()
        if region is None:
            return

        start, end = region
        duration = end - start

        audio = AudioSegment.from_wav(self.file_path)
        selected_audio = audio[start * 1000:end * 1000]  
        def play_audio():
            self.playing = True
            play(selected_audio)
            self.playing = False

        threading.Thread(target=play_audio).start()

        self.audio_cursor.setRegion([start, start])
        self.audio_cursor.show()
        threading.Thread(target=self.animate_cursor, args=(start, end, duration)).start()

    def animate_cursor(self, start, end, duration):
        start_time = time.time()
        while time.time() - start_time < duration:
            elapsed_time = time.time() - start_time
            current_pos = start + elapsed_time
            self.audio_cursor.setRegion([start, current_pos])
            time.sleep(0.01) 
        self.stop_audio()

    def stop_audio(self):
        self.audio_cursor.hide()
        self.playing = False
        self.update_timer.stop()

    def update_cursor(self):
        if self.playing:
            current_pos = self.audio_cursor.getRegion()[1]
            self.audio_cursor.setRegion([self.audio_cursor.getRegion()[0], current_pos + 0.01])  


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
        # print(f"Clicked on panel {panel_index + 1} at x: {x}, y: {y}")

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

            mfcc_curve.setCurveClickable(True)
            mfcc_curve.sigClicked.connect(self.add_point_on_click)

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

            formant_points = pg.ScatterPlotItem(x=f_times, y=formant_values, symbol='o', size=5, pen=pg.mkPen('b'), brush=pg.mkBrush('b'))
            formant_points.sigClicked.connect(self.add_point_on_click)

            print(f"Formant {formant_num} curve connected to click event")
            panel.secondary_viewbox.addItem(formant_points)
            if not hasattr(panel, 'plot_items'):
                panel.plot_items = {}
            panel.plot_items[row] = [formant_points]
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
                ema_curve.setCurveClickable(True)
                ema_curve.sigClicked.connect(self.add_point_on_click)

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
                # Masquer ou afficher les points de maximum et de minimum associés
                if hasattr(plot_item, 'max_points'):
                    if state == Qt.Checked:
                        plot_item.max_points.show()
                    else:
                        plot_item.max_points.hide()
                if hasattr(plot_item, 'min_points'):
                    if state == Qt.Checked:
                        plot_item.min_points.show()
                    else:
                        plot_item.min_points.hide()

    def compute_max(self):
        selected_panel = int(self.analysis_panel_combo_box.currentText()) - 1
        panel = self.panels[selected_panel][1]
        if not hasattr(panel, 'plot_items'):
            return
        for plot_items in panel.plot_items.values():
            for item in plot_items:
                if isinstance(item, pg.PlotDataItem):
                    x, y = item.xData, item.yData
                elif isinstance(item, pg.ScatterPlotItem):
                    spots = item.points()
                    x = np.array([spot.pos()[0] for spot in spots])
                    y = np.array([spot.pos()[1] for spot in spots])
                else:
                    continue

                max_finder = MinMaxFinder()
                x_max, y_max = max_finder.analyse_maximum(x, y, self.get_selected_region_interval())
                if len(x_max) == 0 or len(y_max) == 0:
                    print("No maximums found within the selected region.")
                    continue
                max_points = pg.ScatterPlotItem(x=x_max, y=y_max, symbol="x", size=10, pen=pg.mkPen("g"), brush=pg.mkBrush("b"))
                if hasattr(panel, 'secondary_viewbox') and item.getViewBox() is panel.secondary_viewbox:
                    panel.secondary_viewbox.addItem(max_points)
                elif hasattr(panel, 'tertiary_viewbox') and item.getViewBox() is panel.tertiary_viewbox:
                    panel.tertiary_viewbox.addItem(max_points)
                else:
                    panel.addItem(max_points)
                item.max_points = max_points  # Associate max points with the scatter plot item

    def compute_min(self):
        selected_panel = int(self.analysis_panel_combo_box.currentText()) - 1
        panel = self.panels[selected_panel][1]
        if not hasattr(panel, 'plot_items'):
            return
        for plot_items in panel.plot_items.values():
            for item in plot_items:
                if isinstance(item, pg.PlotDataItem):
                    x, y = item.xData, item.yData
                elif isinstance(item, pg.ScatterPlotItem):
                    spots = item.points()
                    x = np.array([spot.pos()[0] for spot in spots])
                    y = np.array([spot.pos()[1] for spot in spots])
                else:
                    continue

                min_finder = MinMaxFinder()
                x_min, y_min = min_finder.analyse_minimum(x, y, self.get_selected_region_interval())
                if len(x_min) == 0 or len(y_min) == 0:
                    print("No minimums found within the selected region.")
                    continue
                min_points = pg.ScatterPlotItem(x=x_min, y=y_min, symbol="o", size=10, pen=pg.mkPen("r"), brush=pg.mkBrush("r"))
                if hasattr(panel, 'secondary_viewbox') and item.getViewBox() is panel.secondary_viewbox:
                    panel.secondary_viewbox.addItem(min_points)
                elif hasattr(panel, 'tertiary_viewbox') and item.getViewBox() is panel.tertiary_viewbox:
                    panel.tertiary_viewbox.addItem(min_points)
                else:
                    panel.addItem(min_points)
                item.min_points = min_points  


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
        play_button = QPushButton("Play Selected Region")
        play_button.setStyleSheet("QPushButton { background-color: lightblue; border: 1px solid black; padding: 5px; }")
        play_button.clicked.connect(self.play_selected_region)
        layout.addWidget(play_button)
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
        sound_widget.addItem(self.audio_cursor)  # Ajouter le curseur ici
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
                    if isinstance(curve, pg.PlotDataItem):
                        x, y = curve.xData, curve.yData
                    elif isinstance(curve, pg.ScatterPlotItem):
                        spots = curve.points()
                        x = np.array([spot.pos()[0] for spot in spots])
                        y = np.array([spot.pos()[1] for spot in spots])
                    else:
                        continue

                    interval = self.get_selected_region_interval()
                    min_finder = MinMaxFinder()
                    max_finder = MinMaxFinder()
                    x_min, y_min = min_finder.analyse_minimum(x, y, interval)
                    x_max, y_max = max_finder.analyse_maximum(x, y, interval)
                    interval_times, interval_values = min_finder.find_in_interval(x, y, interval)
                    avg_min_peaks = np.mean(y_min) if len(y_min) > 0 else None
                    avg_max_peaks = np.mean(y_max) if len(y_max) > 0 else None
                    avg_all_values = np.mean(interval_values) if len(interval_values) > 0 else None
                    writer.writerow({
                        'Curve Name': curve_name,
                        'Average Min Peaks': avg_min_peaks,
                        'Average Max Peaks': avg_max_peaks,
                        'Average All Values': avg_all_values
                    })


    def add_point_on_click(self, plot_item, event):
        pos = event.scenePos()
        if not plot_item.getViewBox().sceneBoundingRect().contains(pos):
            return

        mouse_point = plot_item.getViewBox().mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()

        if self.manual_peak_maximum_addition.isChecked():
            if not hasattr(plot_item, "max_points"):
                return

            points_x, points_y = plot_item.max_points.getData()
            closest_index = np.argmin(np.abs(points_x - x))
            points_x = np.insert(points_x, closest_index, x)
            points_y = np.insert(points_y, closest_index, y)

            plot_item.max_points.setData(points_x, points_y)
            plot_item.max_points.show()
            print("Added max point")
        elif self.manual_peak_minimum_addition.isChecked():
            if not hasattr(plot_item, "min_points"):
                return

            points_x, points_y = plot_item.min_points.getData()
            closest_index = np.argmin(np.abs(points_x - x))
            points_x = np.insert(points_x, closest_index, x)
            points_y = np.insert(points_y, closest_index, y)
            plot_item.min_points.setData(points_x, points_y)
            plot_item.min_points.show()
            print("Added min point")
        elif self.manual_peak_removal.isChecked():

            if not hasattr(plot_item, "max_points"):
                return

            if not hasattr(plot_item, "min_points"):
                return

            points_x, points_y = plot_item.max_points.getData()
            distances = np.sqrt((points_x - x) ** 2 + (points_y - y) ** 2)
            closest_index = np.argmin(distances)
            points_x = np.delete(points_x, closest_index)
            points_y = np.delete(points_y, closest_index)
            plot_item.max_points.setData(points_x, points_y)
            plot_item.max_points.show()
            print("Removed point")

    def zoom_in(self):
        selected_panel = int(self.analysis_panel_combo_box.currentText()) - 1
        panel = self.panels[selected_panel][1]
        view_range = panel.viewRange()
        x_range = view_range[0]
        y_range = view_range[1]
        zoom_factor = 0.8  # Zoom in by 20%
        panel.setXRange(x_range[0] + (x_range[1] - x_range[0]) * (1 - zoom_factor) / 2,
                        x_range[1] - (x_range[1] - x_range[0]) * (1 - zoom_factor) / 2)
        panel.setYRange(y_range[0] + (y_range[1] - y_range[0]) * (1 - zoom_factor) / 2,
                        y_range[1] - (y_range[1] - y_range[0]) * (1 - zoom_factor) / 2)

    def zoom_out(self):
        selected_panel = int(self.analysis_panel_combo_box.currentText()) - 1
        panel = self.panels[selected_panel][1]
        view_range = panel.viewRange()
        x_range = view_range[0]
        y_range = view_range[1]
        zoom_factor = 1.2  # Zoom out by 20%
        panel.setXRange(x_range[0] + (x_range[1] - x_range[0]) * (1 - zoom_factor) / 2,
                        x_range[1] - (x_range[1] - x_range[0]) * (1 - zoom_factor) / 2)
        panel.setYRange(y_range[0] + (y_range[1] - y_range[0]) * (1 - zoom_factor) / 2,
                        y_range[1] - (y_range[1] - y_range[0]) * (1 - zoom_factor) / 2)

    def zoom_to_region(self):
        selected_panel = int(self.analysis_panel_combo_box.currentText()) - 1
        panel = self.panels[selected_panel][1]
        if self.selected_region.isVisible():
            region = self.selected_region.getRegion()
            panel.setXRange(region[0], region[1], padding=0)
            self.restore_y_ranges(panel)

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
