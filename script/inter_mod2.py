import sys
import os
import numpy as np
import csv
import sounddevice as sd
import wave
from librosa import feature as lf
from PyQt5.QtCore import QTimer, Qt
import threading
import time
from pydub import AudioSegment
from pydub.playback import play
from PyQt5.QtWidgets import QCheckBox, QHeaderView, QGroupBox, QTableWidget, QTableWidgetItem, QComboBox, QMenu, QStackedWidget, QMenuBar, QToolBar, QAction, QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QGridLayout, QLabel, QScrollArea, QListWidget, QAbstractItemView, QDialog, QDialogButtonBox, QColorDialog, QInputDialog
from scipy.io import wavfile
from scipy.interpolate import Akima1DInterpolator
import pyqtgraph as pg
import parselmouth
import tgt
from praat_py_ui import tiers as ui_tiers, textgridtools as ui_tgt, spectrogram as specto, parselmouth_calc as calc
from datasources.mfcc import load_channel, get_MFCCS_change
from scrollable_window import Info, InfoBox, Output
from calc import calc_formants, MinMaxFinder, read_AG50x, calculate_amplitude_envelope
from ui import create_plot_widget, SelectableListDialog, Crosshair, MinMaxAnalyser

pg.setConfigOptions(foreground="black", background="w")

class AudioAnalyzer(QMainWindow):
    textgrid: ui_tiers.TextGrid | None = None
    spectrogram_stacked_widget: QStackedWidget
    spectrogram_loaded: bool = False

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Analyzer")

        main_widget = QWidget(self)
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        self.textgrid_path = None  
        self.audio_name_group_box = QGroupBox("Audio Loaded")
        self.audio_name_group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid gray;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
        """)
        audio_name_layout = QVBoxLayout()
        self.audio_name_label = QLabel("No audio loaded")
        self.audio_name_label.setStyleSheet("font-size: 16px; color: blue;")
        audio_name_layout.addWidget(self.audio_name_label)
        self.audio_name_group_box.setLayout(audio_name_layout)
        right_layout.addWidget(self.audio_name_group_box)
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
            empty_plot.getViewBox().setMouseEnabled(y=False)
            empty_plot.getViewBox().setLimits(xMin=0)
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
        self.dashboard = QTableWidget(4, 7, self)  
        self.dashboard.setHorizontalHeaderLabels(["Acoustique", "EMA", "Couleur", "Panel", "Visibility", "Clear", "Dérivée"])
        main_layout.addLayout(left_layout, 2)  
        main_layout.addLayout(right_layout, 1)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        header = self.dashboard.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        header.setSectionResizeMode(6, QHeaderView.Fixed)
        header.resizeSection(3, 10)  
        header.resizeSection(4, 10)  
        header.resizeSection(4, 10) 
        header.resizeSection(6, 80) 
        for row in range(4):
            combo_box = QComboBox()
            combo_box.addItems(["Choose", "Modulation cepstrale", "Formant 1", "Formant 2", "Formant 3", "Courbes ema", "Amplitude Envelope"])
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
            color_names = ["brown", "red", "green", "blue", "orange", "purple", "pink", "black"]
            for color in color_names:
                color_button = QPushButton()
                color_button.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
                color_button.setFixedSize(20, 20)
                color_button.clicked.connect(lambda _, row=row, color=color: self.change_curve_color(row, color))
                color_buttons_layout.addWidget(color_button)
            color_buttons_widget = QWidget()
            color_buttons_widget.setLayout(color_buttons_layout)
            self.dashboard.setCellWidget(row, 2, color_buttons_widget)
            clear_button = QPushButton("Clear")
            clear_button.setStyleSheet("QPushButton { background-color: lightcoral; border: 1px solid black; padding: 5px; }")
            clear_button.clicked.connect(lambda _, row=row: self.clear_curve(row))
            self.dashboard.setCellWidget(row, 5, clear_button)
            derived_combo_box = QComboBox()  
            derived_combo_box.addItems(["Original", "Dérivée"])
            derived_combo_box.currentIndexChanged.connect(lambda index, row=row: self.update_derived(row, index))
            self.dashboard.setCellWidget(row, 6, derived_combo_box) 
        self.add_row_button = QPushButton("+")
        self.add_row_button.setStyleSheet("QPushButton { background-color: lightgreen; border: 1px solid black; padding: 5px; }")
        self.add_row_button.clicked.connect(self.add_dashboard_row)
        right_layout.addWidget(self.dashboard)
        right_layout.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.add_row_button)
        right_layout.addWidget(self.create_load_buttons())
        right_layout.addWidget(self.create_audio_control_buttons())

        self.textgrid_controls_group_box = QGroupBox("TextGrid Controls")
        textgrid_controls_layout = QVBoxLayout()
        self.textgrid_status_label = QLabel("No TextGrid loaded")
        self.textgrid_status_label.setStyleSheet("font-size: 16px; color: red;")
        textgrid_controls_layout.addWidget(self.textgrid_status_label)
        self.display_textgrid_checkbox = QCheckBox("Afficher TextGrid")
        self.display_textgrid_checkbox.setChecked(False)
        self.display_textgrid_checkbox.stateChanged.connect(self.toggle_textgrid_display)
        self.display_textgrid_checkbox.setEnabled(True)
        textgrid_controls_layout.addWidget(self.display_textgrid_checkbox)
        self.textgrid_controls_group_box.setLayout(textgrid_controls_layout)
        right_layout.addWidget(self.textgrid_controls_group_box)

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
        self.zoom_in_action = QAction("Zoom In", self)
        self.zoom_in_action.triggered.connect(self.zoom_in)
        self.zoom_out_action = QAction("Zoom Out", self)
        self.zoom_out_action.triggered.connect(self.zoom_out)
        self.zoom_to_region_action = QAction("Zoom to Region", self)
        self.zoom_to_region_action.triggered.connect(self.zoom_to_region)
        self.zoom_toolbar.addAction(self.zoom_in_action)
        self.zoom_toolbar.addAction(self.zoom_out_action)
        self.zoom_toolbar.addAction(self.zoom_to_region_action)
        right_layout.addWidget(self.zoom_toolbar) 
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
        right_layout.addStretch(1) 
        main_layout.addLayout(left_layout, 3)  
        main_layout.addLayout(right_layout, 1)  
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        pen = pg.mkPen(width=4, color='b')  
        self.selected_region = pg.LinearRegionItem(pen=pen, brush=None)
        self.selected_region.setZValue(10)
        self.selected_region.hide()
        self.textgrid_visibility_checkboxes = {}
        self.crosshair = Crosshair([])
        self.connect_panel_clicks()
        self.recording = False
        self.frames = []
        self.record_button = QPushButton("Record Audio")
        self.record_button.setStyleSheet("QPushButton { background-color: lightgreen; border: 1px solid black; padding: 5px; }")
        self.record_button.clicked.connect(self.toggle_recording)

    def init_real_time_plot(self):
        if not hasattr(self, 'real_time_plot') or self.real_time_plot is None:
            self.real_time_plot = pg.PlotWidget(title="Real-time Audio")
            self.real_time_curve = self.real_time_plot.plot(pen='y')
            self.curve_layout.addWidget(self.real_time_plot)

    def update_real_time_plot(self, audio_data):
        self.real_time_curve.setData(audio_data.flatten())

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
        if (self.playing):
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
        combo_box.addItems(["Choose", "Modulation cepstrale", "Formant 1", "Formant 2", "Formant 3", "Courbes ema", "Amplitude Envelope"])
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
        color_names = ["brown", "red", "green", "blue", "orange", "purple", "pink", "black"]
        for color in color_names:
            color_button = QPushButton()
            color_button.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
            color_button.setFixedSize(20, 20)
            color_button.clicked.connect(lambda _, row=row, color=color: self.change_curve_color(row, color))
            color_buttons_layout.addWidget(color_button)
        color_buttons_widget = QWidget()
        color_buttons_widget.setLayout(color_buttons_layout)
        self.dashboard.setCellWidget(row, 2, color_buttons_widget)
        clear_button = QPushButton("Clear")
        clear_button.setStyleSheet("QPushButton { background-color: lightcoral; border: 1px solid black; padding: 5px; }")
        clear_button.clicked.connect(lambda _, row=row: self.clear_curve(row))
        self.dashboard.setCellWidget(row, 5, clear_button)
        derived_combo_box = QComboBox() 
        derived_combo_box.addItems(["Original", "Dérivée"])
        derived_combo_box.currentIndexChanged.connect(lambda index, row=row: self.update_derived(row, index))
        self.dashboard.setCellWidget(row, 6, derived_combo_box) 

    def clear_curve(self, row):
        panel_combo_box = self.dashboard.cellWidget(row, 3)
        selected_panel = int(panel_combo_box.currentText()) - 1
        panel = self.panels[selected_panel][1]

        if hasattr(panel, 'plot_items') and row in panel.plot_items:
            for plot_item in panel.plot_items[row]:
                if hasattr(panel, 'secondary_viewbox') and plot_item.getViewBox() is panel.secondary_viewbox:
                    panel.secondary_viewbox.removeItem(plot_item)
                elif hasattr(panel, 'tertiary_viewbox') and plot_item.getViewBox() is panel.tertiary_viewbox:
                    panel.tertiary_viewbox.removeItem(plot_item)
                else:
                    panel.removeItem(plot_item)
            del panel.plot_items[row]
        print(f"Cleared row {row}")

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

        if hasattr(panel, 'plot_items') and row in panel.plot_items:
            for old_item in panel.plot_items[row]:
                if hasattr(panel, 'secondary_viewbox') and old_item.getViewBox() is panel.secondary_viewbox:
                    panel.secondary_viewbox.removeItem(old_item)
                elif hasattr(panel, 'tertiary_viewbox') and old_item.getViewBox() is panel.tertiary_viewbox:
                    panel.tertiary_viewbox.removeItem(old_item)
                elif hasattr(panel, 'quaternary_viewbox') and old_item.getViewBox() is panel.quaternary_viewbox:
                    panel.quaternary_viewbox.removeItem(old_item)
                else:
                    panel.removeItem(old_item)
            del panel.plot_items[row]

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

            table_item = self.dashboard.cellWidget(row, 0)
            table_item.setItemText(5, channel_label)

        if index == 6:  # Amplitude Envelope
            start, end = self.get_selected_region_interval()
            if start is None or end is None:
                return
            sample_rate, audio_signal = wavfile.read(self.file_path)
            audio_signal = audio_signal[int(start * sample_rate):int(end * sample_rate)]
            amplitude_envelope = calculate_amplitude_envelope(audio_signal, sample_rate)
            time_axis = np.linspace(start, end, len(amplitude_envelope))

            if not hasattr(panel, 'quaternary_viewbox'):
                panel.quaternary_viewbox = pg.ViewBox()
                panel.getPlotItem().scene().addItem(panel.quaternary_viewbox)
                right_axis = pg.AxisItem('right')
                panel.getPlotItem().layout.addItem(right_axis, 2, 4)
                right_axis.linkToView(panel.quaternary_viewbox)
                panel.quaternary_viewbox.setXLink(panel)
                panel.getPlotItem().getViewBox().sigResized.connect(lambda: panel.quaternary_viewbox.setGeometry(panel.getPlotItem().getViewBox().sceneBoundingRect()))

            envelope_curve = pg.PlotDataItem(time_axis, amplitude_envelope, pen='m', clear=False)
            envelope_curve.setCurveClickable(True)
            envelope_curve.sigClicked.connect(self.add_point_on_click)

            print("Amplitude Envelope curve connected to click event")
            panel.quaternary_viewbox.addItem(envelope_curve)
            if not hasattr(panel, 'plot_items'):
                panel.plot_items = {}
            panel.plot_items[row] = [envelope_curve]
            panel.getPlotItem().showAxis('right')
            panel.getPlotItem().getAxis('right').setLabel('Amplitude Envelope')

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
                elif hasattr(panel, 'quaternary_viewbox') and item.getViewBox() is panel.quaternary_viewbox:
                    panel.quaternary_viewbox.addItem(max_points)
                else:
                    panel.addItem(max_points)
                item.max_points = max_points  

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
                elif hasattr(panel, 'quaternary_viewbox') and item.getViewBox() is panel.quaternary_viewbox:
                    panel.quaternary_viewbox.addItem(min_points)
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

    def create_load_buttons(self):
        load_group_box = QGroupBox("Load Audio and TextGrid Controls")
        load_layout = QVBoxLayout()

        load_audio_button = QPushButton("Load Audio")
        load_audio_button.setStyleSheet("QPushButton { background-color: lightblue; border: 1px solid black; padding: 5px; }")
        load_audio_button.clicked.connect(self.load_audio)
        load_layout.addWidget(load_audio_button)

        load_textgrid_button = QPushButton("Load TextGrid")
        load_textgrid_button.setStyleSheet("QPushButton { background-color: lightblue; border: 1px solid black; padding: 5px; }")
        load_textgrid_button.clicked.connect(self.load_annotations)
        load_layout.addWidget(load_textgrid_button)

        self.record_button = QPushButton("Record Audio")
        self.record_button.setStyleSheet("QPushButton { background-color: lightgreen; border: 1px solid black; padding: 5px; }")
        self.record_button.clicked.connect(self.toggle_recording)
        load_layout.addWidget(self.record_button)

        load_group_box.setLayout(load_layout)
        return load_group_box

    def create_audio_control_buttons(self):
        audio_control_group_box = QGroupBox("Audio Control")
        audio_control_layout = QVBoxLayout()

        play_button = QPushButton("Play Selected Region")
        play_button.setStyleSheet("QPushButton { background-color: lightblue; border: 1px solid black; padding: 5px; }")
        play_button.clicked.connect(self.play_selected_region)
        audio_control_layout.addWidget(play_button)

        self.toggle_spectrogram_button = QPushButton("Show/Mask Spectrogram")
        self.toggle_spectrogram_button.setCheckable(True)
        self.toggle_spectrogram_button.setStyleSheet("QPushButton { background-color: lightcoral; border: 1px solid black; padding: 5px; }")
        self.toggle_spectrogram_button.clicked.connect(self.toggle_spectrogram)
        audio_control_layout.addWidget(self.toggle_spectrogram_button)

        audio_control_group_box.setLayout(audio_control_layout)
        return audio_control_group_box

    def curves_container(self) -> QWidget:
        curve_parent = QWidget()
        scroll_area = QScrollArea()
        scroll_layout = QVBoxLayout()
        curve_parent.setLayout(scroll_layout)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(curve_parent)
        return scroll_area, scroll_layout

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav)")
        if not file_path:
            return
        self.clear_panels() 
        self.file_path = file_path
        sound_data = calc.Parselmouth(file_path)
        snd = sound_data.get_sound()
        self.spc = sound_data.get_spectrogram()
        self.audio_widget = create_plot_widget(snd.timestamps, snd.amplitudes[0], color='k')
        self.audio_widget.getViewBox().setMouseEnabled(y=False)
        self.audio_widget.getViewBox().setLimits(xMin=0)  
        self.audio_widget.addItem(self.audio_cursor)  
        self.add_selection_tool(self.audio_widget)
        self.curve_layout.addWidget(self.audio_widget)

        spectrogram_widget = None
        if self.spectrogram_loaded:
            self.spectrogram_widget = specto.create_spectrogram_plot(self.spc.frequencies, self.spc.timestamps, self.spc.data_matrix)
            self.spectrogram_stacked_widget.addWidget(self.spectrogram_widget)
            self.spectrogram_stacked_widget.setCurrentWidget(self.spectrogram_widget)
        audio_data = load_channel(file_path)
        x_mfccs, y_mfccs = get_MFCCS_change(audio_data)
        self.add_selection_tool(self.audio_widget)
        a = MinMaxAnalyser("Mfcc", x_mfccs, y_mfccs, MinMaxFinder(), self.get_selected_region_interval)
        self.crosshair = Crosshair([self.audio_widget])
        if self.spectrogram_widget is not None:
            self.crosshair.add_central_plot(self.spectrogram_widget)

        self.a = a
        self.set_x_axis_limits(x_min=0, x_max=snd.timestamps[-1])
        audio_name = os.path.basename(file_path)
        self.audio_name_label.setText(f"Loaded audio: {audio_name}")
        self.selected_region.setRegion([0, 1])

    def set_x_axis_limits(self, x_min, x_max):
        for panel, plot_widget in self.panels:
            self._set_plot_x_limits(plot_widget, x_min, x_max)
        if hasattr(self, 'audio_widget'):
            self._set_plot_x_limits(self.audio_widget, x_min, x_max)

    def _set_plot_x_limits(self, plot_widget, x_min, x_max):
        view_box = plot_widget.getPlotItem().getViewBox()
        view_box.setLimits(xMin=x_min, xMax=x_max)
        view_box.setXRange(x_min, x_max, padding=0)
        if hasattr(plot_widget, 'secondary_viewbox'):
            plot_widget.secondary_viewbox.setLimits(xMin=x_min, xMax=x_max)
            plot_widget.secondary_viewbox.setXRange(x_min, x_max, padding=0)
        if hasattr(plot_widget, 'tertiary_viewbox'):
            plot_widget.tertiary_viewbox.setLimits(xMin=x_min, xMax=x_max)
            plot_widget.tertiary_viewbox.setXRange(x_min, x_max, padding=0)
        if hasattr(plot_widget, 'quaternary_viewbox'):
            plot_widget.quaternary_viewbox.setLimits(xMin=x_min, xMax=x_max)
            plot_widget.quaternary_viewbox.setXRange(x_min, x_max, padding=0)

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

    def load_annotations(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open TextGrid File", "", "TextGrid Files (*.TextGrid)")
        if not filepath:
            self.textgrid_path = None 
            return self.textgrid_path
        self.textgrid_path = filepath 
        tgt_textgrid = tgt.io.read_textgrid(filepath)
        self.textgrid = ui_tgt.TextgridTGTConvert().from_textgrid(tgt_textgrid, self.a.plot_widget)
        for tier in self.textgrid.get_tiers():
            self.crosshair.add_display_plot(tier)
        self.curve_layout.addWidget(self.textgrid)
        self.textgrid.setMaximumHeight(100) 
        self.textgrid_status_label.setText(f"Loaded TextGrid: {os.path.basename(filepath)}")
        self.textgrid_status_label.setStyleSheet("font-size: 16px; color: green;")
        self.display_textgrid_checkbox.setEnabled(True)

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

        min_time = float('inf')
        max_time = float('-inf')
        for plot_items in panel.plot_items.values():
            for curve in plot_items:
                if isinstance(curve, pg.PlotDataItem):
                    x_data = curve.xData
                elif isinstance(curve, pg.ScatterPlotItem):
                    spots = curve.points()
                    x_data = np.array([spot.pos()[0] for spot in spots])
                    y_data = np.array([spot.pos()[1] for spot in spots])
                else:
                    continue
                min_time = min(min_time, x_data.min())
                max_time = max(max_time, x_data.max())

        time_axis = np.arange(min_time, max_time, 0.005)

        fieldnames = ['Time']
        measurements = {}
        all_rows = {t: {'Time': t} for t in time_axis}

        textgrid_intervals = {}
        if self.textgrid_path:
            textgrid = tgt.io.read_textgrid(self.textgrid_path)
            if textgrid.get_tier_names():
                first_tier = textgrid.get_tier_by_name(textgrid.get_tier_names()[0])
                for interval in first_tier:
                    start_time = interval.start_time
                    end_time = interval.end_time
                    label = interval.text
                    for t in time_axis:
                        if start_time <= t <= end_time:
                            if t not in textgrid_intervals:
                                textgrid_intervals[t] = label

        for row in panel.plot_items.keys():
            combo_box = self.dashboard.cellWidget(row, 0)
            curve_name = combo_box.currentText()
            fieldnames.append(f"{curve_name} Y Values")
            fieldnames.append(f"{curve_name}_MoyenneAllValues")
            fieldnames.append(f"{curve_name}_MoyennePicMax")
            fieldnames.append(f"{curve_name}_MoyennePicMin")

            for curve in panel.plot_items[row]:
                if isinstance(curve, pg.PlotDataItem):
                    x_data, y_data = curve.xData, curve.yData
                elif isinstance(curve, pg.ScatterPlotItem):
                    spots = curve.points()
                    x_data = np.array([spot.pos()[0] for spot in spots])
                    y_data = np.array([spot.pos()[1] for spot in spots])
                else:
                    continue

                interpolator = Akima1DInterpolator(x_data, y_data)
                y_interpolated = interpolator(time_axis)
                interval = self.get_selected_region_interval()
                min_finder = MinMaxFinder()
                max_finder = MinMaxFinder()
                x_min, y_min = min_finder.analyse_minimum(x_data, y_data, interval)
                x_max, y_max = max_finder.analyse_maximum(x_data, y_data, interval)
                interval_times, interval_values = min_finder.find_in_interval(x_data, y_data, interval)
                avg_min_peaks = np.mean(y_min) if len(y_min) > 0 else None
                avg_max_peaks = np.mean(y_max) if len(y_max) > 0 else None
                avg_all_values = np.mean(interval_values) if len(interval_values) > 0 else None

                measurements[curve_name] = {
                    'MoyenneAllValues': avg_all_values,
                    'MoyennePicMax': avg_max_peaks,
                    'MoyennePicMin': avg_min_peaks
                }

                for t, y_val in zip(time_axis, y_interpolated):
                    all_rows[t][f"{curve_name} Y Values"] = y_val

        sorted_rows = [all_rows[t] for t in sorted(all_rows.keys())]

        if textgrid_intervals:
            fieldnames.append("TextGrid Interval")
            for t in time_axis:
                all_rows[t]["TextGrid Interval"] = textgrid_intervals.get(t, "")

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_rows)
            measurement_row = {'Time': 'Measurements'}
            for curve_name, measurement in measurements.items():
                measurement_row[f"{curve_name}_MoyenneAllValues"] = measurement['MoyenneAllValues']
                measurement_row[f"{curve_name}_MoyennePicMax"] = measurement['MoyennePicMax']
                measurement_row[f"{curve_name}_MoyennePicMin"] = measurement['MoyennePicMin']
            writer.writerow(measurement_row)

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

    def update_derived(self, row, index):
        panel_combo_box = self.dashboard.cellWidget(row, 3)
        selected_panel = int(panel_combo_box.currentText()) - 1
        panel = self.panels[selected_panel][1]

        if hasattr(panel, 'plot_items') and row in panel.plot_items:
            for plot_item in panel.plot_items[row]:
                if hasattr(panel, 'secondary_viewbox') and plot_item.getViewBox() is panel.secondary_viewbox:
                    panel.secondary_viewbox.removeItem(plot_item)
                elif hasattr(panel, 'tertiary_viewbox') and plot_item.getViewBox() is panel.tertiary_viewbox:
                    panel.tertiary_viewbox.removeItem(plot_item)
                elif hasattr(panel, 'quaternary_viewbox') and plot_item.getViewBox() is panel.quaternary_viewbox:
                    panel.quaternary_viewbox.removeItem(plot_item)
                else:
                    panel.removeItem(plot_item)
            del panel.plot_items[row]

        if index == 1:  # Dérivée
            combo_box = self.dashboard.cellWidget(row, 0)
            original_index = combo_box.currentIndex()

            if original_index == 6:  # Amplitude Envelope
                start, end = self.get_selected_region_interval()
                if start is None or end is None:
                    return
                sample_rate, audio_signal = wavfile.read(self.file_path)
                audio_signal = audio_signal[int(start * sample_rate):int(end * sample_rate)]
                amplitude_envelope = calculate_amplitude_envelope(audio_signal, sample_rate)
                y_derived = np.gradient(np.gradient(amplitude_envelope))
                time_axis = np.linspace(start, end, len(y_derived))

                if not hasattr(panel, 'quaternary_viewbox'):
                    panel.quaternary_viewbox = pg.ViewBox()
                    panel.getPlotItem().scene().addItem(panel.quaternary_viewbox)
                    right_axis = pg.AxisItem('right')
                    panel.getPlotItem().layout.addItem(right_axis, 2, 4)
                    right_axis.linkToView(panel.quaternary_viewbox)
                    panel.quaternary_viewbox.setXLink(panel)
                    panel.getPlotItem().getViewBox().sigResized.connect(lambda: panel.quaternary_viewbox.setGeometry(panel.getPlotItem().getViewBox().sceneBoundingRect()))

                derived_plot = pg.PlotDataItem(time_axis, y_derived, pen='r')
                panel.quaternary_viewbox.addItem(derived_plot)
                if not hasattr(panel, 'plot_items'):
                    panel.plot_items = {}
                panel.plot_items[row] = [derived_plot]
                panel.getPlotItem().showAxis('right')
                panel.getPlotItem().getAxis('right').setLabel('Amplitude Envelope Dérivée')
            else:
                start, end = self.get_selected_region_interval()
                if start is None or end is None:
                    return

                if original_index == 1:  # MFCC
                    audio_data = load_channel(self.file_path)
                    x_mfccs, y_mfccs = get_MFCCS_change(audio_data)
                    y_derived = np.gradient(np.gradient(y_mfccs))
                    derived_plot = pg.PlotDataItem(x_mfccs, y_derived, pen='r')
                    panel.addItem(derived_plot)
                    if not hasattr(panel, 'plot_items'):
                        panel.plot_items = {}
                    panel.plot_items[row] = [derived_plot]
                    panel.getPlotItem().getAxis('left').setLabel('MFCC Dérivée')
                elif original_index in [2, 3, 4]:  # Formants
                    formant_num = original_index - 1
                    f_times, f1_values, f2_values, f3_values = calc_formants(parselmouth.Sound(self.file_path), start, end)
                    if formant_num == 1:
                        formant_values = f1_values
                        formant_label = 'Formant 1 Dérivée'
                    elif formant_num == 2:
                        formant_values = f2_values
                        formant_label = 'Formant 2 Dérivée'
                    else:
                        formant_values = f3_values
                        formant_label = 'Formant 3 Dérivée'

                    y_derived = np.gradient(np.gradient(formant_values))
                    if not hasattr(panel, 'secondary_viewbox'):
                        panel.secondary_viewbox = pg.ViewBox()
                        panel.getPlotItem().scene().addItem(panel.secondary_viewbox)
                        right_axis = pg.AxisItem('right')
                        panel.getPlotItem().layout.addItem(right_axis, 2, 3)
                        right_axis.linkToView(panel.secondary_viewbox)
                        panel.secondary_viewbox.setXLink(panel)
                        panel.getPlotItem().getViewBox().sigResized.connect(lambda: panel.secondary_viewbox.setGeometry(panel.getPlotItem().getViewBox().sceneBoundingRect()))

                    derived_plot = pg.ScatterPlotItem(x=f_times, y=y_derived, symbol='o', size=5, pen=pg.mkPen('r'), brush=pg.mkBrush('r'))
                    panel.secondary_viewbox.addItem(derived_plot)
                    if not hasattr(panel, 'plot_items'):
                        panel.plot_items = {}
                    panel.plot_items[row] = [derived_plot]
                    panel.getPlotItem().showAxis('right')
                    panel.getPlotItem().getAxis('right').setLabel(formant_label)
        else: 
            self.update_panel(row, self.dashboard.cellWidget(row, 0).currentIndex())

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def hide_real_time_plot(self):
        if hasattr(self, 'real_time_plot') and self.real_time_plot is not None:
            self.real_time_plot.setParent(None)
            self.real_time_plot = None

    def start_recording(self):
        self.recording = True
        self.record_button.setText("Stop Recording")
        self.frames = []
        self.init_real_time_plot()
        self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=44100, dtype='int16')
        self.stream.start()


    def stop_recording(self):
        self.recording = False
        self.record_button.setText("Record Audio")
        self.stream.stop()
        self.stream.close()
        audio_data = np.concatenate(self.frames)
        non_zero_audio_data = audio_data[audio_data != 0]
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Recorded Audio", os.getcwd(), "Audio Files (*.wav)")
        if not file_path:
            return
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) 
            wf.setframerate(44100)
            wf.writeframes(non_zero_audio_data.tobytes())
        self.hide_real_time_plot()
        self.file_path = file_path
        self.load_audio()
        self.selected_region.setRegion([0, 1]) 
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.frames.append(indata.copy())
        audio_data = np.concatenate(self.frames)
        self.update_real_time_plot(audio_data)
    def reset_state(self):
        self.file_path = ""
        self.spectrogram_loaded = False
        self.selected_region.hide()
        self.reset_dashboard()
        self.clear_panels()
        self.clear_audio()        
        if self.textgrid:
            self.textgrid.setParent(None)
            self.textgrid = None
        self.textgrid_status_label.setText("No TextGrid loaded")
        self.textgrid_status_label.setStyleSheet("font-size: 16px; color: red;")
        self.display_textgrid_checkbox.setEnabled(False)
        self.display_textgrid_checkbox.setChecked(False)
        self.clear_formants_and_envelope()

    def clear_formants_and_envelope(self):
        for panel, plot_widget in self.panels:
            if hasattr(plot_widget, 'plot_items'):
                for row, items in plot_widget.plot_items.items():
                    for item in items:
                        if hasattr(panel, 'secondary_viewbox') and item.getViewBox() is panel.secondary_viewbox:
                            panel.secondary_viewbox.removeItem(item)
                        elif hasattr(panel, 'tertiary_viewbox') and item.getViewBox() is panel.tertiary_viewbox:
                            panel.tertiary_viewbox.removeItem(item)
                        elif hasattr(panel, 'quaternary_viewbox') and item.getViewBox() is panel.quaternary_viewbox:
                            panel.quaternary_viewbox.removeItem(item)
                        else:
                            plot_widget.removeItem(item)
                del plot_widget.plot_items

    def reset_dashboard(self):
        self.dashboard.clearContents()
        self.dashboard.setRowCount(4)
        for row in range(4):
            combo_box = QComboBox()
            combo_box.addItems(["Choose", "Modulation cepstrale", "Formant 1", "Formant 2", "Formant 3", "Courbes ema", "Amplitude Envelope"])
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
            color_names = ["brown", "red", "green", "blue", "orange", "purple", "pink", "black"]
            for color in color_names:
                color_button = QPushButton()
                color_button.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
                color_button.setFixedSize(20, 20)
                color_button.clicked.connect(lambda _, row=row, color=color: self.change_curve_color(row, color))
                color_buttons_layout.addWidget(color_button)
            color_buttons_widget = QWidget()
            color_buttons_widget.setLayout(color_buttons_layout)
            self.dashboard.setCellWidget(row, 2, color_buttons_widget)
            clear_button = QPushButton("Clear")
            clear_button.setStyleSheet("QPushButton { background-color: lightcoral; border: 1px solid black; padding: 5px; }")
            clear_button.clicked.connect(lambda _, row=row: self.clear_curve(row))
            self.dashboard.setCellWidget(row, 5, clear_button)
            derived_combo_box = QComboBox() 
            derived_combo_box.addItems(["Original", "Dérivée"])
            derived_combo_box.currentIndexChanged.connect(lambda index, row=row: self.update_derived(row, index))
            self.dashboard.setCellWidget(row, 6, derived_combo_box) 

    def clear_panels(self):
    
        for panel, plot_widget in self.panels:
            plot_widget.clear()
            plot_widget.getPlotItem().getAxis('left').setLabel('')
            plot_widget.getPlotItem().getAxis('bottom').setLabel('')
            if hasattr(plot_widget, 'secondary_viewbox'):
                plot_widget.getPlotItem().getAxis('right').setLabel('')
            if hasattr(plot_widget, 'tertiary_viewbox'):
                plot_widget.getPlotItem().getAxis('right').setLabel('')
            if hasattr(plot_widget, 'quaternary_viewbox'):
                plot_widget.getPlotItem().getAxis('right').setLabel('')
            if hasattr(plot_widget, 'plot_items'):
                keys_to_delete = list(plot_widget.plot_items.keys())
                for row in keys_to_delete:
                    items = plot_widget.plot_items[row]
                    for item in items:
                        item.clear()
                    del plot_widget.plot_items[row]
            if hasattr(panel, 'secondary_viewbox'):
                panel.secondary_viewbox.clear()
                panel.getPlotItem().layout.removeItem(panel.getPlotItem().getAxis('right'))
                delattr(panel, 'secondary_viewbox')

            if hasattr(panel, 'tertiary_viewbox'):
                panel.tertiary_viewbox.clear()
                panel.getPlotItem().layout.removeItem(panel.getPlotItem().getAxis('right'))
                delattr(panel, 'tertiary_viewbox')

            if hasattr(panel, 'quaternary_viewbox'):
                panel.quaternary_viewbox.clear()
                panel.getPlotItem().layout.removeItem(panel.getPlotItem().getAxis('right'))
                delattr(panel, 'quaternary_viewbox')

            plot_widget.getViewBox().setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
            plot_widget.getViewBox().enableAutoRange(axis=pg.ViewBox.XYAxes)

        if hasattr(self, 'audio_widget'):
            self.audio_widget.setParent(None)
            del self.audio_widget

        if hasattr(self, 'spectrogram_widget') and self.spectrogram_widget:
            self.spectrogram_widget.setParent(None)
            self.spectrogram_widget = None

        if hasattr(self, 'real_time_plot') and self.real_time_plot:
            self.real_time_plot.setParent(None)
            self.real_time_plot = None

        self.reset_dashboard()
        self.selected_region.hide()
        self.selected_region.setRegion([0, 0])
        if self.textgrid:
            self.textgrid.setParent(None)
            self.textgrid = None
            self.textgrid_status_label.setText("No TextGrid loaded")
            self.textgrid_status_label.setStyleSheet("font-size: 16px; color: red;")
            self.display_textgrid_checkbox.setEnabled(False)
            self.display_textgrid_checkbox.setChecked(False)

        self.file_path = ""
        self.audio_cursor.hide()
        self.playing = False
        self.update_timer.stop()
        self.frames.clear()

    def clear_audio(self):
        if hasattr(self, 'file_path'):
            self.file_path = ""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = AudioAnalyzer()
    ex.show()
    sys.exit(app.exec_())
