from abc import ABC, abstractmethod
from typing import override
import os
import sys
import threading
import wave
import time
import csv
import sys
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import find_peaks
from pydub.playback import play
from pydub import AudioSegment

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog

import pyqtgraph as pg
import parselmouth
import tgt

from config_dialog import UnifiedConfigDialog
from mfcc import load_channel, get_MFCCS_change
from calc import (
    calc_formants,
    calculate_amplitude_envelope,
    get_f0,
    get_velocity,
    read_AG50x,
)
from ui import Crosshair, create_plot_widget, ZoomToolbar
from praat_py_ui.parselmouth_calc import Parselmouth
from quadruple_axis_plot_item import (
    QuadrupleAxisPlotItem,
    Panel,
    PointOperation,
    CalculationValues,
    PanelWidget,
    SoundInformation,
    DisplayInterval,
)
class ExportCSVDialog(QtWidgets.QDialog):
    def __init__(self, axis_ids, curve_names, tier_names=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Data to Export")

        layout = QtWidgets.QVBoxLayout()

        self.selections = {}
        self.tier_selections = {}
        self.calculation_choices = {}

        # Curve selection
        for axis_id, curve_name in zip(axis_ids, curve_names):
            group_box = QtWidgets.QGroupBox(f"Curve {curve_name} Data")
            group_layout = QtWidgets.QFormLayout()

            x_checkbox = QtWidgets.QCheckBox("Include X values")
            y_checkbox = QtWidgets.QCheckBox("Include Y values")
            min_checkbox = QtWidgets.QCheckBox("Include Min Peaks")
            max_checkbox = QtWidgets.QCheckBox("Include Max Peaks")

            group_layout.addRow(x_checkbox)
            group_layout.addRow(y_checkbox)
            group_layout.addRow(min_checkbox)
            group_layout.addRow(max_checkbox)

            group_box.setLayout(group_layout)
            layout.addWidget(group_box)

            self.selections[curve_name] = {
                "x": x_checkbox,
                "y": y_checkbox,
                "min": min_checkbox,
                "max": max_checkbox,
            }

        # TextGrid tier selection if available
        if tier_names:
            tier_group_box = QtWidgets.QGroupBox("TextGrid Tiers to Include")
            tier_group_layout = QtWidgets.QFormLayout()

            for tier_name in tier_names:
                tier_checkbox = QtWidgets.QCheckBox(f"Include tier '{tier_name}'")
                tier_group_layout.addRow(tier_checkbox)
                self.tier_selections[tier_name] = tier_checkbox

            tier_group_box.setLayout(tier_group_layout)
            layout.addWidget(tier_group_box)

        # Calculation options
        calc_group_box = QtWidgets.QGroupBox("Calculations")
        calc_group_layout = QtWidgets.QFormLayout()

        duration_checkbox = QtWidgets.QCheckBox("Calculate Duration")
        mean_checkbox = QtWidgets.QCheckBox("Calculate Mean")
        region_or_tier_combo = QtWidgets.QComboBox()
        region_or_tier_combo.addItem("Region Selection")
        if tier_names:
            region_or_tier_combo.addItems(tier_names)

        calc_group_layout.addRow(duration_checkbox)
        calc_group_layout.addRow(mean_checkbox)
        calc_group_layout.addRow(QtWidgets.QLabel("Calculate on:"))
        calc_group_layout.addRow(region_or_tier_combo)

        calc_group_box.setLayout(calc_group_layout)
        layout.addWidget(calc_group_box)

        self.calculation_choices = {
            "duration": duration_checkbox,
            "mean": mean_checkbox,
            "region_or_tier": region_or_tier_combo,
        }

        self.ok_button = QtWidgets.QPushButton("Export")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def get_selections(self):
        selections = {}
        for curve_name, options in self.selections.items():
            selections[curve_name] = {
                "x": options["x"].isChecked(),
                "y": options["y"].isChecked(),
                "min": options["min"].isChecked(),
                "max": options["max"].isChecked(),
            }
        return selections

    def get_selected_tiers(self):
        selected_tiers = []
        for tier_name, checkbox in self.tier_selections.items():
            if checkbox.isChecked():
                selected_tiers.append(tier_name)
        return selected_tiers

    def get_calculation_choices(self):
        return {
            "calculate_duration": self.calculation_choices["duration"].isChecked(),
            "calculate_mean": self.calculation_choices["mean"].isChecked(),
            "region_or_tier": self.calculation_choices["region_or_tier"].currentText(),
        }


class POSChannelSelectionDialog(QtWidgets.QDialog):
    def __init__(self, pos_channels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select and Rename POS Channels")
        self.pos_channels = pos_channels
        self.selected_channels = {}

        self.layout = QtWidgets.QVBoxLayout(self)
        channel_layout = QtWidgets.QGridLayout()

        self.checkboxes = {}
        self.rename_edits = {}

        for i, channel in enumerate(self.pos_channels):
            checkbox = QtWidgets.QCheckBox(f"Channel {channel}")
            rename_edit = QtWidgets.QLineEdit(self)
            rename_edit.setPlaceholderText("Enter new name (optional)")

            self.checkboxes[channel] = checkbox
            self.rename_edits[channel] = rename_edit

            channel_layout.addWidget(checkbox, i, 0)
            channel_layout.addWidget(rename_edit, i, 1)

        self.layout.addLayout(channel_layout)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def get_selected_channels(self):
        for channel, checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                custom_name = (
                    self.rename_edits[channel].text()
                    if self.rename_edits[channel].text()
                    else f"Channel {channel}"
                )
                self.selected_channels[channel] = custom_name
        return self.selected_channels


class ColorSelection(QtWidgets.QWidget):
    color_chosen = QtCore.pyqtSignal(str)
    colors: list[str]

    def __init__(self, colors: tuple[str] | None = None) -> None:
        super().__init__()
        if colors is None:
            colors = (
                "brown",
                "red",
                "green",
                "blue",
                "orange",
                "purple",
                "pink",
                "black",
            )

        self.colors = colors

        color_combo = self.create_color_combo()
        self.color_indicator = QtWidgets.QLabel()
        self.color_indicator.setFixedSize(20, 20)
        self.choose_color(0)

        color_combo.currentIndexChanged.connect(self.choose_color)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(color_combo)
        layout.addWidget(self.color_indicator)

        self.setLayout(layout)

    def create_color_combo(self) -> None:
        color_combo = QtWidgets.QComboBox()
        color_model = QtGui.QStandardItemModel(color_combo)

        for color in self.colors:
            color_item = QtGui.QStandardItem()
            color_item.setBackground(QtGui.QColor(color))
            color_item.setText("")
            color_model.appendRow(color_item)

        color_combo.setModel(color_model)
        color_combo.setStyleSheet(
            """
            QComboBox::item {
                background: transparent;
            }
            QComboBox::item:selected {
                background: transparent;
            }
            """
        )

        return color_combo

    def choose_color(self, color_idx: int) -> None:
        color = self.colors[color_idx]
        self.color_indicator.setStyleSheet(
            f"background-color: {color}; border: 1px solid black;"
        )
        self.color_chosen.emit(color)


class TreeWidgetItem(QtWidgets.QTreeWidgetItem):
    id: int
    parent: QtWidgets.QTreeWidget

    def __init__(self, parent: QtWidgets.QTreeWidget, id: int = 0) -> None:
        super().__init__(parent)

        self.id = id
        self.parent = parent

        self.create_widgets()
        self.lay_out_widgets()
        self.setup_signals()

    def create_widgets(self) -> None:
        self._curve_type = QtWidgets.QComboBox()
        self.ema_type = QtWidgets.QPushButton(f"Button {self.id+1},{2}")
        self.color_selection = ColorSelection()
        self.panel_choice = QtWidgets.QComboBox()
        self.visibility_checkbox = QtWidgets.QCheckBox()
        self._derivation_type = QtWidgets.QComboBox()

        self._curve_type.addItems(
            ["Choose", "Mod_Cepstr", "F1", "F2", "F3", "F0", "ENV_AMP"]
        )
        self.ema_type.setStyleSheet(
            "background-color: lightblue; border: 1px solid black; padding: 5px"
        )
        self.panel_choice.addItems(["1", "2", "3", "4"])
        self.visibility_checkbox.setChecked(True)
        self._derivation_type.addItems(
            ["Traj. (f(x))", "vel. (f(x)')", "acc. (f(x)'')"]
        )

    def lay_out_widgets(self) -> None:
        # Affecte les widgets aux colonnes correspondantes (sauter la colonne EMA)
        self.parent.setItemWidget(self, 0, self._curve_type)
        # self.parent.setItemWidget(self, 1, self.ema_type)  # Supprimer cette ligne
        self.parent.setItemWidget(self, 1, self.color_selection)  # Décale la colonne
        self.parent.setItemWidget(self, 2, self.panel_choice)
        self.parent.setItemWidget(self, 3, self.visibility_checkbox)
        self.parent.setItemWidget(self, 4, self._derivation_type)



    def setup_signals(self) -> None:
        self.curve_type_changed = self._curve_type.currentIndexChanged
        # self.ema_type_changed = self.ema_type.currentIndexChanged
        self.color_changed = self.color_selection.color_chosen
        self.panel_changed = self.panel_choice.currentIndexChanged
        self.visibility_changed = self.visibility_checkbox.stateChanged
        self.derivation_type_changed = self._derivation_type.currentIndexChanged

    @property
    def curve_type(self) -> int:
        return self._curve_type.currentIndex()

    @property
    def selected_panel(self) -> int:
        return self.panel_choice.currentIndex()

    @property
    def derivation_type(self) -> int:
        return self._derivation_type.currentIndex()


class Dashboard(QtWidgets.QTreeWidget):
    curve_type_changed = QtCore.pyqtSignal(int, int)
    color_changed = QtCore.pyqtSignal(int, str)
    panel_changed = QtCore.pyqtSignal(int, int)
    visibility_changed = QtCore.pyqtSignal(int, int)
    derivation_type_changed = QtCore.pyqtSignal(int, int)

    update_curve = QtCore.pyqtSignal(int, int, int)

    row_count: int
    headers: list[str]

    def __init__(self, custom_curves) -> None:
        super().__init__()
        self.custom_curves = custom_curves
        self.row_count = 0
        # Supprime "EMA" de la liste des en-têtes
        self.headers = ["Curves", "Color", "Panel", "Show", "Derivative"]

        self.setColumnCount(len(self.headers))
        self.setHeaderLabels(self.headers)
        self.resize_column()

    def update_curve_choices(self, item):
        item._curve_type.addItems(
            ["Choose", "Mod_Cepstr", "F1", "F2", "F3", "F0", "ENV_AMP"]
        )

        for custom_curve_name in self.custom_curves:
            item._curve_type.addItem(custom_curve_name)
        # for _ in range(4):
        #     self.append_row()
    def resize_column(self) -> None:
        # Ajustez les indices des colonnes maintenant qu'il n'y a plus d'EMA
        self.setColumnWidth(self.headers.index("Color"), 90)
        self.setColumnWidth(self.headers.index("Panel"), 50)
        self.setColumnWidth(self.headers.index("Show"), 20)
    def _update_curve(self, item: TreeWidgetItem) -> None:
        self.update_curve.emit(item.id, item.curve_type, item.derivation_type)

    def append_row(self) -> None:
        item = TreeWidgetItem(self, self.row_count)
        item.curve_type_changed.connect(lambda _: self._update_curve(item))
        item.derivation_type_changed.connect(lambda _: self._update_curve(item))

        item.color_changed.connect(
            lambda color, row=item.id: self.color_changed.emit(row, color)
        )
        item.panel_changed.connect(
            lambda index, row=item.id: self.panel_changed.emit(row, index)
        )
        item.visibility_changed.connect(
            lambda state, row=item.id: self.visibility_changed.emit(row, state)
        )

        self.addTopLevelItem(item)
        self.row_count += 1



    def selected_panel(self, row_idx: int) -> int:
        if row_idx < 0 or row_idx >= self.row_count:
            raise ValueError(f"Incorrect row id given {row_idx}")
    def reset(self) -> None:
        # Iterate through the items and remove them
        for i in reversed(range(self.topLevelItemCount())):
            item = self.takeTopLevelItem(i)
            del item  # Ensure the item is deleted from memory

        # Reset the row count
        self.row_count = 0
        print("Dashboard reset complete.")
        
class DashboardWidget(QtWidgets.QWidget):
    dashboard: Dashboard
    row_added = QtCore.pyqtSignal(int)

    def __init__(self, custom_curves) -> None:
        super().__init__()

        self.dashboard = Dashboard(custom_curves)

        add_row_button = StyledButton("+", "lightgreen")
        add_row_button.clicked.connect(self._row_added)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.dashboard)
        layout.addWidget(add_row_button)

        self.setLayout(layout)

    def _row_added(self) -> None:
        self.dashboard.append_row()
        self.row_added.emit(self.dashboard.row_count)

    def reset(self) -> None:
        print("Resetting dashboard widget...")
        self.dashboard.reset()
        print("Dashboard widget reset complete.")

class FileLoadIndicator(QtWidgets.QGroupBox):

    def __init__(
        self, title: str, default_text: str, color: str, *args, **kargs
    ) -> None:
        super().__init__(title, *args, **kargs)

        self.setStyleSheet(
            """
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
        """
        )

        layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel(default_text)
        self.label.setWordWrap(True)
        self.label.setStyleSheet(f"font-size: 16px; color: {color};")

        layout.addWidget(self.label)

        self.setLayout(layout)

    def file_loaded(self, file_path: str) -> None:
        file_name = os.path.basename(file_path)
        self.label.setText(file_name)


class StyledButton(QtWidgets.QPushButton):

    def __init__(self, text: str, color: str = "lightblue", *args, **kargs) -> None:
        super().__init__(text, *args, **kargs)
        self.setStyleSheet(
            f"background-color: {color}; border: 1px solid black; padding: 5px"
        )


class TierSelection(QtWidgets.QGroupBox):
    button_group: QtWidgets.QButtonGroup
    tier_checked = QtCore.pyqtSignal(str)
    tier_clear = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__("Select TextGrid Tier")

        layout = QtWidgets.QVBoxLayout()

        self.button_group = QtWidgets.QButtonGroup(self)
        self.button_group.setExclusive(True)
        self.button_group.buttonToggled.connect(self._tier_checked)

        self.no_tier_btn = QtWidgets.QRadioButton("None")
        self.button_group.addButton(self.no_tier_btn)

        self.setLayout(layout)

        self.layout().addWidget(self.no_tier_btn)

    def set_data(self, data: tgt.io.TextGrid) -> None:
        self.reset()

        self.populate_textgrid_selection(data.get_tier_names())

    def populate_textgrid_selection(self, tiers: list[str]) -> None:
        for tier_name in tiers:
            btn = QtWidgets.QRadioButton(tier_name)

            self.button_group.addButton(btn)
            self.layout().addWidget(btn)

    def _tier_checked(self, button: QtWidgets.QRadioButton, checked: bool) -> None:
        if not checked:
            return

        if button is self.no_tier_btn:
            self.tier_clear.emit()
            return

        tier_name = button.text()
        self.tier_checked.emit(tier_name)

    def reset(self) -> None:
        layout = self.layout()

        for btn in self.button_group.buttons():
            if btn is self.no_tier_btn:
                continue

            layout.removeWidget(btn)
            self.button_group.removeButton(btn)

            btn.deleteLater()



class ManualPointManagement(QtWidgets.QToolBar):
    # Define custom signals
    panel_changed: QtCore.pyqtSignal = QtCore.pyqtSignal(int)
    checkbox_toggled: QtCore.pyqtSignal = QtCore.pyqtSignal(bool)
    operation_changed: QtCore.pyqtSignal = QtCore.pyqtSignal(int)
    min_analysis_clicked: QtCore.pyqtSignal = QtCore.pyqtSignal()
    max_analysis_clicked: QtCore.pyqtSignal = QtCore.pyqtSignal()
    export_to_csv_clicked: QtCore.pyqtSignal = QtCore.pyqtSignal()

    def __init__(
        self, panel_nb: int = 4, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)

        self.panel_nb = panel_nb

        self.panel_selector: QtWidgets.QComboBox = QtWidgets.QComboBox(self)
        self.add_min_action: QtWidgets.QAction = QtWidgets.QAction("Analyze Min", self)
        self.add_max_action: QtWidgets.QAction = QtWidgets.QAction("Analyze Max", self)
        self.export_to_csv_action: QtWidgets.QAction = QtWidgets.QAction(
            "Export to CSV", self
        )

        self.enable_checkbox: QtWidgets.QCheckBox = QtWidgets.QCheckBox(
            "Manual management", self
        )
        self.operation_selector: QtWidgets.QComboBox = QtWidgets.QComboBox(self)

        self.panel_selector.addItems([f"Panel {i+1}" for i in range(self.panel_nb)])
        self.operation_selector.addItem("Add min", PointOperation.ADD_MIN)
        self.operation_selector.addItem("Add max", PointOperation.ADD_MAX)
        self.operation_selector.addItem("Remove point", PointOperation.REMOVE)  # New operation type

        self.panel_selector.currentIndexChanged.connect(self.on_panel_changed)
        self.add_min_action.triggered.connect(self.on_add_min_clicked)
        self.add_max_action.triggered.connect(self.on_add_max_clicked)
        self.export_to_csv_action.triggered.connect(self.on_export_to_csv_clicked)

        self.addWidget(self.enable_checkbox)
        self.addWidget(self.operation_selector)
        self.addSeparator()
        self.addWidget(self.panel_selector)
        self.addAction(self.add_min_action)
        self.addAction(self.add_max_action)
        self.addAction(self.export_to_csv_action)

    def on_panel_changed(self, index: int) -> None:
        self.panel_changed.emit(index)

    def on_add_min_clicked(self) -> None:
        self.min_analysis_clicked.emit()

    def on_add_max_clicked(self) -> None:
        self.max_analysis_clicked.emit()

    def on_export_to_csv_clicked(self) -> None:
        self.export_to_csv_clicked.emit()

    @property
    def is_enabled(self) -> bool:
        return self.enable_checkbox.isChecked()

    @property
    def operation(self) -> PointOperation:
        return self.operation_selector.currentData()

    @property
    def panel(self) -> int:
        return self.panel_selector.currentIndex()

class DataSource(ABC):
    """
    Defines the interface for the curve data calculation.
    """

    @abstractmethod
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        @Returns x_values, y_values
        """
        pass


class Transformation(ABC):

    @abstractmethod
    def transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str,
        width: int,
        accOrder: int,
        polyOrder: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass


class Trajectory(Transformation):

    @override
    def transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str,
        width: int,
        accOrder: int,
        polyOrder: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return x, y


class Velocity(Transformation):

    @override
    def transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str,
        width: int,
        accOrder: int,
        polyOrder: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        velocity = get_velocity(
            y,
            sr=1.0,
            difference=1,
            method=method,
            width=width,
            accOrder=accOrder,
            polyOrder=polyOrder,
        )
        return x, velocity


class Acceleration(Transformation):

    @override
    def transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str,
        width: int,
        accOrder: int,
        polyOrder: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        acceleration = get_velocity(
            y,
            sr=1.0,
            difference=2,
            method=method,
            width=width,
            accOrder=accOrder,
            polyOrder=polyOrder,
        )
        return x, acceleration


class Soundwave(DataSource):

    @override
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        a = Parselmouth(audio_path)
        s = a.get_sound()

        return s.timestamps, s.amplitudes[0]


class Mfcc(DataSource):

    @override
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        data = load_channel(audio_path)
        print(audio_path)
        sig_sr = 10000
        channel_n = 0
        t_step = 0.005
        win_len = 0.025
        n_mfcc = 13
        n_fft = 512
        min_freq = 100
        max_freq = 10000
        remove_first = 1
        filt_cutoff = 12
        filt_ord = 6
        diff_method = "grad"
        out_filter = "iir"
        out_filt_type = "low"
        out_filt_cutoff = [12]
        out_filt_len = 6
        out_filt_poly_ord = 3

        y, x = get_MFCCS_change(
            audio_path,
            sig_sr,
            channelN=channel_n,
            tStep=t_step,
            winLen=win_len,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            minFreq=min_freq,
            maxFreq=max_freq,
            removeFirst=remove_first,
            filtCutoff=filt_cutoff,
            filtOrd=filt_ord,
            diffMethod=diff_method,
            outFilter=out_filter,
            outFiltType=out_filt_type,
            outFiltCutOff=out_filt_cutoff,
            outFiltLen=out_filt_len,
            outFiltPolyOrd=out_filt_poly_ord,
        )

        return x, y


class Formant1(DataSource):

    @override
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        f_times, f1_values, _, _ = calc_formants(
            parselmouth.Sound(audio_path), 0, 99999, 20
        )
        return f_times, f1_values


class Formant2(DataSource):

    @override
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        f_times, _, f2_values, _ = calc_formants(
            parselmouth.Sound(audio_path), 0, 99999, 20
        )
        return f_times, f2_values


class Formant3(DataSource):

    @override
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        f_times, _, _, f3_values = calc_formants(
            parselmouth.Sound(audio_path), 0, 99999, 20
        )
        return f_times, f3_values


class F0(DataSource):

    @override
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        sig_sr, audio_data = wavfile.read(audio_path)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        method = "praatac"
        hop_size = 0.005
        min_pitch = 75
        max_pitch = 600
        interp_unvoiced = "linear"
        out_filter = "iir"
        out_filt_type = "low"
        out_filt_cutoff = [12]
        out_filt_len = 6
        out_filt_poly_ord = 3

        f0, f0_times = get_f0(
            audio_data,
            sig_sr,
            method=method,
            hopSize=hop_size,
            minPitch=min_pitch,
            maxPitch=max_pitch,
            interpUnvoiced=interp_unvoiced,
            outFilter=out_filter,
            outFiltType=out_filt_type,
            outFiltCutOff=out_filt_cutoff,
            outFiltLen=out_filt_len,
            outFiltPolyOrd=out_filt_poly_ord,
        )
        return f0_times, f0


class AmplitudeEnvelope(DataSource):

    @override
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        sample_rate, audio_signal = wavfile.read(audio_path)

        # audio_signal = audio_signal[int(start * sample_rate):int(end * sample_rate)]
        amplitude_envelope, time_axis = calculate_amplitude_envelope(
            audio_signal, sample_rate
        )

        return time_axis, amplitude_envelope


class Plotter(ABC):

    def __init__(self, toolbar: "ManualPointManagement") -> None:
        self.toolbar = toolbar

    @abstractmethod
    def plot(self, x: np.ndarray, y: np.ndarray) -> CalculationValues:
        pass

class CurvePlotter(Plotter):

    @override
    def plot(self, x: np.ndarray, y: np.ndarray) -> CalculationValues:
        curve = pg.PlotDataItem(x=x, y=y)
        min = pg.ScatterPlotItem()
        max = pg.ScatterPlotItem()

        return CalculationValues(curve, min, max, self.toolbar)


class ScatterPlotPlotter(Plotter):

    @override
    def plot(self, x: np.ndarray, y: np.ndarray) -> CalculationValues:
        curve = pg.ScatterPlotItem(x=x, y=y)
        min = pg.ScatterPlotItem()
        max = pg.ScatterPlotItem()

        return CalculationValues(curve, min, max, self.toolbar)


class CurveGenerator:
    datasources: list[DataSource]
    derivations: list[Transformation]
    plotters: list[Plotter]

    def __init__(self, toolbar : "ManualPointManagement") -> None:
        self.toolbar = toolbar
        self.datasources = [
            None,
            Mfcc(),
            Formant1(),
            Formant2(),
            Formant3(),
            F0(),
            AmplitudeEnvelope(),
        ]
        self.derivations = [Trajectory(), Velocity(), Acceleration()]
        self.plotters = [
            None,
            CurvePlotter(self.toolbar),
            ScatterPlotPlotter(self.toolbar),
            ScatterPlotPlotter(self.toolbar),
            ScatterPlotPlotter(self.toolbar),
            CurvePlotter(self.toolbar),
            CurvePlotter(self.toolbar),
                        CurvePlotter(self.toolbar),
                                    CurvePlotter(self.toolbar),

            
        ]

    def generate(
        self, audio_path: str, curve_type_id: int, curve_derivation: int
    ) -> CalculationValues:
        if curve_type_id < 0 or curve_type_id >= len(self.datasources):
            raise IndexError("Curve type ID is out of range")

        source = self.datasources[curve_type_id]

        if source is None:
            raise ValueError("Invalid data source for the given curve type ID")

        operation = self.derivations[curve_derivation]
        plotter = self.plotters[curve_type_id]

        data = source.calculate(audio_path)
        derivative_method = "gradient"
        sg_width = 3
        fin_diff_acc_order = 2
        sg_poly_order = 2

        x, y = operation.transform(
            *data,
            method=derivative_method,
            width=sg_width,
            accOrder=fin_diff_acc_order,
            polyOrder=sg_poly_order,
        )

        return plotter.plot(x, y)

    def generate_custom_formant2(
        self, audio_path: str, params: dict, derivation_id: int
    ) -> CalculationValues:
        sound = parselmouth.Sound(audio_path)
        f_times, _, f2_values, _ = calc_formants(
            sound,
            0,
            99999,
            energy_threshold=params["energy_threshold"],
            time_step=params["time_step"],
            max_number_of_formants=params["max_num_formants"],
            maximum_formant=params["max_formant"],
            window_length=params["window_length"],
            pre_emphasis_from=params["pre_emphasis_from"],
        )

        operation = self.derivations[derivation_id]
        x, y = operation.transform(
            f_times,
            f2_values,
            params["derivative_method"],
            params["sg_width"],
            params["fin_diff_acc_order"],
            params["sg_poly_order"],
        )

        plotter = ScatterPlotPlotter(self.toolbar)
        return plotter.plot(x, y)

    def generate_custom_formant3(
        self, audio_path: str, params: dict, derivation_id: int
    ) -> CalculationValues:
        sound = parselmouth.Sound(audio_path)
        f_times, _, _, f3_values = calc_formants(
            sound,
            0,
            99999,
            energy_threshold=params["energy_threshold"],
            time_step=params["time_step"],
            max_number_of_formants=params["max_num_formants"],
            maximum_formant=params["max_formant"],
            window_length=params["window_length"],
            pre_emphasis_from=params["pre_emphasis_from"],
        )

        operation = self.derivations[derivation_id]
        x, y = operation.transform(
            f_times,
            f3_values,
            params["derivative_method"],
            params["sg_width"],
            params["fin_diff_acc_order"],
            params["sg_poly_order"],
        )

        plotter = ScatterPlotPlotter(self.toolbar)
        return plotter.plot(x, y)

    def generate_custom_formant1(
        self, audio_path: str, params: dict, derivation_id: int
    ) -> CalculationValues:
        sound = parselmouth.Sound(audio_path)
        f_times, f1_values, _, _ = calc_formants(
            sound,
            0,
            99999,
            energy_threshold=params["energy_threshold"],
            time_step=params["time_step"],
            max_number_of_formants=params["max_num_formants"],
            maximum_formant=params["max_formant"],
            window_length=params["window_length"],
            pre_emphasis_from=params["pre_emphasis_from"],
        )

        operation = self.derivations[derivation_id]
        x, y = operation.transform(
            f_times,
            f1_values,
            params["derivative_method"],
            params["sg_width"],
            params["fin_diff_acc_order"],
            params["sg_poly_order"],
        )

        plotter = ScatterPlotPlotter(self.toolbar)
        return plotter.plot(x, y)

    def generate_custom_mfcc(
        self, audio_path: str, params: dict, derivation_id: int
    ) -> CalculationValues:
        y, x = get_MFCCS_change(
            audio_path,
            params["signal_sample_rate"],
            channelN=0,
            tStep=params["tStep"],
            winLen=params["winLen"],
            n_mfcc=params["n_mfcc"],
            n_fft=params["n_fft"],
            removeFirst=params["removeFirst"],
            filtCutoff=params["filtCutoff"],
            filtOrd=params["filtOrd"],
            diffMethod=params["diffMethod"],
            outFilter=params["outFilter"],
            outFiltType=params["outFiltType"],
            outFiltCutOff=params["outFiltCutOff"],
            outFiltLen=params["outFiltLen"],
            outFiltPolyOrd=params["outFiltPolyOrd"],
        )
        operation = self.derivations[derivation_id]
        x, y = operation.transform(
            x,
            y,
            params["derivative_method"],
            params["sg_width"],
            params["fin_diff_acc_order"],
            params["sg_poly_order"],
        )

        plotter = CurvePlotter(self.toolbar)
        return plotter.plot(x, y)

    def generate_custom_amplitude(
        self, audio_path: str, params: dict, derivation_id: int
    ) -> CalculationValues:
        sample_rate, audio_signal = wavfile.read(audio_path)
        amplitude, time_axis = calculate_amplitude_envelope(
            audio_signal,
            sample_rate,
            method=params["method"],
            winLen=params["winLen"],
            hopLen=params["hopLen"],
            center=params["center"],
            outFilter=params["outFilter"],
            outFiltType=params["outFiltType"],
            outFiltCutOff=params["outFiltCutOff"],
            outFiltLen=params["outFiltLen"],
            outFiltPolyOrd=params["outFiltPolyOrd"],
        )

        operation = self.derivations[derivation_id]
        time_axis, amplitude = operation.transform(
            time_axis,
            amplitude,
            params["derivative_method"],
            params["sg_width"],
            params["fin_diff_acc_order"],
            params["sg_poly_order"],
        )

        plotter = CurvePlotter(self.toolbar)
        return plotter.plot(time_axis, amplitude)

    def generate_custom_f0(
        self, audio_path: str, params: dict, derivation_id: int
    ) -> CalculationValues:
        sig_sr, audio_data = wavfile.read(audio_path)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        f0, f0_times = get_f0(
            audio_data,
            sig_sr,
            method=params["method"],
            hopSize=params["hopSize"],
            minPitch=params["minPitch"],
            maxPitch=params["maxPitch"],
            interpUnvoiced=params["interpUnvoiced"],
            outFilter=params["outFilter"],
            outFiltType=params["outFiltType"],
            outFiltCutOff=params["outFiltCutOff"],
            outFiltLen=params["outFiltLen"],
            outFiltPolyOrd=params["outFiltPolyOrd"],
        )

        operation = self.derivations[derivation_id]
        x, y = operation.transform(
            f0_times,
            f0,
            params["derivative_method"],
            params["sg_width"],
            params["fin_diff_acc_order"],
            params["sg_poly_order"],
        )

        plotter = CurvePlotter(self.toolbar)
        return plotter.plot(x, y)


class MainWindow(QtWidgets.QMainWindow):
    audio_path: str | None
    audio_widget: SoundInformation

    annotation_path: str | None
    annotation_data: tgt.core.TextGrid | None
    annotation_widget: DisplayInterval

    panels: list[PanelWidget]
    curves: dict[int, list[int, int]]

    def __init__(self) -> None:
        super().__init__()
        nb_panels = 4
        self.selected_max_peaks = {}  # Dictionnaire pour stocker les pics max sélectionnés par panel et axis_id
        self.selected_min_peaks = {}  # Dictionnaire pour stocker les pics min sélectionnés par panel et axis_id

        self.init_main_layout()
        self.custom_curves = {}
        self.audio_path = None
        self.audio_widget = SoundInformation()

        self.annotation_path = None
        self.annotation_data = None
        self.annotation_widget = DisplayInterval(self.audio_widget)

        self.point_management_toolbar = ManualPointManagement(nb_panels)
        self.curve_generator = CurveGenerator(self.point_management_toolbar)
        self.dashboard_widget = DashboardWidget(self.custom_curves)
        self.zoom = ZoomToolbar(self.audio_widget.selection_region)

        self.audio_indicator = FileLoadIndicator(
            "Loaded Audio", "No audio Loaded", "blue"
        )
        self.annotation_indicator = FileLoadIndicator(
            "Loaded TextGrid", "No textGrid loaded", "red"
        )
        self.tier_selection = TierSelection()
        self.config_mfcc_button = StyledButton("Configure")
        self.custom_curves = {}
        self.tier_selection.tier_checked.connect(
            lambda tier_name: self.annotation_widget.display(
                self.annotation_data.get_tier_by_name(tier_name)
            )
        )
        self.tier_selection.tier_clear.connect(self.annotation_widget.clear)
        self.config_mfcc_button.clicked.connect(self.open_config)

        self.dashboard_widget.row_added.connect(self.handle_new_row)

        self.dashboard_widget.dashboard.update_curve.connect(self.update_curve)
        self.dashboard_widget.dashboard.color_changed.connect(self.change_curve_color)
        self.dashboard_widget.dashboard.panel_changed.connect(self.change_curve_panel)
        self.dashboard_widget.dashboard.visibility_changed.connect(
            self.change_curve_visibility
        )

        self.add_control_widget(self.audio_indicator)
        self.add_control_widget(self.annotation_indicator)
        self.add_control_widget(self.create_load_buttons())
        self.add_control_widget(self.create_audio_control_buttons())
        self.add_control_widget(self.create_spectrogram_checkbox())
        self.add_control_widget(self.tier_selection)
        self.add_control_widget(self.dashboard_widget)
        self.add_control_widget(self.config_mfcc_button)

        self.add_curve_widget(self.audio_widget)

        self.curves = {}
        self.panels = []
        self.custom_mfcc_params = {}
        self.custom_amplitude_params = {}
        self.custom_formant1_params = {}
        self.custom_formant2_params = {}
        self.custom_formant3_params = {}
        self.custom_f0_params = {}

        for i in range(nb_panels):
            panel_widget = PanelWidget(i + 1)

            self.zoom.link_viewbox(panel_widget.panel)

            self.add_curve_widget(panel_widget)
            self.panels.append(panel_widget)

        self.add_curve_widget(self.zoom)

        # self.add_control_widget(self.create_analysis_controls())
        self.add_control_widget(self.point_management_toolbar)
        self.point_management_toolbar.min_analysis_clicked.connect(
            self.analyze_min_peaks
        )
        self.point_management_toolbar.max_analysis_clicked.connect(
            self.analyze_max_peaks
        )
        self.point_management_toolbar.export_to_csv_clicked.connect(self.export_to_csv)

        self.recording = False
        self.frames = []
        self.recorded_audio = []
        self.stream = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)
        self.playing = False
        self.audio_cursor = pg.LinearRegionItem()
        self.audio_cursor.setBrush(pg.mkBrush(0, 0, 255, 150))
        self.audio_widget.sound_plot.addItem(self.audio_cursor)
        self.audio_cursor.hide()

    def init_main_layout(self) -> None:
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QHBoxLayout(central_widget)

        self.curve_column_layout = QtWidgets.QVBoxLayout()
        curve_column_widget = QtWidgets.QWidget()
        curve_column_widget.setLayout(self.curve_column_layout)

        self.control_column_layout = QtWidgets.QVBoxLayout()
        control_column_widget = QtWidgets.QWidget()
        control_column_widget.setLayout(self.control_column_layout)

        main_layout.addWidget(curve_column_widget, 3)
        main_layout.addWidget(control_column_widget, 2)


    def create_load_buttons(self) -> QtWidgets.QGroupBox:
        load_group_box = QtWidgets.QGroupBox("Load Audio, TextGrid and POS")
        load_layout = QtWidgets.QVBoxLayout()

        load_audio_button = StyledButton("Load Audio")
        load_textgrid_button = StyledButton("Load TextGrid")
        load_pos_button = StyledButton("Load POS File")
        self.record_button = StyledButton("Record Audio", "lightgreen")

        load_audio_button.clicked.connect(self.load_audio)
        load_textgrid_button.clicked.connect(self.load_annotations)
        load_pos_button.clicked.connect(self.load_pos_file)
        self.record_button.clicked.connect(self.toggle_recording)

        load_layout.addWidget(load_audio_button)
        load_layout.addWidget(load_textgrid_button)
        load_layout.addWidget(load_pos_button)
        load_layout.addWidget(self.record_button)

        load_group_box.setLayout(load_layout)
        return load_group_box

    def load_pos_file(self) -> None:
        pos_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open POS File", "", "POS Files (*.pos)"
        )

        if not pos_path:
            return

        self.pos_data = read_AG50x(pos_path)
        self.pos_channels = self.pos_data.channels.values

        dialog = POSChannelSelectionDialog(self.pos_channels, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected_channels = dialog.get_selected_channels()

            self.add_pos_channels_to_dashboard(selected_channels)

    def add_pos_channels_to_dashboard(self, selected_channels: dict) -> None:
        for original_channel_id, custom_name in selected_channels.items():
            channel_id = int(original_channel_id)
            channel_name = custom_name

            self.custom_curves[channel_name] = {
                "generator_function": self.generate_pos_curve,
                "params": {"channel_id": channel_id},
            }
            for i in range(self.dashboard_widget.dashboard.topLevelItemCount()):
                item = self.dashboard_widget.dashboard.topLevelItem(i)
                item._curve_type.addItem(channel_name)

    def generate_pos_curve(
        self, audio_path: str, params: dict, derivation_id: int
    ) -> CalculationValues:
        channel_id = params["channel_id"]
        pos_data = self.pos_data.ema.sel(channels=channel_id)

        time_axis = pos_data.time.values
        y_values = pos_data.sel(dimensions="z").values

        operation = self.curve_generator.derivations[derivation_id]
        x, y = operation.transform(time_axis, y_values, "gradient", 3, 2, 2)

        # Corrected line:
        plotter = CurvePlotter(self.point_management_toolbar)
        return plotter.plot(x, y)

    def add_curve_widget(self, widget: QtWidgets.QWidget) -> None:
        viewbox = None

        if isinstance(widget, (pg.PlotWidget, pg.PlotItem)):
            viewbox = widget.getViewBox()
        elif isinstance(widget, PanelWidget):
            viewbox = widget.panel.getViewBox()

        if viewbox is not None:
            viewbox.setXLink(self.audio_widget.reference_viewbox)

        self.curve_column_layout.addWidget(widget)

    def add_control_widget(self, widget: QtWidgets.QWidget) -> None:
        self.control_column_layout.addWidget(widget)

    def create_audio_control_buttons(self) -> QtWidgets.QGroupBox:
        audio_control_group_box = QtWidgets.QGroupBox("Audio Control")
        audio_control_layout = QtWidgets.QVBoxLayout()

        play_button = StyledButton("Play Selected Region")

        play_button.clicked.connect(self.play_selected_region)

        audio_control_layout.addWidget(play_button)

        audio_control_group_box.setLayout(audio_control_layout)
        return audio_control_group_box

    def create_analysis_controls(self) -> QtWidgets.QWidget:
        analysis_controls_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()

        self.panel_selector = QtWidgets.QComboBox()
        self.panel_selector.addItems(
            [f"Panel {i + 1}" for i in range(len(self.panels))]
        )

        self.analyze_max_button = QtWidgets.QPushButton("Analyze Max Peaks")
        self.analyze_min_button = QtWidgets.QPushButton("Analyze Min Peaks")
        self.export_csv_button = QtWidgets.QPushButton("Export CSV")
        self.export_csv_button.clicked.connect(self.export_to_csv)
        self.analyze_max_button.clicked.connect(self.analyze_max_peaks)
        self.analyze_min_button.clicked.connect(self.analyze_min_peaks)

        layout.addWidget(QtWidgets.QLabel("Select Panel:"))
        layout.addWidget(self.panel_selector)
        layout.addWidget(self.analyze_max_button)
        layout.addWidget(self.analyze_min_button)
        layout.addWidget(self.export_csv_button)
        analysis_controls_widget.setLayout(layout)
        return analysis_controls_widget
    def export_to_csv(self):
        panel = self.panels[self.point_management_toolbar.panel].panel
        axis_ids = list(panel.rotation.keys())
        curve_names = []

        for i in range(self.dashboard_widget.dashboard.topLevelItemCount()):
            item = self.dashboard_widget.dashboard.topLevelItem(i)
            curve_name = item._curve_type.currentText()
            if i < len(axis_ids):
                curve_names.append(curve_name)

        if self.annotation_data:
            tier_names = self.annotation_data.get_tier_names()
            export_dialog = ExportCSVDialog(axis_ids, curve_names, tier_names, self)
        else:
            export_dialog = ExportCSVDialog(axis_ids, curve_names, parent=self)

        if export_dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected_data = export_dialog.get_selections()
            selected_tiers = export_dialog.get_selected_tiers()
            calculation_choices = export_dialog.get_calculation_choices()

            csv_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
            if not csv_path:
                return

            self.save_curves_to_csv(panel, selected_data, csv_path, axis_ids, curve_names, selected_tiers, calculation_choices)
    def save_curves_to_csv(self, panel, selected_data, csv_path, axis_ids, curve_names, selected_tiers=None, calculation_choices=None):
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            headers = []
            csv_data = {}

            # Add curve data
            for idx, axis_id in enumerate(axis_ids):
                curve_name = curve_names[idx]
                if curve_name not in selected_data:
                    continue

                options = selected_data[curve_name]
                axis = panel.rotation[axis_id]
                x_data, y_data = axis.curve.getData() if axis.curve is not None else ([], [])

                if options["x"]:
                    headers.append(f"{curve_name} X")
                if options["y"]:
                    headers.append(f"{curve_name} Y")

                # Populate csv_data for each curve
                for i, x in enumerate(x_data):
                    if i not in csv_data:
                        csv_data[i] = {}

                    if options["x"]:
                        csv_data[i][f"{curve_name} X"] = x
                    if options["y"]:
                        csv_data[i][f"{curve_name} Y"] = y_data[i]

                # Add Min and Max Peaks if selected
                if options["min"]:
                    min_peaks = [(p.pos().x(), p.pos().y()) for p in axis.min.points()]
                    headers.extend([f"Min Peak {curve_name} X", f"Min Peak {curve_name} Y"])
                    for i, (x, y) in enumerate(min_peaks):
                        csv_data[i][f"Min Peak {curve_name} X"] = x
                        csv_data[i][f"Min Peak {curve_name} Y"] = y

                if options["max"]:
                    max_peaks = [(p.pos().x(), p.pos().y()) for p in axis.max.points()]
                    headers.extend([f"Max Peak {curve_name} X", f"Max Peak {curve_name} Y"])
                    for i, (x, y) in enumerate(max_peaks):
                        csv_data[i][f"Max Peak {curve_name} X"] = x
                        csv_data[i][f"Max Peak {curve_name} Y"] = y

                # Calculate the mean for the selected region or tier for this curve
                if calculation_choices and calculation_choices["calculate_mean"]:
                    headers.append(f"{curve_name} Mean")

                    if calculation_choices["region_or_tier"] == "Region Selection":
                        # Calculate mean based on the selected region
                        region = self.audio_widget.selection_region.getRegion()
                        region_start, region_end = region
                        region_y_values = [
                            y for x, y in zip(x_data, y_data) if region_start <= x <= region_end
                        ]
                        mean_value = np.mean(region_y_values) if region_y_values else 0
                        csv_data[0][f"{curve_name} Mean"] = mean_value

                    else:
                        # Calculate mean for the selected TextGrid tier
                        tier_name = calculation_choices["region_or_tier"]
                        tier = self.annotation_data.get_tier_by_name(tier_name)
                        interval_means = []

                        for interval in tier.intervals:
                            interval_start, interval_end = interval.start_time, interval.end_time
                            interval_y_values = [
                                y for x, y in zip(x_data, y_data) if interval_start <= x <= interval_end
                            ]
                            interval_mean = np.mean(interval_y_values) if interval_y_values else 0
                            interval_means.append(interval_mean)

                        # Calculate overall mean for the intervals within the tier
                        average_mean = np.mean(interval_means) if interval_means else 0
                        csv_data[0][f"{curve_name} Mean"] = average_mean

            # Calculate duration for the region or tier if needed
            if calculation_choices and calculation_choices["calculate_duration"]:
                headers.append("Duration")

                if calculation_choices["region_or_tier"] == "Region Selection":
                    # Calculate duration based on the selected region
                    region = self.audio_widget.selection_region.getRegion()
                    region_start, region_end = region
                    duration = region_end - region_start
                    csv_data[0]["Duration"] = duration

                else:
                    # Calculate duration for the selected TextGrid tier
                    tier_name = calculation_choices["region_or_tier"]
                    tier = self.annotation_data.get_tier_by_name(tier_name)
                    total_duration = sum(
                        interval.end_time - interval.start_time for interval in tier.intervals
                    )
                    csv_data[0]["Duration"] = total_duration

            # Write headers and data to the CSV file
            writer.writerow(headers)
            for i in sorted(csv_data.keys()):
                row = [csv_data[i].get(header, "") for header in headers]
                writer.writerow(row)

        QtWidgets.QMessageBox.information(self, "Export Successful", f"Data has been successfully exported to {csv_path}")

    def analyze_max_peaks(self) -> None:
        panel_id = self.point_management_toolbar.panel
        if panel_id < 0:
            return

        panel = self.panels[panel_id].panel

        region = self.audio_widget.selection_region.getRegion()
        region_start, region_end = region

        for axis_id, item in panel.rotation.items():
            calculated_curve: CalculationValues = item

            x_data, y_data = calculated_curve.curve.getData()

            region_mask = (x_data >= region_start) & (x_data <= region_end)

            x_data_region = x_data[region_mask]
            y_data_region = y_data[region_mask]

            peaks, _ = find_peaks(y_data_region)

            peak_x = x_data_region[peaks]
            peak_y = y_data_region[peaks]

            calculated_curve.max.setData(peak_x, peak_y)

            peak_info = "\n".join(
                [
                    f"Peak {i + 1}: X = {px}, Y = {py}"
                    for i, (px, py) in enumerate(zip(peak_x, peak_y))
                ]
            )
            QtWidgets.QMessageBox.information(
                self,
                "Peak Analysis",
                f"Peaks in Panel {panel_id + 1} (Selected Region):\n\n{peak_info}",
            )

    def analyze_min_peaks(self) -> None:
        panel_id = self.point_management_toolbar.panel
        if panel_id < 0:
            return

        panel = self.panels[panel_id].panel

        region = self.audio_widget.selection_region.getRegion()
        region_start, region_end = region

        for axis_id, item in panel.rotation.items():
            calculated_curve: CalculationValues = item

            x_data, y_data = calculated_curve.curve.getData()

            region_mask = (x_data >= region_start) & (x_data <= region_end)

            x_data_region = x_data[region_mask]
            y_data_region = y_data[region_mask]

            peaks, _ = find_peaks(-y_data_region)

            peak_x = x_data_region[peaks]
            peak_y = y_data_region[peaks]

            calculated_curve.min.setData(peak_x, peak_y)

            min_info = "\n".join(
                [
                    f"Minimum {i + 1}: X = {px}, Y = {py}"
                    for i, (px, py) in enumerate(zip(peak_x, peak_y))
                ]
            )
            QtWidgets.QMessageBox.information(
                self,
                "Minimum Peak Analysis",
                f"Minima in Panel {panel_id + 1} (Selected Region):\n\n{min_info}",
            )

    def create_spectrogram_checkbox(self) -> QtWidgets.QGroupBox:
        spectrogram_group_box = QtWidgets.QGroupBox("Select Spectrogram")
        spectrolayout = QtWidgets.QVBoxLayout()

        spectrogram_checkbox = QtWidgets.QCheckBox("Show/Hide Spectrogram")
        spectrolayout.addWidget(spectrogram_checkbox)

        spectrogram_group_box.setLayout(spectrolayout)
        spectrogram_checkbox.setChecked(False)
        spectrogram_checkbox.toggled.connect(self.audio_widget.toggle_spectrogram)

        return spectrogram_group_box

    def load_audio(self) -> None:
        audio_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav)"
        )

        if not audio_path:
            return
        self.dashboard_widget.reset()
        self.audio_indicator.file_loaded(audio_path)

        self.audio_path = audio_path

        self.audio_widget.set_data(Parselmouth(audio_path))

        self.reset_curves()
    def load_annotations(self) -> None:
        annotation_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open TextGrid File", "", "TextGrid Files (*.TextGrid)"
        )

        if not annotation_path:
            return

        self.annotation_indicator.file_loaded(annotation_path)

        self.annotation_path = annotation_path
        self.annotation_data = tgt.io.read_textgrid(annotation_path)

        self.tier_selection.set_data(self.annotation_data)

    def change_curve_panel(self, row_id: int, new_panel_id: int) -> None:
        if row_id not in self.curves:
            return

        curve, current_panel = self.curves.get(row_id, [None, None])
        new_panel = self.panels[new_panel_id]

        self.curves[row_id][1] = new_panel

        if curve is None:
            return

        if current_panel is not None:
            try:
                current_panel.panel.remove_curve(curve)
            except ValueError:
                pass

        new_panel.panel.add_curve(curve)

    def change_curve_color(self, row_id: int, new_color: str) -> None:
        curve, panel = self.curves.get(row_id, [None, None])

        if curve is None:
            return

        curve.curve.setPen(color=new_color)
        panel.panel.update_y_axis_color(curve, new_color)

    def change_curve_visibility(self, row_id: int, is_visible: bool) -> None:
        curve, _ = self.curves[row_id]

        if curve is None:
            return

        if is_visible:
            curve.show()
        else:
            curve.hide()

    def update_curve(
        self, row_id: int, curve_type_id: int, curve_derivation_id: int
    ) -> None:
        if not self.audio_path:
            return

        old_curve, panel = self.curves.get(row_id, [None, None])
        new_curve = None

        item = self.dashboard_widget.dashboard.topLevelItem(row_id)
        curve_name = item._curve_type.currentText()
        derivation_id = item._derivation_type.currentIndex()

        if curve_name in self.custom_curves:
            custom_curve_config = self.custom_curves[curve_name]
            generator_function = custom_curve_config["generator_function"]
            params = custom_curve_config["params"]
            new_curve = generator_function(self.audio_path, params, derivation_id)
        else:
            if curve_type_id >= 0 and curve_type_id < len(
                self.curve_generator.datasources
            ):
                new_curve = self.curve_generator.generate(
                    self.audio_path, curve_type_id, derivation_id
                )
            else:
                return

        if panel is None:
            return

        if old_curve is not None:
            try:
                panel.panel.remove_curve(old_curve)
            except ValueError:
                pass

        if new_curve is not None:
            panel.panel.add_curve(new_curve)
            self.curves[row_id][0] = new_curve

    def handle_new_row(self, row_count: int) -> None:
        new_row_id = row_count - 1

        assert new_row_id >= 0
        assert new_row_id not in self.curves
        assert len(self.panels) > 0

        self.curves[new_row_id] = [None, self.panels[0]]
    def reset_dashboard(self) -> None:
        self.dashboard_widget.dashboard.reset()
        # Also clear any internal tracking of curves
        self.curves.clear()



    def reset_curves(self) -> None:
        self.curves.clear()
        for panel in self.panels:
            panel.panel.reset()

    def open_config(self):
        dialog = UnifiedConfigDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            params = dialog.get_parameters()
            if params["mfcc"]["enabled"]:
                self.add_custom_curve(
                    params["mfcc"],
                    params["mfcc"]["panel"],
                    "Custom MFCC",
                    self.curve_generator.generate_custom_mfcc,
                )
            if params["amplitude"]["enabled"]:
                self.add_custom_curve(
                    params["amplitude"],
                    params["amplitude"]["panel"],
                    "Custom Amplitude",
                    self.curve_generator.generate_custom_amplitude,
                )
            if params["formant1"]["enabled"]:
                self.add_custom_curve(
                    params["formant1"],
                    params["formant1"]["panel"],
                    "Custom Formant1",
                    self.curve_generator.generate_custom_formant1,
                )
            if params["formant2"]["enabled"]:
                self.add_custom_curve(
                    params["formant2"],
                    params["formant2"]["panel"],
                    "Custom Formant2",
                    self.curve_generator.generate_custom_formant2,
                )
            if params["formant3"]["enabled"]:
                self.add_custom_curve(
                    params["formant3"],
                    params["formant3"]["panel"],
                    "Custom Formant3",
                    self.curve_generator.generate_custom_formant3,
                )
            if params["f0"]["enabled"]:
                self.add_custom_curve(
                    params["f0"],
                    params["f0"]["panel"],
                    "Custom F0",
                    self.curve_generator.generate_custom_f0,
                )

    def add_custom_curve(
        self, params, panel_id, default_curve_name, generator_function
    ):
        derivation_id = params["derivation_type"]

        curve_values = generator_function(self.audio_path, params, derivation_id)
        panel = self.panels[panel_id].panel
        panel.add_curve(curve_values)

        self.dashboard_widget.dashboard.append_row()
        row_id = self.dashboard_widget.dashboard.row_count - 1
        item = self.dashboard_widget.dashboard.topLevelItem(row_id)

        curve_name = params.get("name", default_curve_name)
        item._curve_type.addItem(curve_name)
        index = item._curve_type.findText(curve_name)
        if index != -1:
            item._curve_type.setCurrentIndex(index)

        item.panel_choice.setCurrentIndex(panel_id)
        item._derivation_type.setCurrentIndex(derivation_id)

        self.curves[row_id] = [curve_values, self.panels[panel_id]]
        self.custom_curves[curve_name] = {
            "params": params,
            "panel_id": panel_id,
            "generator_function": generator_function,
        }

    def add_custom_f0_curve(self, params, panel_id):
        derivation_id = 0
        curve_values = self.curve_generator.generate_custom_f0(
            self.audio_path, params, derivation_id
        )
        panel = self.panels[panel_id].panel
        panel.add_curve(curve_values)

        self.dashboard_widget.dashboard.append_row()
        row_id = self.dashboard_widget.dashboard.row_count - 1
        item = self.dashboard_widget.dashboard.topLevelItem(row_id)

        item._curve_type.addItem(params["name"])
        index = item._curve_type.findText(params["name"])
        if index != -1:
            item._curve_type.setCurrentIndex(index)

        item.panel_choice.setCurrentIndex(panel_id)

        self.curves[row_id] = [curve_values, self.panels[panel_id]]
        self.custom_f0_params[row_id] = params

    def add_custom_formant1_curve(self, params, panel_id):
        curve_values = self.curve_generator.generate_custom_formant1(
            self.audio_path, params, 0
        )
        panel = self.panels[panel_id].panel
        panel.add_curve(curve_values)

        self.dashboard_widget.dashboard.append_row()
        row_id = self.dashboard_widget.dashboard.row_count - 1
        item = self.dashboard_widget.dashboard.topLevelItem(row_id)

        item._curve_type.addItem(params["name"])
        index = item._curve_type.findText(params["name"])
        if index != -1:
            item._curve_type.setCurrentIndex(index)

        item.panel_choice.setCurrentIndex(panel_id)

        self.curves[row_id] = [curve_values, self.panels[panel_id]]
        self.custom_formant1_params[row_id] = params

    def add_custom_formant2_curve(self, params, panel_id):
        curve_values = self.curve_generator.generate_custom_formant2(
            self.audio_path, params, 0
        )
        panel = self.panels[panel_id].panel
        panel.add_curve(curve_values)

        self.dashboard_widget.dashboard.append_row()
        row_id = self.dashboard_widget.dashboard.row_count - 1
        item = self.dashboard_widget.dashboard.topLevelItem(row_id)

        item._curve_type.addItem(params["name"])
        index = item._curve_type.findText(params["name"])
        if index != -1:
            item._curve_type.setCurrentIndex(index)

        item.panel_choice.setCurrentIndex(panel_id)

        self.curves[row_id] = [curve_values, self.panels[panel_id]]
        self.custom_formant2_params[row_id] = params

    def add_custom_formant3_curve(self, params, panel_id):
        curve_values = self.curve_generator.generate_custom_formant3(
            self.audio_path, params, 0
        )
        panel = self.panels[panel_id].panel
        panel.add_curve(curve_values)

        self.dashboard_widget.dashboard.append_row()
        row_id = self.dashboard_widget.dashboard.row_count - 1
        item = self.dashboard_widget.dashboard.topLevelItem(row_id)

        item._curve_type.addItem(params["name"])
        index = item._curve_type.findText(params["name"])
        if index != -1:
            item._curve_type.setCurrentIndex(index)

        item.panel_choice.setCurrentIndex(panel_id)

        self.curves[row_id] = [curve_values, self.panels[panel_id]]
        self.custom_formant3_params[row_id] = params

    def add_custom_mfcc_curve(self, params, panel_id):
        curve_values = self.curve_generator.generate_custom_mfcc(
            self.audio_path, params, 0
        )
        panel = self.panels[panel_id].panel
        panel.add_curve(curve_values)

        self.dashboard_widget.dashboard.append_row()
        row_id = self.dashboard_widget.dashboard.row_count - 1
        item = self.dashboard_widget.dashboard.topLevelItem(row_id)

        item._curve_type.addItem(params["name"])
        index = item._curve_type.findText(params["name"])
        if index != -1:
            item._curve_type.setCurrentIndex(index)

        item.panel_choice.setCurrentIndex(panel_id)

        self.curves[row_id] = [curve_values, self.panels[panel_id]]
        self.custom_mfcc_params[row_id] = params

    def add_custom_amplitude_curve(self, params, panel_id):
        derivation_id = 0
        curve_values = self.curve_generator.generate_custom_amplitude(
            self.audio_path, params, derivation_id
        )
        panel = self.panels[panel_id].panel
        panel.add_curve(curve_values)

        self.dashboard_widget.dashboard.append_row()
        row_id = self.dashboard_widget.dashboard.row_count - 1
        item = self.dashboard_widget.dashboard.topLevelItem(row_id)

        item._curve_type.addItem(params["name"])
        index = item._curve_type.findText(params["name"])
        if index != -1:
            item._curve_type.setCurrentIndex(index)

        item.panel_choice.setCurrentIndex(panel_id)

        self.curves[row_id] = [curve_values, self.panels[panel_id]]
        self.custom_amplitude_params[row_id] = params

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.record_button.setText("Stop Recording")
        self.stream = sd.InputStream(
            callback=self.audio_callback, channels=1, samplerate=44100, dtype="int16"
        )
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        self.record_button.setText("Record Audio")
        self.stream.stop()
        self.stream.close()
        self.timer.stop()

        recorded_audio = np.concatenate(self.frames, axis=0)
        non_zero_audio_data = recorded_audio[recorded_audio != 0]

        audio_path, _ = QFileDialog.getSaveFileName(
            self, "Save Recorded Audio", "", "Audio Files (*.wav)"
        )
        if audio_path:
            wavfile.write(audio_path, 44100, non_zero_audio_data)

            self.audio_path = audio_path
            self.audio_indicator.file_loaded(audio_path)
            self.audio_widget.set_data(Parselmouth(audio_path))
            self.reset_curves()

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.frames.append(indata.copy())
            self.update_plot()

    def update_plot(self):
        if self.frames:
            audio_data = np.concatenate(self.frames, axis=0)
            self.audio_widget.update_audio_waveform(audio_data)

    def play_selected_region(self):
        if not self.audio_path:
            return

        region = self.audio_widget.selection_region.getRegion()
        start, end = region
        duration = end - start
        audio = AudioSegment.from_wav(self.audio_path)
        selected_audio = audio[start * 1000 : end * 1000]

        def play_audio():
            self.playing = True
            play(selected_audio)
            self.playing = False

        threading.Thread(target=play_audio).start()

        self.audio_cursor.setRegion([start, start])
        self.audio_cursor.show()
        threading.Thread(
            target=self.animate_cursor, args=(start, end, duration)
        ).start()

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


if __name__ == "__main__":
    pg.setConfigOptions(foreground="black", background="w")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
