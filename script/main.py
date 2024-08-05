import os
import sys
import numpy as np

from abc import ABC, abstractmethod
from typing import override

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import parselmouth
import tgt
import sounddevice as sd
import wave
from PyQt5.QtWidgets import QFileDialog

from scipy.io import wavfile

from mfcc import load_channel, get_MFCCS_change
from calc import calc_formants, calculate_amplitude_envelope,get_f0
from ui import Crosshair, create_plot_widget, ZoomToolbar
from praat_py_ui.parselmouth_calc import Parselmouth
from quadruple_axis_plot_item import (
    QuadrupleAxisPlotItem,
    Panel,
    CalculationValues,
    PanelWidget,
    SoundInformation,
    DisplayInterval,
)
class UnifiedConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Parameters")
        self.layout = QtWidgets.QGridLayout()

        # Main layout with a scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QGridLayout(scroll_content)
        scroll_area.setWidget(scroll_content)

        # Checkbox for enabling/disabling MFCC customization
        self.mfcc_enable_checkbox = QtWidgets.QCheckBox("Enable MFCC Customization")
        self.mfcc_enable_checkbox.setChecked(True)
        self.mfcc_enable_checkbox.stateChanged.connect(self.toggle_mfcc_fields)

        # MFCC Configuration
        self.mfcc_sample_rate_input = self.create_input_field("Sample Rate (Hz):", "10000")
        self.mfcc_tstep_input = self.create_input_field("Time Step (s):", "0.005")
        self.mfcc_winlen_input = self.create_input_field("Window Length (s):", "0.025")
        self.mfcc_nmfcc_input = self.create_input_field("Number of MFCCs:", "13")
        self.mfcc_nfft_input = self.create_input_field("Number of FFT Points:", "512")
        self.mfcc_remove_first_input = self.create_input_field("Remove First MFCC (1/0):", "1")
        self.mfcc_filt_cutoff_input = self.create_input_field("Filter Cutoff Frequency (Hz):", "12")
        self.mfcc_filt_ord_input = self.create_input_field("Filter Order:", "6")
        self.mfcc_name_input = self.create_input_field("Curve Name:", "Custom MFCC")
        self.mfcc_panel_choice = QtWidgets.QComboBox()
        self.mfcc_panel_choice.addItems(["1", "2", "3", "4"])

        # Checkbox for enabling/disabling Amplitude customization
        self.amp_enable_checkbox = QtWidgets.QCheckBox("Enable Amplitude Customization")
        self.amp_enable_checkbox.setChecked(True)
        self.amp_enable_checkbox.stateChanged.connect(self.toggle_amp_fields)

        # Amplitude Configuration
        self.amp_method_input = self.create_input_field("Method (RMS/Hilb/RMSpraat):", "RMS")
        self.amp_winlen_input = self.create_input_field("Window Length (s):", "0.1")
        self.amp_hoplen_input = self.create_input_field("Hop Length (s):", "0.01")
        self.amp_center_input = self.create_input_field("Center (True/False):", "True")
        self.amp_outfilter_input = self.create_input_field("Output Filter (None/iir/fir/sg):", "None")
        self.amp_outfilt_type_input = self.create_input_field("Filter Type (low/band):", "low")
        self.amp_outfilt_cutoff_input = self.create_input_field("Filter Cutoff Frequency (Hz):", "12")
        self.amp_outfilt_len_input = self.create_input_field("Filter Length:", "6")
        self.amp_outfilt_polyord_input = self.create_input_field("Filter Polynomial Order:", "3")
        self.amp_name_input = self.create_input_field("Curve Name:", "Custom Amplitude")
        self.amp_panel_choice = QtWidgets.QComboBox()
        self.amp_panel_choice.addItems(["1", "2", "3", "4"])

        # Checkbox for enabling/disabling Formant1 customization
        self.formant1_enable_checkbox = QtWidgets.QCheckBox("Enable Formant1 Customization")
        self.formant1_enable_checkbox.setChecked(True)
        self.formant1_enable_checkbox.stateChanged.connect(self.toggle_formant1_fields)

        # Formant1 Configuration
        self.formant1_energy_threshold_input = self.create_input_field("Energy Threshold:", "20.0")
        self.formant1_tstep_input = self.create_input_field("Time Step (s):", "0.005")
        self.formant1_max_num_formants_input = self.create_input_field("Max Number of Formants:", "5")
        self.formant1_max_formant_input = self.create_input_field("Maximum Formant (Hz):", "5500.0")
        self.formant1_winlen_input = self.create_input_field("Window Length (s):", "0.025")
        self.formant1_pre_emphasis_input = self.create_input_field("Pre-emphasis From (Hz):", "50.0")
        self.formant1_name_input = self.create_input_field("Curve Name:", "Custom Formant1")
        self.formant1_panel_choice = QtWidgets.QComboBox()
        self.formant1_panel_choice.addItems(["1", "2", "3", "4"])

        # Checkbox for enabling/disabling Formant2 customization
        self.formant2_enable_checkbox = QtWidgets.QCheckBox("Enable Formant2 Customization")
        self.formant2_enable_checkbox.setChecked(True)
        self.formant2_enable_checkbox.stateChanged.connect(self.toggle_formant2_fields)

        # Formant2 Configuration
        self.formant2_energy_threshold_input = self.create_input_field("Energy Threshold:", "20.0")
        self.formant2_tstep_input = self.create_input_field("Time Step (s):", "0.005")
        self.formant2_max_num_formants_input = self.create_input_field("Max Number of Formants:", "5")
        self.formant2_max_formant_input = self.create_input_field("Maximum Formant (Hz):", "5500.0")
        self.formant2_winlen_input = self.create_input_field("Window Length (s):", "0.025")
        self.formant2_pre_emphasis_input = self.create_input_field("Pre-emphasis From (Hz):", "50.0")
        self.formant2_name_input = self.create_input_field("Curve Name:", "Custom Formant2")
        self.formant2_panel_choice = QtWidgets.QComboBox()
        self.formant2_panel_choice.addItems(["1", "2", "3", "4"])

        # Checkbox for enabling/disabling Formant3 customization
        self.formant3_enable_checkbox = QtWidgets.QCheckBox("Enable Formant3 Customization")
        self.formant3_enable_checkbox.setChecked(True)
        self.formant3_enable_checkbox.stateChanged.connect(self.toggle_formant3_fields)

        # Formant3 Configuration
        self.formant3_energy_threshold_input = self.create_input_field("Energy Threshold:", "20.0")
        self.formant3_tstep_input = self.create_input_field("Time Step (s):", "0.005")
        self.formant3_max_num_formants_input = self.create_input_field("Max Number of Formants:", "5")
        self.formant3_max_formant_input = self.create_input_field("Maximum Formant (Hz):", "5500.0")
        self.formant3_winlen_input = self.create_input_field("Window Length (s):", "0.025")
        self.formant3_pre_emphasis_input = self.create_input_field("Pre-emphasis From (Hz):", "50.0")
        self.formant3_name_input = self.create_input_field("Curve Name:", "Custom Formant3")
        self.formant3_panel_choice = QtWidgets.QComboBox()
        self.formant3_panel_choice.addItems(["1", "2", "3", "4"])

        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.clicked.connect(self.accept)

        # Organize layout into boxes for better readability
        self.add_groupbox_to_layout("MFCC Configuration", [
            self.mfcc_enable_checkbox,
            self.mfcc_sample_rate_input,
            self.mfcc_tstep_input,
            self.mfcc_winlen_input,
            self.mfcc_nmfcc_input,
            self.mfcc_nfft_input,
            self.mfcc_remove_first_input,
            self.mfcc_filt_cutoff_input,
            self.mfcc_filt_ord_input,
            self.mfcc_name_input,
            (QtWidgets.QLabel("MFCC Panel:"), self.mfcc_panel_choice)
        ], scroll_layout, 0, 0)

        self.add_groupbox_to_layout("Amplitude Configuration", [
            self.amp_enable_checkbox,
            self.amp_method_input,
            self.amp_winlen_input,
            self.amp_hoplen_input,
            self.amp_center_input,
            self.amp_outfilter_input,
            self.amp_outfilt_type_input,
            self.amp_outfilt_cutoff_input,
            self.amp_outfilt_len_input,
            self.amp_outfilt_polyord_input,
            self.amp_name_input,
            (QtWidgets.QLabel("Amplitude Panel:"), self.amp_panel_choice)
        ], scroll_layout, 0, 1)

        self.add_groupbox_to_layout("Formant1 Configuration", [
            self.formant1_enable_checkbox,
            self.formant1_energy_threshold_input,
            self.formant1_tstep_input,
            self.formant1_max_num_formants_input,
            self.formant1_max_formant_input,
            self.formant1_winlen_input,
            self.formant1_pre_emphasis_input,
            self.formant1_name_input,
            (QtWidgets.QLabel("Formant1 Panel:"), self.formant1_panel_choice)
        ], scroll_layout, 0, 2)

        self.add_groupbox_to_layout("Formant2 Configuration", [
            self.formant2_enable_checkbox,
            self.formant2_energy_threshold_input,
            self.formant2_tstep_input,
            self.formant2_max_num_formants_input,
            self.formant2_max_formant_input,
            self.formant2_winlen_input,
            self.formant2_pre_emphasis_input,
            self.formant2_name_input,
            (QtWidgets.QLabel("Formant2 Panel:"), self.formant2_panel_choice)
        ], scroll_layout, 1, 0)

        self.add_groupbox_to_layout("Formant3 Configuration", [
            self.formant3_enable_checkbox,
            self.formant3_energy_threshold_input,
            self.formant3_tstep_input,
            self.formant3_max_num_formants_input,
            self.formant3_max_formant_input,
            self.formant3_winlen_input,
            self.formant3_pre_emphasis_input,
            self.formant3_name_input,
            (QtWidgets.QLabel("Formant3 Panel:"), self.formant3_panel_choice)
        ], scroll_layout, 1, 1)

        scroll_layout.addWidget(self.apply_button, 2, 0, 1, 3)

        self.layout.addWidget(scroll_area)
        self.setLayout(self.layout)

    def create_input_field(self, label_text, default_value):
        label = QtWidgets.QLabel(label_text)
        input_field = QtWidgets.QLineEdit(default_value)
        container = QtWidgets.QVBoxLayout()
        container.addWidget(label)
        container.addWidget(input_field)
        widget = QtWidgets.QWidget()
        widget.setLayout(container)
        return widget, input_field

    def add_groupbox_to_layout(self, title, widgets, layout, row, col):
        group_box = QtWidgets.QGroupBox(title)
        group_box_layout = QtWidgets.QVBoxLayout()
        group_box.setLayout(group_box_layout)

        for widget in widgets:
            if isinstance(widget, tuple):
                h_layout = QtWidgets.QHBoxLayout()
                h_layout.addWidget(widget[0])
                h_layout.addWidget(widget[1])
                container = QtWidgets.QWidget()
                container.setLayout(h_layout)
                group_box_layout.addWidget(container)
            else:
                group_box_layout.addWidget(widget)

        layout.addWidget(group_box, row, col)

    def get_parameters(self):
        mfcc_enabled = self.mfcc_enable_checkbox.isChecked()
        amp_enabled = self.amp_enable_checkbox.isChecked()
        formant1_enabled = self.formant1_enable_checkbox.isChecked()
        formant2_enabled = self.formant2_enable_checkbox.isChecked()
        formant3_enabled = self.formant3_enable_checkbox.isChecked()
        params = {
            "mfcc": {
                "enabled": mfcc_enabled,
                "signal_sample_rate": int(self.mfcc_sample_rate_input[1].text()),
                "tStep": float(self.mfcc_tstep_input[1].text()),
                "winLen": float(self.mfcc_winlen_input[1].text()),
                "n_mfcc": int(self.mfcc_nmfcc_input[1].text()),
                "n_fft": int(self.mfcc_nfft_input[1].text()),
                "removeFirst": int(self.mfcc_remove_first_input[1].text()),
                "filtCutoff": float(self.mfcc_filt_cutoff_input[1].text()),
                "filtOrd": int(self.mfcc_filt_ord_input[1].text()),
                "name": self.mfcc_name_input[1].text(),
                "panel": int(self.mfcc_panel_choice.currentIndex())
            },
            "amplitude": {
                "enabled": amp_enabled,
                "method": self.amp_method_input[1].text(),
                "winLen": float(self.amp_winlen_input[1].text()),
                "hopLen": float(self.amp_hoplen_input[1].text()),
                "center": self.amp_center_input[1].text().lower() == 'true',
                "outFilter": None if self.amp_outfilter_input[1].text().lower() == 'none' else self.amp_outfilter_input[1].text(),
                "outFiltType": self.amp_outfilt_type_input[1].text(),
                "outFiltCutOff": [float(c) for c in self.amp_outfilt_cutoff_input[1].text().split()],
                "outFiltLen": int(self.amp_outfilt_len_input[1].text()),
                "outFiltPolyOrd": int(self.amp_outfilt_polyord_input[1].text()),
                "name": self.amp_name_input[1].text(),
                "panel": int(self.amp_panel_choice.currentIndex())
            },
            "formant1": {
                "enabled": formant1_enabled,
                "energy_threshold": float(self.formant1_energy_threshold_input[1].text()),
                "time_step": float(self.formant1_tstep_input[1].text()),
                "max_num_formants": int(self.formant1_max_num_formants_input[1].text()),
                "max_formant": float(self.formant1_max_formant_input[1].text()),
                "window_length": float(self.formant1_winlen_input[1].text()),
                "pre_emphasis_from": float(self.formant1_pre_emphasis_input[1].text()),
                "name": self.formant1_name_input[1].text(),
                "panel": int(self.formant1_panel_choice.currentIndex())
            },
            "formant2": {
                "enabled": formant2_enabled,
                "energy_threshold": float(self.formant2_energy_threshold_input[1].text()),
                "time_step": float(self.formant2_tstep_input[1].text()),
                "max_num_formants": int(self.formant2_max_num_formants_input[1].text()),
                "max_formant": float(self.formant2_max_formant_input[1].text()),
                "window_length": float(self.formant2_winlen_input[1].text()),
                "pre_emphasis_from": float(self.formant2_pre_emphasis_input[1].text()),
                "name": self.formant2_name_input[1].text(),
                "panel": int(self.formant2_panel_choice.currentIndex())
            },
            "formant3": {
                "enabled": formant3_enabled,
                "energy_threshold": float(self.formant3_energy_threshold_input[1].text()),
                "time_step": float(self.formant3_tstep_input[1].text()),
                "max_num_formants": int(self.formant3_max_num_formants_input[1].text()),
                "max_formant": float(self.formant3_max_formant_input[1].text()),
                "window_length": float(self.formant3_winlen_input[1].text()),
                "pre_emphasis_from": float(self.formant3_pre_emphasis_input[1].text()),
                "name": self.formant3_name_input[1].text(),
                "panel": int(self.formant3_panel_choice.currentIndex())
            }
        }
        print("Parameters from dialog:", params)  # Debugging output
        return params

    def toggle_mfcc_fields(self, state):
        enabled = state == QtCore.Qt.Checked
        for widget in [
            self.mfcc_sample_rate_input[1],
            self.mfcc_tstep_input[1],
            self.mfcc_winlen_input[1],
            self.mfcc_nmfcc_input[1],
            self.mfcc_nfft_input[1],
            self.mfcc_remove_first_input[1],
            self.mfcc_filt_cutoff_input[1],
            self.mfcc_filt_ord_input[1],
            self.mfcc_name_input[1],
            self.mfcc_panel_choice
        ]:
            widget.setEnabled(enabled)

    def toggle_amp_fields(self, state):
        enabled = state == QtCore.Qt.Checked
        for widget in [
            self.amp_method_input[1],
            self.amp_winlen_input[1],
            self.amp_hoplen_input[1],
            self.amp_center_input[1],
            self.amp_outfilter_input[1],
            self.amp_outfilt_type_input[1],
            self.amp_outfilt_cutoff_input[1],
            self.amp_outfilt_len_input[1],
            self.amp_outfilt_polyord_input[1],
            self.amp_name_input[1],
            self.amp_panel_choice
        ]:
            widget.setEnabled(enabled)

    def toggle_formant1_fields(self, state):
        enabled = state == QtCore.Qt.Checked
        for widget in [
            self.formant1_energy_threshold_input[1],
            self.formant1_tstep_input[1],
            self.formant1_max_num_formants_input[1],
            self.formant1_max_formant_input[1],
            self.formant1_winlen_input[1],
            self.formant1_pre_emphasis_input[1],
            self.formant1_name_input[1],
            self.formant1_panel_choice
        ]:
            widget.setEnabled(enabled)

    def toggle_formant2_fields(self, state):
        enabled = state == QtCore.Qt.Checked
        for widget in [
            self.formant2_energy_threshold_input[1],
            self.formant2_tstep_input[1],
            self.formant2_max_num_formants_input[1],
            self.formant2_max_formant_input[1],
            self.formant2_winlen_input[1],
            self.formant2_pre_emphasis_input[1],
            self.formant2_name_input[1],
            self.formant2_panel_choice
        ]:
            widget.setEnabled(enabled)

    def toggle_formant3_fields(self, state):
        enabled = state == QtCore.Qt.Checked
        for widget in [
            self.formant3_energy_threshold_input[1],
            self.formant3_tstep_input[1],
            self.formant3_max_num_formants_input[1],
            self.formant3_max_formant_input[1],
            self.formant3_winlen_input[1],
            self.formant3_pre_emphasis_input[1],
            self.formant3_name_input[1],
            self.formant3_panel_choice
        ]:
            widget.setEnabled(enabled)

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
            ["Choose", "Mod_Cepstr", "F1", "F2", "F3", "F0", "ENV_AMP"]        )
        self.ema_type.setStyleSheet(
            "background-color: lightblue; border: 1px solid black; padding: 5px"
        )
        self.panel_choice.addItems(["1", "2", "3", "4"])
        self.visibility_checkbox.setChecked(True)
        self._derivation_type.addItems(
            ["Traj. (f(x))", "vel. (f(x)')", "acc. (f(x)'')"]
        )

    def lay_out_widgets(self) -> None:
        self.parent.setItemWidget(self, 0, self._curve_type)
        self.parent.setItemWidget(self, 1, self.ema_type)
        self.parent.setItemWidget(self, 2, self.color_selection)
        self.parent.setItemWidget(self, 3, self.panel_choice)
        self.parent.setItemWidget(self, 4, self.visibility_checkbox)
        self.parent.setItemWidget(self, 5, self._derivation_type)

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
    def derivation_type(self) -> None:
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

    def __init__(self) -> None:
        super().__init__()

        self.row_count = 0
        self.headers = ["Acoustique", "EMA", "Couleur", "Panel", "Show", "Dérivée"]

        self.setColumnCount(len(self.headers))
        self.setHeaderLabels(self.headers)
        self.resize_column()

        # for _ in range(4):
        #     self.append_row()

    def resize_column(self) -> None:
        self.setColumnWidth(self.headers.index("EMA"), 90)
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

    def reset(self) -> None:
        self.row_count = 0
        self.clear()

    def selected_panel(self, row_idx: int) -> int:
        if row_idx < 0 or row_idx >= self.row_count:
            raise ValueError(f"Incorrect row id given {row_idx}")


class DashboardWidget(QtWidgets.QWidget):
    dashboard: Dashboard
    row_added = QtCore.pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__()

        self.dashboard = Dashboard()

        add_row_button = StyledButton("+", "lightgreen")
        add_row_button.clicked.connect(self._row_added)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.dashboard)
        layout.addWidget(add_row_button)

        self.setLayout(layout)

    def _row_added(self) -> None:
        self.dashboard.append_row()
        self.row_added.emit(self.dashboard.row_count)


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
    def transform(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass


class Trajectory(Transformation):

    @override
    def transform(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x, y


class Velocity(Transformation):

    @override
    def transform(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x, np.gradient(y)


class Acceleration(Transformation):

    @override
    def transform(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x, np.gradient(np.gradient(y))


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
        x, y = get_MFCCS_change(data)

        return x, y


class Formant1(DataSource):

    @override
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        f_times, f1_values, _, _ = calc_formants(
            parselmouth.Sound(audio_path), 0, 99999, 40
        )
        return f_times, f1_values


class Formant2(DataSource):

    @override
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        f_times, _, f2_values, _ = calc_formants(
            parselmouth.Sound(audio_path), 0, 99999, 40
        )
        return f_times, f2_values


class Formant3(DataSource):

    @override
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        f_times, _, _, f3_values = calc_formants(
            parselmouth.Sound(audio_path), 0, 99999, 40
        )
        return f_times, f3_values
class F0(DataSource):

    @override
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        sample_rate, audio_signal = wavfile.read(audio_path)
        f0, f0_times = get_f0(audio_signal, sample_rate)
        return f0_times, f0

class AmplitudeEnvelope(DataSource):

    @override
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        sample_rate, audio_signal = wavfile.read(audio_path)

        # audio_signal = audio_signal[int(start * sample_rate):int(end * sample_rate)]
        amplitude_envelope,time_axis = calculate_amplitude_envelope(audio_signal, sample_rate)


        return time_axis, amplitude_envelope


class Plotter(ABC):

    @abstractmethod
    def plot(self, x: np.ndarray, y: np.ndarray) -> CalculationValues:
        pass
        f_times, _, _, f2_values = calc_formants(
            parselmouth.Sound(audio_path), 0, 99999
        )
        return f_times, f2_values


class CurvePlotter(Plotter):

    @override
    def plot(self, x: np.ndarray, y: np.ndarray) -> CalculationValues:
        curve = pg.PlotDataItem(x=x, y=y)
        min = pg.ScatterPlotItem()
        max = pg.ScatterPlotItem()

        return CalculationValues(curve, min, max)


class ScatterPlotPlotter(Plotter):

    @override
    def plot(self, x: np.ndarray, y: np.ndarray) -> CalculationValues:
        curve = pg.ScatterPlotItem(x=x, y=y)
        min = pg.ScatterPlotItem()
        max = pg.ScatterPlotItem()

        return CalculationValues(curve, min, max)


class CurveGenerator:
    datasources: list[DataSource]
    derivations: list[Transformation]
    plotters: list[Plotter]

    def __init__(self) -> None:
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
            CurvePlotter(),
            ScatterPlotPlotter(),
            ScatterPlotPlotter(),
            ScatterPlotPlotter(),
            CurvePlotter(),
            CurvePlotter(),
        ]

    def generate(self, audio_path: str, curve_type_id: int, curve_derivation: int) -> CalculationValues:
        if curve_type_id < 0 or curve_type_id >= len(self.datasources):
            raise IndexError("Curve type ID is out of range")

        source = self.datasources[curve_type_id]

        if source is None:
            raise ValueError("Invalid data source for the given curve type ID")

        operation = self.derivations[curve_derivation]
        plotter = self.plotters[curve_type_id]

        data = source.calculate(audio_path)
        x, y = operation.transform(*data)

        return plotter.plot(x, y)
    def generate_custom_formant2(self, audio_path: str, params: dict, derivation_id: int) -> CalculationValues:
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
            pre_emphasis_from=params["pre_emphasis_from"]
        )

        operation = self.derivations[derivation_id]
        x, y = operation.transform(f_times, f2_values)
        
        plotter = ScatterPlotPlotter()
        return plotter.plot(x, y)

    def generate_custom_formant3(self, audio_path: str, params: dict, derivation_id: int) -> CalculationValues:
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
            pre_emphasis_from=params["pre_emphasis_from"]
        )

        operation = self.derivations[derivation_id]
        x, y = operation.transform(f_times, f3_values)
        
        plotter = ScatterPlotPlotter()
        return plotter.plot(x, y)
    def generate_custom_formant1(self, audio_path: str, params: dict, derivation_id: int) -> CalculationValues:
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
            pre_emphasis_from=params["pre_emphasis_from"]
        )

        operation = self.derivations[derivation_id]
        x, y = operation.transform(f_times, f1_values)
        
        plotter = ScatterPlotPlotter()
        return plotter.plot(x, y)
    def generate_custom_mfcc(self, audio_path: str, params: dict, derivation_id: int) -> CalculationValues:
        data = load_channel(audio_path)
        x, y = get_MFCCS_change(
            audio_data=data,
            signal_sample_rate=params["signal_sample_rate"],
            tStep=params["tStep"],
            winLen=params["winLen"],
            n_mfcc=params["n_mfcc"],
            n_fft=params["n_fft"],
            removeFirst=params["removeFirst"],
            filtCutoff=params["filtCutoff"],
            filtOrd=params["filtOrd"]
        )
        operation = self.derivations[derivation_id]
        x, y = operation.transform(x, y)
        print(x)
        plotter = CurvePlotter()
        return plotter.plot(x, y)
    
    def generate_custom_amplitude(self, audio_path: str, params: dict, derivation_id: int) -> CalculationValues:
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
            outFiltPolyOrd=params["outFiltPolyOrd"]
        )

        operation = self.derivations[derivation_id]
        time_axis, amplitude = operation.transform(time_axis, amplitude)


        plotter = CurvePlotter()
        return plotter.plot(time_axis, amplitude)

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
        self.init_main_layout()

        self.audio_path = None
        self.audio_widget = SoundInformation()

        self.annotation_path = None
        self.annotation_data = None
        self.annotation_widget = DisplayInterval(self.audio_widget)

        self.curve_generator = CurveGenerator()
        self.dashboard_widget = DashboardWidget()

        self.zoom = ZoomToolbar(self.audio_widget.selection_region)

        self.audio_indicator = FileLoadIndicator(
            "Loaded Audio", "No audio Loaded", "blue"
        )
        self.annotation_indicator = FileLoadIndicator(
            "Loaded TextGrid", "No textGrid loaded", "red"
        )
        self.tier_selection = TierSelection()
        self.config_mfcc_button = StyledButton("Configure")

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
        self.add_control_widget(self.zoom)

        self.add_curve_widget(self.audio_widget)

        self.curves = {}
        self.panels = []
        self.custom_mfcc_params = {}
        self.custom_amplitude_params = {}
        self.custom_formant1_params = {}
        self.custom_formant2_params = {}
        self.custom_formant3_params = {}
        for i in range(4):
            panel_widget = PanelWidget(i + 1)

            self.zoom.link_viewbox(panel_widget.panel)

            self.add_curve_widget(panel_widget)
            self.panels.append(panel_widget)


        # Add recording state
        self.recording = False
        self.frames = []
        self.recorded_audio = []
        self.stream = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)  # Update plot every 100 ms

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

    def create_load_buttons(self) -> QtWidgets.QGroupBox:
        load_group_box = QtWidgets.QGroupBox("Load Audio and TextGrid")
        load_layout = QtWidgets.QVBoxLayout()

        load_audio_button = StyledButton("Load Audio")
        load_textgrid_button = StyledButton("Load TextGrid")
        self.record_button = StyledButton("Record Audio", "lightgreen")

        load_audio_button.clicked.connect(self.load_audio)
        load_textgrid_button.clicked.connect(self.load_annotations)
        self.record_button.clicked.connect(self.toggle_recording)
        load_layout.addWidget(load_audio_button)
        load_layout.addWidget(load_textgrid_button)
        load_layout.addWidget(self.record_button)

        load_group_box.setLayout(load_layout)
        return load_group_box

    def create_audio_control_buttons(self) -> QtWidgets.QGroupBox:
        audio_control_group_box = QtWidgets.QGroupBox("Audio Control")
        audio_control_layout = QtWidgets.QVBoxLayout()

        play_button = StyledButton("Play Selected Region")

        audio_control_layout.addWidget(play_button)

        audio_control_group_box.setLayout(audio_control_layout)
        return audio_control_group_box

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

    def update_curve(self, row_id: int, curve_type_id: int, curve_derivation_id: int) -> None:
        if not self.audio_path:
            return

        old_curve, panel = self.curves.get(row_id, [None, None])
        new_curve = None
        if curve_type_id >= 0 and curve_type_id < len(self.curve_generator.datasources):
            new_curve = self.curve_generator.generate(
                self.audio_path, curve_type_id, curve_derivation_id
            )
        elif curve_type_id == len(self.curve_generator.datasources):
            if row_id not in self.custom_mfcc_params:
                return
            params = self.custom_mfcc_params[row_id]
            new_curve = self.curve_generator.generate_custom_mfcc(
                self.audio_path, params, curve_derivation_id
            )
        elif curve_type_id == len(self.curve_generator.datasources):
            if row_id not in self.custom_amplitude_params:
                return
            params = self.custom_amplitude_params[row_id]
            new_curve = self.curve_generator.generate_custom_amplitude(
                self.audio_path, params, curve_derivation_id
            )
        elif curve_type_id == len(self.curve_generator.datasources):
            if row_id not in self.custom_formant1_params:
                return
            params = self.custom_formant1_params[row_id]
            new_curve = self.curve_generator.generate_custom_formant1(
                self.audio_path, params, curve_derivation_id
            )
        elif curve_type_id == len(self.curve_generator.datasources):
            if row_id not in self.custom_formant2_params:
                return
            params = self.custom_formant2_params[row_id]
            new_curve = self.curve_generator.generate_custom_formant2(
                self.audio_path, params, curve_derivation_id
            )
        elif curve_type_id == len(self.curve_generator.datasources):
            if row_id not in self.custom_formant3_params:
                return
            params = self.custom_formant3_params[row_id]
            new_curve = self.curve_generator.generate_custom_formant3(
                self.audio_path, params, curve_derivation_id
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

    def handle_new_row(self, row_count: int) -> None:
        new_row_id = row_count - 1

        assert new_row_id >= 0
        assert new_row_id not in self.curves
        assert len(self.panels) > 0

        self.curves[new_row_id] = [None, self.panels[0]]

    def reset_curves(self) -> None:
        self.curves.clear()
        for panel in self.panels:
            panel.panel.reset()

    def open_config(self):
        dialog = UnifiedConfigDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            params = dialog.get_parameters()
            if params["mfcc"]["enabled"]:
                self.add_custom_mfcc_curve(params["mfcc"], params["mfcc"]["panel"])
            if params["amplitude"]["enabled"]:
                self.add_custom_amplitude_curve(params["amplitude"], params["amplitude"]["panel"])
            if params["formant1"]["enabled"]:
                self.add_custom_formant1_curve(params["formant1"], params["formant1"]["panel"])
            if params["formant2"]["enabled"]:
                self.add_custom_formant2_curve(params["formant2"], params["formant2"]["panel"])
            if params["formant3"]["enabled"]:
                self.add_custom_formant3_curve(params["formant3"], params["formant3"]["panel"])

    def add_custom_formant1_curve(self, params, panel_id):
        curve_values = self.curve_generator.generate_custom_formant1(self.audio_path, params, 0)
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
        curve_values = self.curve_generator.generate_custom_formant2(self.audio_path, params, 0)
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
        curve_values = self.curve_generator.generate_custom_formant3(self.audio_path, params, 0)
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
        curve_values = self.curve_generator.generate_custom_mfcc(self.audio_path, params, 0)
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
        derivation_id = 0  # Par défaut à la trajectoire ; modifiez si nécessaire
        curve_values = self.curve_generator.generate_custom_amplitude(self.audio_path, params, derivation_id)
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
        self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=44100, dtype='int16')
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        self.record_button.setText("Record Audio")
        self.stream.stop()
        self.stream.close()
        self.timer.stop()

        recorded_audio = np.concatenate(self.frames, axis=0)
        non_zero_audio_data = recorded_audio[recorded_audio != 0]
        
        audio_path, _ = QFileDialog.getSaveFileName(self, "Save Recorded Audio", "", "Audio Files (*.wav)")
        if audio_path:
            wavfile.write(audio_path, 44100, non_zero_audio_data)

            self.audio_path = audio_path
            self.audio_indicator.file_loaded(audio_path)
            self.audio_widget.set_data(Parselmouth(audio_path))
            self.reset_curves()


    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.frames.append(indata.copy())
            # Ajoutez cette ligne pour mettre à jour le signal en temps réel
            self.update_plot()

    def update_plot(self):
        if self.frames:
            audio_data = np.concatenate(self.frames, axis=0)
            self.audio_widget.update_audio_waveform(audio_data)


if __name__ == "__main__":
    pg.setConfigOptions(foreground="black", background="w")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
