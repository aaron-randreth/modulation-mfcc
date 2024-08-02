import os
import sys
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import tgt
from ui import Crosshair, create_plot_widget, ZoomToolbar
from praat_py_ui.parselmouth_calc import Parselmouth
from quadruple_axis_plot_item import (
    QuadrupleAxisPlotItem,
    Panel,
    CalculationValues,
    PanelWidget,
    SoundInformation,
    DisplayInterval
)


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
        self.curve_combobox = QtWidgets.QComboBox()
        self.ema_button = QtWidgets.QPushButton(f"Button {self.id+1},{2}")
        self.color_selection = ColorSelection()
        self.panel_choice = QtWidgets.QComboBox()
        self.visibility_checkbox = QtWidgets.QCheckBox()
        self.derived_combo_box = QtWidgets.QComboBox()

        self.curve_combobox.addItems(
            ["Choose", "Mod_Cepstr", "F1", "F2", "F3", "Courbes ema", "ENV_AMP"]
        )
        self.ema_button.setStyleSheet(
            "background-color: lightblue; border: 1px solid black; padding: 5px"
        )
        self.panel_choice.addItems(["1", "2", "3", "4"])
        self.visibility_checkbox.setChecked(True)
        self.derived_combo_box.addItems(
            ["Traj. (f(x))", "vel. (f(x)')", "acc. (f(x)'')"]
        )

    def lay_out_widgets(self) -> None:
        self.parent.setItemWidget(self, 0, self.curve_combobox)
        self.parent.setItemWidget(self, 1, self.ema_button)
        self.parent.setItemWidget(self, 2, self.color_selection)
        self.parent.setItemWidget(self, 3, self.panel_choice)
        self.parent.setItemWidget(self, 4, self.visibility_checkbox)
        self.parent.setItemWidget(self, 5, self.derived_combo_box)

    def setup_signals(self) -> None:
        self.update_panel = self.curve_combobox.currentIndexChanged
        self.toggle_visibility = self.visibility_checkbox.stateChanged
        self.change_curve_color = self.color_selection.color_chosen
        self.update_derived = self.derived_combo_box.currentIndexChanged

    @property
    def selected_panel(self) -> int:
        return self.panel_choice.currentIndex()


class Dashboard(QtWidgets.QTreeWidget):
    update_panel = QtCore.pyqtSignal(int, int)
    toggle_visibility = QtCore.pyqtSignal(int, int)
    change_curve_color = QtCore.pyqtSignal(int, str)
    update_derived = QtCore.pyqtSignal(int, int)

    row_count: int
    headers: list[str]

    def __init__(self) -> None:
        super().__init__()

        self.row_count = 0
        self.headers = ["Acoustique", "EMA", "Couleur", "Panel", "Show", "Dérivée"]

        self.setColumnCount(len(self.headers))
        self.setHeaderLabels(self.headers)
        self.resize_column()

        for _ in range(4):
            self.append_row()

    def resize_column(self) -> None:
        self.setColumnWidth(self.headers.index("EMA"), 90)
        self.setColumnWidth(self.headers.index("Panel"), 50)
        self.setColumnWidth(self.headers.index("Show"), 20)

    def append_row(self) -> None:
        item = TreeWidgetItem(self, self.row_count)
        item.update_panel.connect(
            lambda index, row=item.id: self.update_panel.emit(row, index)
        )
        item.toggle_visibility.connect(
            lambda state, row=item.id: self.toggle_visibility.emit(row, state)
        )
        item.change_curve_color.connect(
            lambda color, row=item.id: self.change_curve_color.emit(row, color)
        )
        item.update_derived.connect(
            lambda index, row=item.id: self.update_derived.emit(row, index)
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

    def __init__(self) -> None:
        super().__init__()

        self.dashboard = Dashboard()
        add_row_button = StyledButton("+", "lightgreen")
        add_row_button.clicked.connect(self.dashboard.append_row)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.dashboard)
        layout.addWidget(add_row_button)

        self.setLayout(layout)


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
    textgrid_data: tgt.io.TextGrid | None

    button_group: QtWidgets.QButtonGroup
    tier_checked = QtCore.pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__("Select TextGrid Tier")

        textgrid_data = None
        layout = QtWidgets.QVBoxLayout()

        self.button_group = QtWidgets.QButtonGroup(self)
        self.button_group.setExclusive(True)
        self.button_group.buttonToggled.connect(self._tier_checked)

        self.setLayout(layout)

    def set_data(self, data: tgt.io.TextGrid) -> None:
        self.reset()

        self.textgrid_data = data

        self.populate_textgrid_selection(data.get_tier_names())

    def populate_textgrid_selection(self, tiers: list[str]) -> None:
        for tier_name in tiers:
            btn = QtWidgets.QRadioButton(tier_name)

            self.button_group.addButton(btn)
            self.layout().addWidget(btn)

    def _tier_checked(self, button: QtWidgets.QRadioButton, checked: bool) -> None:
        if not checked:
            return

        tier_name = button.text()
        self.tier_checked.emit(tier_name)

    def reset(self) -> None:
        layout = self.layout()

        for btn in self.button_group.buttons():
            layout.removeWidget(btn)
            self.button_group.removeButton(btn)

            btn.deleteLater()


class MainWindow(QtWidgets.QMainWindow):
    audio_path: str | None
    audio_data: Parselmouth | None
    audio_widget: SoundInformation

    annotation_path: str | None
    annotation_data: str | None
    annotation_widget: DisplayInterval

    panels: list[PanelWidget]

    def __init__(self) -> None:
        super().__init__()
        self.init_main_layout()

        self.audio_path = None
        self.audio_data = None
        self.audio_widget = SoundInformation()

        self.annotation_path = None
        self.annotation_data = None
        self.annotation_widget = DisplayInterval(self.audio_widget)

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

        self.textgrid_annotations = []

        self.tier_selection.tier_checked.connect(self.display_annotations)
        # self.config_mfcc_button.clicked.connect(self.open_mfcc_config)

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

        self.panels = []

        for i in range(4):
            panel_widget = PanelWidget(i + 1)

            self.zoom.link_viewbox(panel_widget.panel)

            self.add_curve_widget(panel_widget)
            self.panels.append(panel_widget)

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

    def create_load_buttons(self) -> None:
        load_group_box = QtWidgets.QGroupBox("Load Audio and TextGrid")
        load_layout = QtWidgets.QVBoxLayout()

        load_audio_button = StyledButton("Load Audio")
        load_textgrid_button = StyledButton("Load TextGrid")
        self.record_button = StyledButton("Record Audio", "lightgreen")

        load_audio_button.clicked.connect(self.load_audio)
        load_textgrid_button.clicked.connect(self.load_annotations)
        # self.record_button.clicked.connect(self.toggle_recording)

        load_layout.addWidget(load_audio_button)
        load_layout.addWidget(load_textgrid_button)
        load_layout.addWidget(self.record_button)

        load_group_box.setLayout(load_layout)
        return load_group_box

    def create_audio_control_buttons(self) -> None:
        audio_control_group_box = QtWidgets.QGroupBox("Audio Control")
        audio_control_layout = QtWidgets.QVBoxLayout()

        play_button = StyledButton("Play Selected Region")
        # play_button.clicked.connect(self.play_selected_region)
        audio_control_layout.addWidget(play_button)

        audio_control_group_box.setLayout(audio_control_layout)
        return audio_control_group_box

    def create_spectrogram_checkbox(self) -> None:
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
        self.audio_data = Parselmouth(audio_path)

        self.audio_widget.set_data(self.audio_data)

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

    def display_annotations(self, tier_name):
        tier = self.annotation_data.get_tier_by_name(tier_name)
        self.annotation_widget.display(tier)


if __name__ == "__main__":
    pg.setConfigOptions(foreground="black", background="w")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
