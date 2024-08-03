import os
import sys
import numpy as np

from abc import ABC, abstractmethod

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import tgt
import parselmouth

from mfcc import load_channel, get_MFCCS_change
from calc import calc_formants
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
            ["Choose", "Mod_Cepstr", "F1", "F2", "F3", "Courbes ema", "ENV_AMP"]
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
    color_changed = QtCore.pyqtSignal(int, int)
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


class Trajectory(Transormation):

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
        return x, np.grandient(np.grandient(y))


class Mfcc(DataSource):

    @override
    def calculate(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        # TODO
        audio_data = ...  # lit le fichier à partir de audio data
        x, y = get_MCCC(
            audio_data,
            valeur=12,  # Valeurs par défaut écrit en dure
            valeur=config["valeur"],
        )

        return x, y


class CurveGenerator:
    datasources: list[DataSource]
    derivations: list[Transformation]

    def __init__(
        # self, datasources: list[DataSources], derivations: list[Transformation]
        self
    ) -> None:
        # self.datasources = datasources
        # self.derivations = derivations

        self.datasources = [Mfcc()]
        self.derivations = [Trajectory(), Velocity(), Acceleration()]

    def generate(
        self, audio_path: str, curve_type_id: int, curve_derivation: int
    ) -> CalculationValues:
        source = self.datasources[curve_type_id]
        operation = self.derivations[curve_derivation]

        data = source.calculate(audio_path)
        x, y = operation.transform(*data)

        # PLOT

class MainWindow(QtWidgets.QMainWindow):
    audio_path: str | None
    audio_widget: SoundInformation

    annotation_path: str | None
    annotation_data: tgt.core.TextGrid | None
    annotation_widget: DisplayInterval

    panels: list[PanelWidget]
    # ["row_id": (curve_id, panel_id)]
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
        # self.config_mfcc_button.clicked.connect(self.open_mfcc_config)

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

        self.audio_widget.set_data(Parselmouth(audio_path))

        # self.dashboard_widget.dashboard.reset()
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

    def update_curve(
        self, row_id: int, curve_type_id: int, curve_derivation_id: int
    ) -> None:
        old_curve, panel = self.curves[row_id]
        new_curve = self.curve_generator.generate(
            self.audio_path, curve_type_id, curve_derivation_id
        )

        if panel is None:
            return

        if old_curve is not None:
            panel.panel.remove_curve(old_curve)

        panel.panel.add_curve(new_curve)

    def change_curve_panel(self, row_id: int, new_panel_id: int) -> None:
        curve, current_panel = self.curves[row_id]
        new_panel = self.panels[new_panel_id]

        self.curves[row_id][1] = new_panel

        if curve is None:
            return

        if current_panel is not None:
            current_panel.panel.remove_curve(curve)

        new_panel.panel.add_curve(curve)

    def change_curve_color(self, row_id: int, new_color: str) -> None:
        curve, _ = self.curves[row_id]

        if curve is None:
            return

        curve.setPen(color=new_color)

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

        self.curves[new_row_id] = (None, self.panels[0])

    def reset_curves(self) -> None:
        self.curves.clear()
        for panel in self.panels:
            panel.panel.reset()


if __name__ == "__main__":
    pg.setConfigOptions(foreground="black", background="w")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
