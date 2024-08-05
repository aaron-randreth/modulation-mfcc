from typing import override
from dataclasses import dataclass

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np
from bidict import bidict
import tgt

from praat_py_ui.parselmouth_calc import Sound, Spectrogram, Parselmouth
from praat_py_ui import spectrogram as display_spect

class QuadrupleAxisPlotItem(pg.PlotItem):
    central_row: int = 2
    column_count: int = 5
    row_count: int = 4

    def __init__(self) -> None:
        super().__init__()

        self.right = pg.ViewBox()
        self.right_bis = pg.ViewBox()

        self.left = self.vb
        self.left_bis = pg.ViewBox()

        self.right.setMouseEnabled(x=True, y=False)
        self.right_bis.setMouseEnabled(x=True, y=False)
        self.left.setMouseEnabled(x=True, y=False)
        self.left_bis.setMouseEnabled(x=True, y=False)

        self.shift_items_right()

        for axis_id in ("left", "bottom", "top"):
            self.axes[axis_id]["vb"] = self.vb

        self.setup_new_axes()
        self.position_axes()

        for axis in self.axes.values():
            axis["item"].hide()
            axis["items_count"] = 0

        self.getAxis("left").show()
        self.getAxis("bottom").show()

    def reset_layout_stretch(self) -> None:

        for i in range(self.row_count):
            self.layout.setRowPreferredHeight(i, 0)
            self.layout.setRowMinimumHeight(i, 0)
            self.layout.setRowSpacing(i, 0)
            self.layout.setRowStretchFactor(i, 1)

        for i in range(self.column_count):
            self.layout.setColumnPreferredWidth(i, 0)
            self.layout.setColumnMinimumWidth(i, 0)
            self.layout.setColumnSpacing(i, 0)
            self.layout.setColumnStretchFactor(i, 1)

        viewbox_row = 2
        self.layout.setRowStretchFactor(viewbox_row, 100)
        self.layout.setColumnStretchFactor(self.central_row, 100)

    def shift_items_right(self) -> None:
        # We know that all items only in pg.Plotitem only span 1 cell
        # We are shifting all items by one column to the right
        # so we go from right to left to avoid collisions.
        for col in reversed(range(self.layout.columnCount())):
            for row in range(self.layout.rowCount()):
                item = self.layout.itemAt(row, col)
                if item is None:
                    continue

                self.layout.removeItem(item)
                self.layout.addItem(item, row, col + 1)

        for axis in self.axes.values():
            row, col = axis["pos"]
            axis["pos"] = (row, col + 1)

    def add_viewboxes_to_scene(self) -> None:
        self.scene().addItem(self.right)
        self.scene().addItem(self.right_bis)
        self.scene().addItem(self.left_bis)

    def setup_new_axes(self) -> None:
        right_axis = self.getAxis("right")
        left_bis_axis = pg.AxisItem("left")
        right_bis_axis = pg.AxisItem("right")

        self.axes["right"]["vb"] = self.right
        self.axes["left_bis"] = {
            "item": left_bis_axis,
            "pos": (self.central_row, 0),
            "vb": self.left_bis,
        }
        self.axes["right_bis"] = {
            "item": right_bis_axis,
            "pos": (self.central_row, self.column_count - 1),
            "vb": self.right_bis,
        }

        right_axis.linkToView(self.right)
        left_bis_axis.linkToView(self.left_bis)
        right_bis_axis.linkToView(self.right_bis)

        self.right.setXLink(self)
        self.right_bis.setXLink(self)
        self.left_bis.setXLink(self)

        xmin, xmax = self.left.state["limits"]["xLimits"]
        self.right.setLimits(xMin=xmin, xMax=xmax)
        self.right_bis.setLimits(xMin=xmin, xMax=xmax)
        self.left_bis.setLimits(xMin=xmin, xMax=xmax)

        self.left.sigResized.connect(self.update_views)

    def position_axes(self) -> None:
        for axis in self.axes.values():
            self.layout.addItem(axis["item"], *axis["pos"])

    def update_views(self) -> None:
        self.right.setGeometry(self.left.sceneBoundingRect())
        self.right_bis.setGeometry(self.left.sceneBoundingRect())
        self.left_bis.setGeometry(self.left.sceneBoundingRect())

        self.right.linkedViewChanged(self.left, self.right.XAxis)
        self.right_bis.linkedViewChanged(self.left, self.right_bis.XAxis)
        self.left_bis.linkedViewChanged(self.left, self.left_bis.XAxis)

    def add_item(
        self,
        axis_id: str,
        item: pg.PlotDataItem | pg.PlotCurveItem | pg.ScatterPlotItem,
    ) -> None:
        if axis_id not in self.axes:
            raise ValueError(f"The axis {axis_id} does not exist.")

        axis = self.axes[axis_id]["item"]
        vb = self.axes[axis_id]["vb"]

        if not axis.isVisible():
            axis.show()

        vb.addItem(item)
        self.axes[axis_id]["items_count"] += 1

    def remove_item(
        self,
        axis_id: str,
        item: pg.PlotDataItem | pg.PlotCurveItem | pg.ScatterPlotItem,
    ) -> None:
        if axis_id not in self.axes:
            raise ValueError(f"The axis {axis_id} does not exist.")

        axis = self.axes[axis_id]["item"]
        vb = self.axes[axis_id]["vb"]
        items_count = self.axes[axis_id]["items_count"]

        if not axis.isVisible() or items_count == 0:
            raise ValueError(f"The chosen axis {axis_id} is empty.")

        vb.removeItem(item)
        items_count -= 1
        self.axes[axis_id]["items_count"] = items_count

        if items_count == 0 and axis_id != "left":
            axis.hide()

@dataclass
class CalculationValues:
    curve: pg.PlotDataItem | pg.PlotCurveItem | pg.ScatterPlotItem
    min: pg.ScatterPlotItem
    max: pg.ScatterPlotItem

    def __hash__(self) -> int:
        return hash(self.curve)

    def hide(self) -> None:
        self.curve.hide()
        self.min.hide()
        self.max.hide()

    def show(self) -> None:
        self.curve.show()
        self.min.show()
        self.max.show()

# TODO Find a way for the dashboard to find their item when deleting
# 1) Store the item in dashboard ? Or,
# 2) Store the dashboard line id in the panel
class Panel(QuadrupleAxisPlotItem):
    item_count: int
    rotation_axes: tuple[str]
    rotation: bidict[str, CalculationValues]

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.item_count = 0

        self.rotation_axes = (
            "left",
            "right",
            "left_bis",
            "right_bis",
        )

        self.rotation = bidict()

        self.setLimits(xMin=0)

    def get_free_axis(self) -> str | None:
        for axis_id in self.rotation_axes:
            if axis_id not in self.rotation:
                return axis_id

        return None
    def update_y_axis_color(self, item: CalculationValues, color: str) -> None:
        axis_id = self.get_item_axis(item)
        if axis_id:
            axis = self.getAxis(axis_id)
            axis.setPen(pg.mkPen(color=color))
            
            # Manually update the tick labels with the new color
            ticks = axis.tickValues()
            for tick in ticks:
                for pos, label in tick:
                    text_item = pg.TextItem(text=label, color=color)
                    text_item.setPos(pos, 0)
                    axis.scene().addItem(text_item)

    def get_item_axis(self, item: CalculationValues) -> str | None:
        if item not in self.rotation.inverse:
            return None

        return self.rotation.inverse[item]

    def add_curve(self, item: CalculationValues) -> None:
        if self.item_count >= 4:
            raise ValueError("This Panel already has 4 curves")

        self.item_count += 1

        axis_to_be_added_to = self.get_free_axis()
        if axis_to_be_added_to is None:
            raise ValueError("This Panel already has 4 curves")

        self.rotation[axis_to_be_added_to] = item

        super().add_item(axis_to_be_added_to, item.curve)
        super().add_item(axis_to_be_added_to, item.min)
        super().add_item(axis_to_be_added_to, item.max)

    def remove_curve(self, item: CalculationValues) -> None:
        if self.item_count == 0:
            raise ValueError("This Panel does not have any curves")

        self.item_count -= 1

        axis_to_be_removed_from = self.get_item_axis(item)
        if axis_to_be_removed_from is None:
            raise ValueError("This curve is is not displayed in any axis")

        self.rotation.pop(axis_to_be_removed_from)

        super().remove_item(axis_to_be_removed_from, item.curve)
        super().remove_item(axis_to_be_removed_from, item.min)
        super().remove_item(axis_to_be_removed_from, item.max)

    def reset(self) -> None:
        for item in self.rotation.inverse.keys():
            self.remove_curve(item)

class PanelWidget(QtWidgets.QWidget):
    id: int
    panel: Panel

    def __init__(self, id: int) -> None:
        super().__init__()

        label = QtWidgets.QLabel(f"Panel {id}")
        plot_widget = pg.PlotWidget()

        layout = QtWidgets.QVBoxLayout()

        self.panel = Panel()

        plot_widget.setCentralItem(self.panel)
        self.panel.add_viewboxes_to_scene()

        layout.addWidget(label)
        layout.addWidget(plot_widget)

        self.setLayout(layout)


class SoundInformation(pg.GraphicsLayoutWidget):
    sound_data: Sound
    spectrogram_data: Spectrogram

    selection_region: pg.LinearRegionItem

    sound_plot: pg.PlotItem
    spectrogram_plot: pg.PlotItem

    sound_plot_data_item: pg.PlotDataItem
    spectrogram_image_item: pg.ImageItem

    reference_viewbox: pg.ViewBox

    def __init__(self) -> None:
        super().__init__()
        self.selection_region = pg.LinearRegionItem(swapMode="sort")

        self.sound_plot = pg.PlotItem()
        self.spectrogram_plot = pg.PlotItem()

        self.sound_plot.addItem(self.selection_region)
        self.sound_plot_data_item = self.sound_plot.plot()
        self.selection_region.setClipItem(self.sound_plot_data_item)

        self.spectrogram_image_item = display_spect.Spectrogram(zoom_blur=False)
        self.spectrogram_plot.addItem(self.spectrogram_image_item)

        self.reference_viewbox = self.sound_plot.getViewBox()

        self.sound_plot.setMouseEnabled(x=True, y=False)
        self.spectrogram_plot.setMouseEnabled(x=True, y=False)

        self.sound_plot.setLimits(xMin=0, yMin=-0.2, yMax=0.2)
        self.spectrogram_plot.setLimits(xMin=0, yMin=0, yMax=5000)
        self.spectrogram_plot.setRange(yRange=(0, 5000))

        self.sound_plot.setXLink(self.spectrogram_plot)
        self.spectrogram_plot.setXLink(self.sound_plot)

        self.selection_region.hide()
        self.spectrogram_plot.hide()
        self.setMinimumHeight(150)

        self.addItem(self.sound_plot)
        self.nextRow()
        self.addItem(self.spectrogram_plot)

        self.spectrogram_plot.getAxis('bottom').setHeight(0)
        self.spectrogram_plot.getAxis('bottom').hide()
    def toggle_spectrogram(self, show: bool) -> None:
        if show:
            self.spectrogram_plot.show()
        else:
            self.spectrogram_plot.hide()

    def set_data(self, data: Parselmouth) -> None:
        self.selection_region.show()

        sound = data.get_sound()
        spectrogram = data.get_spectrogram()

        self.sound_plot_data_item.setData(sound.timestamps, sound.amplitudes[0])

        self.sound_plot.setLimits(xMin=0, xMax=sound.timestamps[-1])

        self.spectrogram_plot.setLimits(xMin=0, xMax=sound.timestamps[-1])

        self.sound_plot.autoRange()
        self.spectrogram_image_item.set_data(
            spectrogram.frequencies, spectrogram.timestamps, spectrogram.data_matrix
        )

    def update_audio_waveform(self, audio_data):
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        time_axis = np.arange(len(audio_data)) / 44100.0
        
        self.sound_plot_data_item.setData(time_axis, audio_data)

        x_max = time_axis[-1]
        if x_max > self.sound_plot.viewRange()[0][1]:
            self.sound_plot.setXRange(0, x_max, padding=0)

class Interval:
    name: str
    parent_plot: pg.PlotItem

    start_line: pg.InfiniteLine
    end_line: pg.InfiniteLine
    text_item: pg.TextItem

    def __init__(
        self,
        interval: tgt.core.Interval,
        parent_plot: pg.PlotItem
    ) -> None:

        self.name = interval.text
        self.parent_plot = parent_plot

        self.start_line = pg.InfiniteLine(
            pos=interval.start_time,
            angle=90,
            pen=pg.mkPen("m", style=QtCore.Qt.DashLine, width=2),
        )

        self.end_line = pg.InfiniteLine(
            pos=interval.end_time,
            angle=90,
            pen=pg.mkPen("m", style=QtCore.Qt.DashLine, width=2),
        )

        mid_time = (interval.start_time + interval.end_time) / 2
        ymax = max(self.parent_plot.listDataItems()[0].yData) 
        
        self.text_item = pg.TextItem(interval.text, anchor=(0.5, 0.5), color="r")
        self.text_item.setPos(mid_time, ymax * 0.9)
        self.text_item.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))

    def add_to_plot_item(self) -> None:
        self.parent_plot.addItem(self.start_line)
        self.parent_plot.addItem(self.end_line)
        self.parent_plot.addItem(self.text_item)

    def removed_from_plot_item(self) -> None:
        self.parent_plot.removeItem(self.start_line)
        self.parent_plot.removeItem(self.end_line)
        self.parent_plot.removeItem(self.text_item)

    def __hash__(self) -> int:
        return hash(self.name)

class DisplayInterval:
    audio_widget: SoundInformation
    intervals: list[Interval]

    def __init__(self, audio_widget: SoundInformation) -> None:
        self.audio_widget = audio_widget
        self.intervals = []

    def display(self, tier: tgt.core.TextGrid) -> None:
        self.clear()

        for interval in tier:
            interv = Interval(interval, self.audio_widget.sound_plot)
            interv.add_to_plot_item()
            self.intervals.append(interv)

    def clear(self) -> None:
        for interval in self.intervals:
            interval.removed_from_plot_item()
        self.intervals.clear()
