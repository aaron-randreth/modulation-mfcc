from typing import Callable, override
from abc import ABC, abstractmethod
from enum import Enum

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from pyqtgraph import PlotWidget
import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import HoverEvent

from bidict import bidict

from .markers import (
        Marker, IntervalMarker,
        MarkerList, IntervalMarkerList
)


class TierType(Enum):
    INTERVAL_TIER = (0,)
    POINT_TIER = 1


THEME_PEN = pg.mkPen("b", width=2)

class Tier(PlotWidget):
    __name: str
    __start_time: float
    __end_time: float
    __tier_type: TierType.INTERVAL_TIER
    __converter = None  # TODO Type hint this
    # (old position, new position)
    ELEMENT_POSITION_CHANGED = pyqtSignal(float, float)

    def __init__(
        self,
        name: str,
        tier_type: TierType,
        start_time: float,
        end_time: float,
        converter,
    ):
        super().__init__()

        self.__name = name
        self.__start_time = start_time
        self.__end_time = end_time

        self.__converter = converter

        self.getAxis("left").setStyle(showValues=False, tickAlpha=0)

        self.setMouseEnabled(y=False)
        self.setYRange(0, 1)

        self.setFixedHeight(200)
        self.setXRange(self.__start_time, self.__end_time)
        self.setLabel("bottom", "Temps", units="s")

    def __eq__(self, other: "Tier") -> bool:
        return self is other

    def __repr__(self) -> str:
        return f"'name : {self.__name}, limits: {self.__start_time} - {self.__end_time}'"

    def get_name(self) -> str:
        return self.__name

    @abstractmethod
    def add_element(self, marker: Marker):
        pass

    @abstractmethod
    def remove_element(self, index: int) -> None:
        pass

    @abstractmethod
    def get_element(self, index: int) -> Marker:
        pass

    @abstractmethod
    def get_elements(self) -> list[Marker]:
        pass

    @abstractmethod
    def change_element_position(self, marker: Marker) -> None:
        pass

    def to_textgrid(self):
        return self.__converter.to_textgrid(self)

    def get_name(self) -> str:
        return self.__name

    def get_start_time(self) -> float:
        return self.__start_time

    def get_end_time(self) -> float:
        return self.__end_time


class PointTier(Tier):
    mlist: MarkerList
    marker_to_display: bidict[pg.InfiniteLine, Marker]
    hovered_line: pg.InfiniteLine | None

    def __init__(self, name: str, start_time: float, end_time: float, converter):
        super().__init__(name, TierType.POINT_TIER, start_time, end_time, converter)

        self.mlist = MarkerList()
        self.marker_to_display = bidict()
        self.hovered_line = None

        self.scene().sigMouseHover.connect(self.mouse_moved)

    def mouse_moved(self, hover_items):
        self.hovered_line = next(
            (el for el in hover_items if type(el) == pg.InfiniteLine), None
        )

    @override
    def add_element(self, element: Marker):
        if element in self.mlist:
            return

        element = self.mlist.add_marker(element)
        element_line = pg.InfiniteLine(
            pos=element.position,
            label=element.name,
            labelOpts={"color": (0, 0, 0)},
            pen=THEME_PEN,
            movable=True,
        )

        self.addItem(element_line)
        self.marker_to_display[element_line] = element

        element_line.sigPositionChangeFinished.connect(
            lambda l: self.change_element_position(self.marker_to_display[l], l.value())
        )

    @override
    def remove_element_by_idx(self, index: int) -> None:
        removed_marker = self.mlist.remove_marker_by_idx(index)
        self.remove_element(removed_marker)

    @override
    def remove_element(self, element: Marker) -> None:
        marker_line = self.marker_to_display.inverse[element]
        self.removeItem(marker_line)

    @override
    def get_element(self, index: int) -> Marker:
        return self.mlist.get_marker(index)

    @override
    def get_elements(self) -> list[Marker]:
        return self.mlist.get_markers()

    @override
    def change_element_position(self, marker: Marker, new_value: float) -> None:
        previous_value = marker.position
        marker.position = new_value
        self.mlist.notify_marker_changed()
        self.ELEMENT_POSITION_CHANGED.emit(previous_value, new_value)

    @override
    def keyPressEvent(self, event):
        super().keyPressEvent(event)

        if self.hovered_line is None:
            return

        line = self.hovered_line

        old_text = line.label.toPlainText()

        if event.key() == Qt.Key_unknown:
            return

        if event.key() == Qt.Key_Backspace:
            line.label.setFormat(old_text[:-1])
        elif event.key():
            line.label.setFormat(old_text + event.text())

        self.marker_to_display[line].name = (
            line.label.toPlainText()
        )

class IntervalTier(Tier):
    mlist: IntervalMarker
    marker_to_display: bidict[Marker, pg.InfiniteLine]
    display_change_events: dict[pg.InfiniteLine]
    marker_label: dict[Marker, pg.TextItem]
    last_mouse_position = None

    def __init__(self, name: str, start_time: float, end_time: float, converter):
        super().__init__(name, TierType.POINT_TIER, start_time, end_time, converter)

        self.mlist = IntervalMarkerList()
        self.marker_to_display = bidict()
        self.display_change_events = {}
        self.marker_label = {}
        self.last_mouse_position = None

        self.add_element(
            IntervalMarker.new_interval(start_time, end_time), movable=False
        )

        self.scene().sigMouseMoved.connect(self.mouse_moved)

    def mouse_moved(self, evt):
        self.last_mouse_position = evt

    @override
    def keyPressEvent(self, event):
        super().keyPressEvent(event)

        if self.last_mouse_position is None:
            return

        if not self.plotItem.vb.sceneBoundingRect().contains(self.last_mouse_position):
            return

        my = self.plotItem.vb.mapSceneToView(self.last_mouse_position).x()

        last_smaller = max(
            (m for m in self.mlist.get_markers() if m.position <= my), default=None
        )

        text_label = self.marker_label[last_smaller]

        old_text = text_label.toPlainText()

        if event.key() == Qt.Key_unknown:
            return

        if event.key() == Qt.Key_Backspace:
            text_label.setPlainText(old_text[:-1])
        elif event.key():
            text_label.setPlainText(old_text + event.text())

        last_smaller.name = text_label.toPlainText()

    def __change_pos(self, line):
        m1 = self.marker_to_display.inverse[line]
        m1.position = line.value()
        self.mlist.notify_marker_changed()
        self.ELEMENT_POSITION_CHANGED.emit(previous_value, new_value)

    def __create_line(self, marker: Marker, movable: bool = True) -> pg.InfiniteLine:
        if marker in self.marker_to_display:
            return self.marker_to_display[marker]

        same_pos = filter(
            lambda l: marker.compare_position(l.value()), self.marker_to_display.inverse
        )

        same_pos = list(same_pos)

        if len(same_pos) > 0:
            return same_pos[0]

        element_line = pg.InfiniteLine(
            pos=marker.position, pen=THEME_PEN, movable=movable
        )

        self.addItem(element_line)
        self.marker_to_display[marker] = element_line

        element_line.sigPositionChanged.connect(
            lambda l: self.change_element_position(
                self.marker_to_display.inverse[l], l.value()
            )
        )

        return element_line

    def __create_label(self, marker: Marker) -> pg.TextItem:
        if marker in self.marker_label:
            text_item = self.marker_label[marker]
            text_item.setPlainText(marker.name)
            return

        marker_idx = self.mlist.get_marker_idx(marker)

        if marker_idx >= len(self.mlist.get_markers()) - 1:
            return

        text_item = pg.TextItem(text=marker.name, color=(0, 0, 0), anchor=(0.5, 1))
        text_item.setFont(pg.QtGui.QFont("Arial", 14))

        self.addItem(text_item)
        self.marker_label[marker] = text_item

        self.__config_text_pos_change(marker)

        return text_item

    def __config_text_pos_change(self, marker: Marker) -> None:
        line = self.marker_to_display[marker]
        marker_idx = self.mlist.get_marker_idx(marker)

        text_item = self.marker_label[marker]

        neighboor = self.mlist.get_marker(marker_idx + 1)
        nline = self.marker_to_display[neighboor]

        text_item.setPos((line.value() + nline.value()) / 2, 0.5)
        line.sigPositionChanged.connect(
            lambda l: text_item.setPos((line.value() + nline.value()) / 2, 0.5)
        )
        nline.sigPositionChanged.connect(
            lambda l: text_item.setPos((line.value() + nline.value()) / 2, 0.5)
        )

    def __config_event_listeners(self):
        for m, text_item in self.marker_label.items():
            l = self.marker_to_display[m]
            self.__config_text_pos_change(m)

    @override
    def add_element(self, element: IntervalMarker, movable: bool = True):
        self.mlist.add_interval(element)

        l1 = self.__create_line(element.start_time, movable)
        l2 = self.__create_line(element.end_time, movable)

        tl = self.__create_label(element.start_time)
        t2 = self.__create_label(element.end_time)

        self.__config_event_listeners()

    @override
    def remove_element_by_idx(self, index: int) -> None:
        removed_marker = self.mlist.remove_marker_by_idx(index)
        self.remove_element(removed_marker)

    @override
    def remove_element(self, element: Marker) -> None:
        marker_line = self.marker_to_display.pop(element)
        self.removeItem(marker_line)

    @override
    def get_element(self, index: int) -> IntervalMarker:
        return self.mlist.get_interval(index)

    @override
    def get_elements(self) -> list[IntervalMarker]:
        return self.mlist.get_intervals()

    @override
    def change_element_position(self, marker: Marker, new_value: float) -> None:
        marker_idx = self.mlist.get_marker_idx(marker)
        next_marker = self.mlist.get_marker(marker_idx + 1)
        previous_marker = self.mlist.get_marker(marker_idx - 1)

        min_interval_duration = 0.005
        if new_value >= next_marker.position:
            self.marker_to_display[marker].setValue(
                next_marker.position - min_interval_duration
            )
            return

        if new_value <= previous_marker.position:
            self.marker_to_display[marker].setValue(
                previous_marker.position + min_interval_duration
            )
            return

        previous_value = marker.position
        marker.position = new_value
        self.mlist.notify_marker_changed()
        self.ELEMENT_POSITION_CHANGED.emit(previous_value, new_value)

class TextGrid(QWidget):
    linked_plot: PlotWidget
    __internal_vb: pg.ViewBox
    __converter = None  # Type hint
    layout: QVBoxLayout
    tiers: list[Tier]

    def __init__(self, linked_plot: PlotWidget, converter):
        super().__init__()

        self.tiers = []
        self.linked_plot = linked_plot
        self.__internal_vb = linked_plot
        self.__converter = converter

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

    def __link_views(self):
        (xmin, xmax), (ymin, ymax) = self.__internal_vb.viewRange()
        for t in self.tiers:
            t.setXLink(self.__internal_vb)
            t.setLimits(xMin=xmin, xMax=xmax)

    # insert at the end of all negative values
    def add_tier(self, new_tier: Tier, tier_index: int = -1) -> None:

        nb_tiers = self.layout.count()
        if tier_index >= nb_tiers:
            msg = f"Invalid tier_index {tier_index} for nb tiers: {nb_tiers}."
            raise ValueError(msg)

        # Inserts at the end if negative (see docs)
        self.layout.insertWidget(tier_index, new_tier)

        if tier_index < 0:
            tier_index = nb_tiers

        # -1 will insert before the last element
        self.tiers.insert(tier_index, new_tier)

        self.__link_views()

    def remove_tier_by_idx(self, tier_index: int) -> None:
        if tier_index >= len(self.tiers):
            msg = f"Invalid tier index {tier_index} for nb tiers: {len(self.tiers)}."
            raise ValueError(msg)

        tier_index = tier_index % len(self.tiers)

        removed_tier = self.tiers.pop(tier_index)
        item = self.layout.takeAt(tier_index)

        if item is None or item.widget() is None:
            return

        item.widget().deleteLater()

        self.__link_views()

    def get_tiers(self) -> list[Tier]:
        return self.tiers.copy()

    def get_tiers_by_name(self, tier_name: str) -> list[Tier]:
        if not tier_name:
            raise ValueError("The given tier_name was empty.")

        return [t for t in self.tiers if t.get_name() == tier_name]

    def get_tier_by_index(self, tier_index: int) -> Tier:
        if tier_index >= len(self.tiers) or abs(tier_index) - 1 >= len(self.tiers):
            msg = f"Invalid tier index {tier_index} for nb tiers: {len(self.tiers)}."
            raise ValueError(msg)

        return self.tiers[tier_index]

    def get_tier_index(self, tier: Tier) -> int | None:
        for i, t in enumerate(self.get_tiers()):
            if t != tier:
                continue
            return i
        
        return None

    def to_textgrid(self):
        return self.__converter.to_textgrid(self)

# Points
# - Ajout d'un marker nomé
# - Suppression d'un marker nomé
# - Déplacement d'un marker nomé, n'importe où dans le tier

# Intervales
# - Ajout de 1/2 markers nomé
# - Suppression d'un marker nomé
# - Déplacement d'un marker nomé, <= prochain marker
