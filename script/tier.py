import statistics as stats
from typing import Callable
from PyQt5.QtCore import pyqtSignal as sig
import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import HoverEvent

class HoverSignalRegion(pg.LinearRegionItem):
    sigMouseHover = sig(HoverEvent)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def hoverEvent(self, event):
        self.sigMouseHover.emit(event)
        super().hoverEvent(event)

class Tier:
    region: pg.LinearRegionItem
    label: pg.TextItem
    neighboors: set[pg.LinearRegionItem]

    def __init__(self, start_time : float, end_time : float, label : str) -> None:
        tier_pen = pg.mkPen('g', width=2)
        #tier_brush = pg.mkBrush('b')
        tier_label_anchor = (0.5, 1)
        tier_label_color = (255, 255, 255)
        tier_label_font  = pg.QtGui.QFont("Arial", 20)

        self.label = pg.TextItem(text=label,
                                color=tier_label_color,
                                anchor=tier_label_anchor)
        self.label.setFont(tier_label_font) 
        self.label.setPos((start_time + end_time) / 2, -0.1) 

        # LinearREgionItem
        self.region = HoverSignalRegion(values=(start_time, end_time),
                                              pen=tier_pen)#, brush=tier_brush)
        self.region.sigRegionChanged.connect(lambda r:
                                                    self.__region_changed(r))
        self.region.sigRegionChangeFinished.connect(lambda r:
                                                    self.__region_change_finished(r))

        #self.region.sigMouseHover.connect(lambda x:)

        self.neighboors = set()

    def plot(self, plot) -> None:
        plot.addItem(self.region)
        plot.addItem(self.label)

    def get_times(self) -> tuple[float, float]:
        """
        @Returns The selected start and end time
        """
        return self.region.getRegion()

    def get_text(self) -> str:
        return self.label.toPlainText()

    def set_text(self, label : str) -> None:
        self.label.setPlainText(label)

    def add_neighboors(self, neighboors: list['Tier']) -> None:
        for n in neighboors:
            self.add_neighboor(n)

    def add_neighboor(self, neighboor: 'Tier') -> None:
        if id(neighboor.region) == id(self.region):
            return

        self.neighboors.add(neighboor.region)

    def remove_neighboor(self, neighboor: 'Tier') -> None:
        # Do nothing if neighboor not in set
        self.neighboors.discard(neighboor.region) 

    def __hash__(self) -> int: 
        return hash((id(self.region), id(self.label)))

    def __region_changed(self, region: pg.LinearRegionItem) -> None:
        # Memory address comparaison :
        if id(region) != id(self.region):
            return

        self.__center_label()

    def __has_overlap(self, other_region: pg.LinearRegionItem) -> bool:
        if id(other_region) == id(self.region):
            return False

        sstart, send = self.region.getRegion()
        ostart, oend = other_region.getRegion()

        return ostart <= sstart <= oend or ostart <= send <= oend

    def __correct_overlap(self, other_region: pg.LinearRegionItem) -> None:
        if id(other_region) == id(self.region):
            return

        sstart, send = self.region.getRegion()
        ostart, oend = other_region.getRegion()

        delta = 0
        if ostart <= sstart <= oend:
            delta = oend - sstart
        elif ostart <= send <= oend:
            delta = ostart - send

        self.region.setRegion((sstart + delta, send + delta))


    def __region_change_finished(self, region: pg.LinearRegionItem) -> None:
        # Memory address comparaison :
        if id(region) != id(self.region):
            return

        overlaps = [self.__has_overlap(n) for n in self.neighboors]
        nb_overlaps = overlaps.count(True)

        # If we are overlaping with more than 1 region
        # we let the use choose by themselves how to handle it,
        # as we would either need to move the other region or
        # change this region's duration, which would be annoying for
        # the user if done automatically
        if nb_overlaps != 1:
            return

        for i, n in enumerate(self.neighboors):
            if not overlaps[i]:
                continue

            self.__correct_overlap(n)
            return

    def __center_label(self) -> None:
        self.label.setPos(stats.mean(self.region.getRegion()), -0.1)
