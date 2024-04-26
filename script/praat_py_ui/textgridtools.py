from abc import ABC, abstractmethod
from typing import Any, override

import tgt

from .markers import Marker, IntervalMarker, MarkerList, IntervalMarkerList

from .tiers import (
    TextGrid,
    PointTier,
    IntervalTier,
)


class TextgridConverter(ABC):

    @abstractmethod
    def to_textgrid(self, to_convert: Any) -> Any:
        pass

    @abstractmethod
    def from_textgrid(self, textgrid: Any) -> Any:
        pass


class PointTierTGTConvert(TextgridConverter):

    @override
    def to_textgrid(self, display_point: PointTier) -> tgt.core.PointTier:
        tgt_tier = tgt.core.PointTier(
            display_point.get_start_time(),
            display_point.get_end_time(),
            display_point.get_name(),
        )

        for point in display_point.get_elements():
            tgt_point = tgt.core.Point(point, point.name)
            tgt_tier.add_point(tgt_point)

        return tgt_tier

    @override
    def from_textgrid(self, pt: tgt.core.PointTier) -> PointTier:
        point_tier = PointTier(pt.name, pt.start_time, pt.end_time, self)

        for p in pt.points:
            point = Marker(p.time, p.text)
            point_tier.add_element(point)

        return point_tier


class IntervalTierTGTConvert(TextgridConverter):

    @override
    def to_textgrid(self, display_interval: IntervalTier):
        tgt_tier = tgt.core.IntervalTier(
            display_interval.get_start_time(),
            display_interval.get_end_time(),
            display_interval.get_name(),
        )

        for interval in display_interval.get_elements():
            tgt_interval = tgt.core.Interval(
                interval.start_time, interval.end_time, interval.get_name()
            )
            tgt_tier.add_interval(tgt_interval)

        return tgt_tier

    @override
    def from_textgrid(self, it: tgt.core.IntervalTier) -> IntervalTier:
        interval_tier = IntervalTier(it.name, it.start_time, it.end_time, self)

        for el in it.intervals:
            interval = IntervalMarker.new_interval(el.start_time, el.end_time, el.text)
            interval_tier.add_element(interval)

        return interval_tier


class TextgridTGTConvert(TextgridConverter):
    valid_tier_types: dict[str, TextgridConverter]

    def __init__(self):
        self.valid_tier_types = {}

        self.valid_tier_types["IntervalTier"] = IntervalTierTGTConvert()
        self.valid_tier_types["TextTier"] = PointTierTGTConvert()

    @override
    def to_textgrid(self, textgrid: TextGrid) -> tgt.core.TextGrid:
        tgt_textgrid = tgt.core.TextGrid()

        for tier in textgrid.get_tiers():
            tgt_tier = tier.to_textgrid()
            tgt_textgrid.add_tier(tgt_tier)

        return tgt_textgrid

    @override
    def from_textgrid(self, tg: tgt.core.TextGrid, linkedplot) -> TextGrid:
        textgrid = TextGrid(linkedplot, self)

        for t in tg.tiers:
            if not t.tier_type() in self.valid_tier_types:
                continue

            tier_converter = self.valid_tier_types[t.tier_type()]
            tier = tier_converter.from_textgrid(t)
            textgrid.add_tier(tier)

        return textgrid
