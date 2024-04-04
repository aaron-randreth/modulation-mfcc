from functools import total_ordering
from typing import override
from dataclasses import dataclass

@total_ordering
@dataclass
class Marker:
    position: float
    name: str = ""
    rounding_digits: int = 5

    def __post_init__(self):
        self.position = float(self.position)

    def __lt__(self, other: "Marker") -> bool:
        return self.position < other.position

    def __eq__(self, other: "Marker") -> bool:
        self_pos = round(self.position, self.rounding_digits)
        other_pos = round(other.position, self.rounding_digits)

        return self_pos == other_pos

    def __hash__(self):
        return hash(id(self))

    def __float__(self):
        return self.position

    def __str__(self):
        return f"{self.name} - Position: {self.position}"

    def has_name(self) -> bool:
        return self.name != ""

    def compare_position(self, other_position: float) -> bool:
        self_pos = round(self.position, self.rounding_digits)
        other_pos = round(other_position, self.rounding_digits)

        return self_pos == other_pos


class MarkerList:
    elements: list[Marker]

    def __init__(self):
        self.elements = []

    def __repr__(self):
        return str(self.elements)

    def __contains__(self, element: Marker) -> bool:
        return element.position in (m.position for m in self.elements)

    def add_marker(self, marker: Marker) -> Marker:
        if marker in self:
            m = self.elements[self.elements.index(marker)]
            m.name = marker.name
            return m

        self.elements.append(marker)
        self.notify_marker_changed()

        return marker

    def remove_marker(self, marker: Marker) -> Marker:
        self.elements.remove(marker)
        self.notify_marker_changed()
        return marker

    def remove_marker_by_idx(self, marker_idx: int) -> Marker:
        res = self.elements.pop(marker_idx)
        self.notify_marker_changed()
        return res

    def get_marker(self, marker_idx: int) -> Marker:
        return self.elements[marker_idx]

    def get_marker_idx(self, marker: Marker) -> int:
        return self.elements.index(marker)

    def get_markers(self) -> list[Marker]:
        return self.elements.copy()

    def notify_marker_changed(self) -> None:
        self.elements.sort()

@dataclass
class IntervalMarker:
    start_time: Marker
    end_time: Marker

    @classmethod
    def new_interval(
        cls, start_time: float, end_time: float, interval_label: str = ""
    ) -> "IntervalMarker":
        start = Marker(start_time, interval_label)
        end = Marker(end_time)

        return cls(start, end)

    def __post_init__(self):
        if self.start_time == self.end_time:
            msg = "The start and end time for the interval cannot be equal."
            raise ValueError(msg)

        if self.start_time > self.end_time:
            msg = "The start time for the interval cannot be after the end time."
            raise ValueError(msg)

    def __hash__(self) -> int:
        return int(hash(self.start_time) + hash(self.end_time))

    def __repr__(self) -> str:
        return f"{start_time} {end_time}"

    def get_name(self) -> str:
        return self.start_time.name

    def set_name(self, new_name: str) -> None:
        self.start_time.name = new_name


class IntervalMarkerList(MarkerList):

    @override
    def remove_marker(self, marker: Marker) -> Marker:
        marker_idx = self.elements.index(marker)
        self.remove_marker_by_idx(marker_idx)

    @override
    def remove_marker_by_idx(self, marker_idx: int) -> Marker:
        marker_to_remove = super().remove_marker_by_idx(marker_idx)

        if marker_idx == 0:
            return marker_to_remove

        if not marker_to_remove.has_name():
            return marker_to_remove

        previous_marker_idx = marker_idx % len(self.elements)
        self.elements[previous_marker_idx].name += marker_to_remove.name

    def add_interval(self, interval: IntervalMarker):
        has_marker_between = any(
            (interval.start_time < m < interval.end_time for m in self.elements)
        )

        if has_marker_between:
            raise ValueError("Impossible to add interval")

        interval.start_time = self.add_marker(interval.start_time)
        interval.end_time = self.add_marker(interval.end_time)

    def get_interval(self, interval_idx: int) -> IntervalMarker:

        index = interval_idx % len(self.elements)

        start_time = self.get_marker(index)
        end_time = self.get_marker(index + 1)

        return IntervalMarker(start_time, end_time)

    def get_intervals(self) -> list[IntervalMarker]:
        markers = self.get_markers()

        m1 = markers
        m2 = markers[1:]
        intervals = map(IntervalMarker, m1, m2)

        return list(intervals)
