from abc import ABC, abstractmethod
from typing import override

import numpy
import numpy.typing as npt
import scipy

from .datasource import DataSource


def analyse_maximum(
    times: numpy.typing.NDArray[numpy.float64],
    values: numpy.typing.NDArray[numpy.float64],
) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
    """
    Return the peak of a signal.

    Parameters
    ----------

    values: ndarray
        Array containing the values of the signal.

    times: ndarray
        Array containing the time for each value of the signal.

    Return
    ------

    peak_times_final : ndarray
        The times at which the peaks where found.

    peak_values_final : ndarray
        The values of the peaks.

    """

    min_peaks, _ = scipy.signal.find_peaks(-values)

    initial_peaks, _ = scipy.signal.find_peaks(values)

    if len(initial_peaks) == 0:
        return numpy.asarray([]), numpy.asarray([])

    if len(initial_peaks) == 1:
        peak_times_final = times[initial_peaks]
        peak_values_final = values[initial_peaks]

        return peak_times_final, peak_values_final

    peak_times = times[initial_peaks]
    time_gaps = numpy.diff(peak_times)

    q75, q25 = numpy.percentile(time_gaps, [100, 10])
    iqr = q75 - q25

    min_distance_time = iqr - 0.03
    min_distance_samples = max(
        int(min_distance_time * len(values) / (times[-1] - times[0])),
        1,
    )

    peaks, _ = scipy.signal.find_peaks(
        values, distance=min_distance_samples, prominence=1
    )
    peak_times_final = times[peaks]
    peak_values_final = values[peaks]

    if len(peak_times_final) == 0 or len(peak_values_final) == 0:
        return peak_times_final, peak_values_final

    last_max_index = peaks[-1]
    last_min_index = min_peaks[-1]

    if last_max_index > last_min_index:
        print("Le dernier maximum dépasse le dernier minimum.")
        peak_times_final = numpy.delete(peak_times_final, -1)
        peak_values_final = numpy.delete(peak_values_final, -1)

    return peak_times_final, peak_values_final


# sampling_frequency: float = 200,
# start_time = float(selected_interval.start_time)
# end_time = float(selected_interval.end_time)

# start_index = int(start_time * fs)
# end_index = int(end_time * fs)

# start_index = max(start_index, 0)
# end_index = min(end_index, len(values))

# interval_time = times[start_index:end_index]
# interval_values = values[start_index:end_index]


class PeakSource(DataSource):
    signal: DataSource
    times: npt.NDArray
    values: npt.NDArray

    def __init__(self, signal: DataSource) -> None:
        self.signal = signal
        self.update()

    @override
    def get_data(self) -> tuple[npt.NDArray[numpy.float64], npt.NDArray[numpy.float64], npt.NDArray[numpy.float64]]:
        if self.signal.file_path is None:
            raise ValueError("The file for this source was not initalized")

        return self.times, self.values

    @override
    def get_subset(self, start_idx: int, end_idx: int) -> "DataSource":
        pass

    @override
    def change_file(self, file_path: str) -> None:
        self.signal.change_file(file_path)

    @override
    def update(self) -> None:
        self.signal.update()
        self.times, self.values = analyse_maximum(*self.signal.get_data())
