import scipy
import numpy as np

import parselmouth

def calc_formants(sound: parselmouth.Sound, start_time: float, end_time: float):
    formants = sound.to_formant_burg()
    time_values = formants.ts()
    formant_values = {i: {time: formants.get_value_at_time(formant_number=i, time=time) for time in time_values} for i in range(1, 4)}
    preserved_formants = {i: {time: formant_values[i][time] for time in time_values if start_time <= time <= end_time} for i in range(1, 4)}
    interpolated_formants = {}
    smoothed_formants = {}
    resampled_formants = {}

    for i in range(1, 4):
        interp_func = scipy.interpolate.interp1d(list(preserved_formants[i].keys()), list(preserved_formants[i].values()), kind='linear')
        time_values_interp = np.linspace(min(preserved_formants[i].keys()), max(preserved_formants[i].keys()), num=1000)
        interpolated_formants[i] = interp_func(time_values_interp)
        smoothed_formants[i] = scipy.signal.savgol_filter(interpolated_formants[i], window_length=101, polyorder=1)
        new_time_values = np.arange(start_time, end_time, 1/200.0)
        interp_func_resampled = scipy.interpolate.interp1d(time_values_interp, smoothed_formants[i], kind='linear', fill_value="extrapolate")
        resampled_formants[i] = interp_func_resampled(new_time_values)

    return new_time_values, resampled_formants[1], resampled_formants[2], resampled_formants[3]

class MinMaxFinder:
    def find_in_interval(self, times: list[float], values: list[float], interval: tuple[float, float]) -> tuple[np.ndarray[float], np.ndarray[float]]:
        start, end = interval
        interval_times = []
        interval_values = []
        for time, value in zip(times, values):
            in_interval: bool = start <= time and time <= end
            if not in_interval:
                continue
            interval_times.append(time)
            interval_values.append(value)
        return np.array(interval_times), np.array(interval_values)

    def analyse_minimum(self, x, y, interval):
        if interval is None:
            print("No interval specified.")
            return [], []
        interval_times, interval_values = self.find_in_interval(x, y, interval)
        min_peaks, _ = scipy.signal.find_peaks(-interval_values)
        if len(min_peaks) == 0:
            return [], []
        min_times = interval_times[min_peaks]
        min_values = interval_values[min_peaks]
        return min_times, min_values

    def analyse_maximum(self, x, y, interval):
        if interval is None:
            print("No interval specified.")
            return [], []
        interval_times, interval_values = self.find_in_interval(x, y, interval)
        max_peaks, _ = scipy.signal.find_peaks(interval_values)
        if len(max_peaks) == 0:
            return [], []
        max_times = interval_times[max_peaks]
        max_values = interval_values[max_peaks]
        return max_times, max_values
