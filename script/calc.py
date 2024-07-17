import scipy
import numpy as np

import parselmouth

def calc_formants(sound: parselmouth.Sound, start_time: float, end_time: float, energy_threshold: float = 40.0):
    formants = sound.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=5500, window_length=0.025, pre_emphasis_from=50)
    time_values = formants.ts()
    
    formant_values = {i: {time: formants.get_value_at_time(formant_number=i, time=time) for time in time_values if start_time <= time <= end_time} for i in range(1, 4)}
    
    intensities = sound.to_intensity()
    frame_energies = {time: intensities.get_value(time) for time in time_values}
    
    filtered_formant_values = {
        i: {time: value for time, value in formant_values[i].items() if frame_energies.get(time, 0) > energy_threshold}
        for i in range(1, 4)
    }
    
    time_values_filtered = sorted(filtered_formant_values[1].keys())
    resampled_formants = {i: np.array([filtered_formant_values[i][time] for time in time_values_filtered]) for i in range(1, 4)}
    
    return time_values_filtered, resampled_formants[1], resampled_formants[2], resampled_formants[3]


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
