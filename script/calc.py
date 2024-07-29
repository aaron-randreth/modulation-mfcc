import scipy
import numpy as np
import xarray as xr
import parselmouth

from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d
def calc_formants(sound: parselmouth.Sound, start_time: float, end_time: float, energy_threshold: float = 20.0):
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
def read_AG50x(path_to_pos_file):
    dims = ["x", "z", "y", "phi", "theta", "rms", "extra"]
    channel_sample_size = {
        8: 56,
        16: 112,
        32: 256
    }
    target_sample_rate = 200  
    pos_file = open(path_to_pos_file, mode="rb")
    file_content = pos_file.read()
    pos_file.seek(0)
    pos_file.readline()
    header_size = int(pos_file.readline().decode("utf8"))
    header_section = file_content[0:header_size]
    header = header_section.decode("utf8").split("\n")
    num_of_channels = int(header[2].split("=")[1])
    ema_samplerate = int(header[3].split("=")[1])
    data = file_content[header_size:]
    data = np.frombuffer(data, np.float32)
    data = np.reshape(data, newshape=(-1, channel_sample_size[num_of_channels]))
    pos = data.reshape(len(data), -1, 7)
    
    original_time = np.linspace(0, len(pos) / ema_samplerate, len(pos))
    
    new_time = np.arange(0, original_time[-1], 1 / target_sample_rate)
    
    interpolated_pos = np.zeros((len(new_time), pos.shape[1], pos.shape[2]))
    for i in range(pos.shape[1]): 
        for j in range(pos.shape[2]):  
            interp_func = interp1d(original_time, pos[:, i, j], kind='linear', fill_value="extrapolate")
            interpolated_pos[:, i, j] = interp_func(new_time)
    
    ema_data = xr.Dataset(
        data_vars=dict(ema=(["time", "channels", "dimensions"], interpolated_pos)),
        coords=dict(
            time=(["time"], new_time),
            channels=(["channels"], np.arange(pos.shape[1])),
            dimensions=(["dimensions"], dims)
        ),
        attrs=dict(
            device="AG50x",
            duration=new_time[-1],
            original_samplerate=ema_samplerate,
            resampled_samplerate=target_sample_rate
        )
    )
    return ema_data

def calculate_amplitude_envelope(signal, sample_rate=44100, frame_size=1024, step_size=0.005):
    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def lowpass_filter(data, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low')
        y = filtfilt(b, a, data)
        return y


    signal = signal / np.max(np.abs(signal))


    filtered_signal = bandpass_filter(signal, 700, 1300, sample_rate)


    analytic_signal = hilbert(filtered_signal)
    amplitude_envelope = np.abs(analytic_signal)


    smoothed_envelope = lowpass_filter(amplitude_envelope, 5, sample_rate)


    step_size_samples = int(step_size * sample_rate)
    

    envelope_with_step = smoothed_envelope[::step_size_samples]
    
    return envelope_with_step
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
