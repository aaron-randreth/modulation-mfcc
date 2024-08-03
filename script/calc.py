import scipy
from parselmouth.praat import call
import numpy.typing as npt
from librosa.feature import rms,mfcc as mfccSpectr
from librosa.core import load as audioLoad
from librosa import pyin
import numpy as np
import xarray as xr
import parselmouth
import inspect
from scipy.signal import hilbert, butter, filtfilt, sosfiltfilt, savgol_filter, firwin,\
    find_peaks
from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d
from scipy import interpolate
import copy
def applyFilter(
                x,
                sr,
                /,*,
                filt:str='iir',
                cutOff:list|npt.NDArray=[None], 
                filtLen:int=6, 
                filtType:str='low', 
                polyOrd:int=3,
                coeffs:None|npt.NDArray=None
                ): 
    """
    apply low, high or band pass filter to input signal 'x' sampled at 'sr' Hertz 
    by using the kind of filter of length 'filtLen' defined by 'filt' (fir, iir or sg), 
    the cut off freqs defined by 'cutOff' and interpreted via 'filtType' as low, high 
    or band pass. Optionally the filter coefficient can be provided via 'coeffs'. 
    All methods avialable (and determined by the filter parameter) are based 
    on scipy.signal implementations.
    
    
    Input
    -------
        x (numpy array): input signal
        
        sr (positive float): sampling freq in Hertz
            
        filter (string or None, default= None): kind of filter to apply. One among None (no filter), iir (infinite response filter), 
                fir (finite inpulse respnse filter), sg (Savitsky Golay low pass filter).
        
        cutOff (list or np array with max two positive, monotonically increasing 
                arguments): cut off frequency/frequencies in Hertz. If it contains one 
                value, this will be the cut-off of a low pass filter, if two values
                these will be the cut-offs of a band pass filter. NOT USED WHEN FILT=sg
        
        filtLen (positive integer): length of the filter in number of samples. 
            When filter is sg (Savitsky Golay), filt len represents the length of the 
            time window
        
        filtType (string, default 'low'): kind of filter, one among 'low' (for low-pass),
        'band' for band-pass, 'high' for high-pass.
            
        polyOrd (positive integer): order of the polinomial used by sg (Savitsky Golay) filter
        
        coeffs (None or np array): precomputed filter coefficients. If provided these are directly applied.
    
    Ouput
    -------        
        np array: Filtered signal
    
    """
    # if len(cutOff)==1:
    #     cutOff=cutOff[0]
        
    if (filt is None) | (cutOff is None) | (cutOff is None):
        if cutOff is None:
            raise Exception('Cannot apply filter without specifying a cut Off freq. (CutOff is None).')
        else:
            raise Exception('Cannot apply filter without specifying a filter method among ''iir'', ''fir'' and '' sg'' (filt is None).')

    filtTypes=np.array(['bandpass','lowpass','highpass'])
    try:
        filtType=filtTypes[np.argwhere([x.startswith(filtType) for x in filtTypes]).flatten()][0]
    except:
        raise Exception('filtType must be one among: lowpass, highpass, bandpass. Partial matches allowed.')
    if any((sr/2)<=np.array(cutOff)):
        raise Exception('Cut off frequencies must be smaller than the half of the sampling freq. of the signal submitted to the filter')
    if (len(cutOff)>0) & (any(np.diff(cutOff)<=0)):
        raise Exception('If two cut off freqs are provided: cutOff[0]<cutOff[1]')
    
    cutOff=np.array(cutOff)
    if filt=='iir':
        if coeffs is None:
            w = cutOff / (sr / 2) 
            
            if ((len(cutOff)==1) and ((filtType=='lowpass') | (filtType=='highpass'))) |\
                ((len(cutOff)==2) and (filtType=='bandpass')) :
                
                sos = butter(filtLen, w, btype=filtType, output='sos')
            
            else:
                raise Exception('only one or two cut off frequencies allowed. If two freqs are provided, filtType must be ''bandpass''' )
        
        y=sosfiltfilt(sos, x)
        
    if filt=='fir':
        if coeffs is None:
            w = cutOff / (sr / 2)
            
            if ((len(cutOff)==1) and ((filtType=='lowpass') | (filtType=='highpass'))) |\
                ((len(cutOff)==2) and (filtType=='bandpass')) :
        
                bFil=firwin(filtLen, w, window=('kaiser', 7.4), pass_zero=filtType)
            
            else:
                raise Exception('only one or two cut off frequencies allowed. If two freqs are provided, filtType must be ''bandpass''' )

        
        y=filtfilt(bFil,1,x)
        
    if filt=='sg':
        if len(cutOff)==1:
            y = savgol_filter(x, filtLen, polyOrd, deriv=0, 
                                mode='interp')
        else:
            raise Exception('sg (savitsky Golay) filters can only be lowpass (one cutOff freq allowed)')
    
    return y

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

def calculate_amplitude_envelope(
                    x:npt.NDArray,
                    sr:float, 
                    /,*,
                    method:str='RMS', 
                    winLen:float=0.1, 
                    hopLen:float=0.01, 
                    center:bool=True, 
                    outFilter:None|str=None,
                    outFiltType:str='low',
                    outFiltCutOff:list|npt.NDArray=[12], 
                    outFiltLen:int=6,  
                    outFiltPolyOrd:int=3
                  ):
    
    """
    Get amplitude from input signal x sampled at frequency sr.
    Different methods can be used and the output can be submitted to a low pass 
    or a bandpass filter.
    
    Avialable methods are:
        RMS: classical Root mean square (it wraps librosa.features.rms) applied to consecutive windows of length 
            equal to winLen secs and spaced by hopLen secs
        RMSpraat: Root mean square applied by Praat to windows sized on the basis of the 
            minimum pitch 
        Hilb: absolute value of the signal's Hilbert transform 
        
    Input
    -------
        x (numpy array): input signal
        
        sr (positive float): sampling freq in Hertz
        
        method (string, default= 'Hilb'):  method to use one among : 'Hilb' 'RMS' 
        
        and 'RMSpraat' (RMS computed by praat)
        
        hopLen (positive float): hop size in secs used by all methods except 'Hilb' (also used for f0 compuation as required by method 'RMSpraat')
        
        winLen (positive float): window length in secs used by method 'RMS' (also used for f0 compuation as required by method 'RMSpraat')

        outFilter (string or None, default= None): filter to apply after computation
                of amplitude. One among None (no filter), iir (infinite response filter), 
                fir (finite inpulse respnse filter), sg (Savitsky Golay low pass filter).
        
        outFiltCutOff (list or np array with max two positive, monotonically increasing 
                arguments): cut off frequency/frequencies in Hertz. If it contains one 
                value, this will be the cut-off of a low pass filter, if two values
                these will be the cut-offs of a band pass filter. NOT USED WHEN FILT=sg
        
        outFiltLen (positive integer): length of the filter in number of samples. 
            When filter is sg (Savitsky Golay), filt len represents the length of the 
            time window
            
        outFiltPolyOrd (positive integer): order of the polinomial used by sg (Savitsky Golay) filter
        
        center (Boolean, default=True): center the result on its mean or not
         
    Ouput
    -------
        np array: Amplitude signal
    """
    
    if method=='Hilb': # amplitude via Hilbert transform
    
        amp=np.abs(hilbert(x))
        
        ampT=np.arange(len(x))/sr
        
        ampSr=sr
    
    elif method=='RMSpraat': # RMS informed by minimum pitch (calls praat)
        
        xObj=parselmouth.Sound(values= x,
                          sampling_frequency = sr, 
                          start_time = 0.0)
        
        tmpPitch = call(xObj, "To Pitch", hopLen, 50, 700)
        
        tmpPitch = tmpPitch.selected_array['frequency']
        
        tmpPitch = tmpPitch[tmpPitch > 20]
        
        quants = np.quantile(tmpPitch, [0.25, 0.75])
        
        tmpPitch = call(xObj, "To Pitch", hopLen,
                        0.75*quants[0], 2.5*quants[1])
        
        tmpPitch = tmpPitch.selected_array['frequency']
        
        if np.min(tmpPitch) > 120:
            
            amp = call(xObj, "To Intensity", np.min(
                tmpPitch), hopLen, 1)
        else:
            
            amp = call(xObj, "To Intensity", 120, 1/sr, 1)
        
        
        ampSr=1/amp.get_time_step()
        
        amp=amp.values.flatten()
        
        ampT=np.arange(len(amp))/ampSr
        
    elif method=='RMS':
        frLen=int(hopLen*sr)
        
        winLen=int(winLen*sr)
        
        amp=rms(y=x, frame_length=winLen, hop_length=frLen, center=center, pad_mode='constant').flatten()
    
    if (method!='hilb')& (method!='RMSpraat'):
        
        ampT=np.arange(len(amp))*hopLen
        
        ampSr=1/hopLen
    
    if outFilter is not None:
        
        amp=applyFilter(amp,ampSr,filt=outFilter,filtType=outFiltType,cutOff=outFiltCutOff, filtLen=outFiltLen, polyOrd=outFiltPolyOrd)
    
    return amp, ampT

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


def interp_NAN(
        X:npt.NDArray, 
        method:str='linear'
        ):
    """
    Interpolate NAN vales according to the method in the second argument
    this can be either pchip or whatever method accepted by scipy interp1d.
    
    Input
    ----------
        X (numpy array): input signal containing nan values
        method (str, default='linear'): interpolation method either a method from 
        scipy.interpolate.PchipInterpolator (e.g.: 'linear') or 'pchip'.
        
    Ouput
    -------
        np array: interpolated signal
    """
    newX = copy.copy(X)
    mynans = np.isnan(newX)
    if np.sum(mynans) == 0:
        return newX

    justnans = np.empty(np.size(X))
    justnans[:] = np.nan
    if method == 'pchip':
        if np.argwhere(mynans)[0] == 0:
            newX[0] = newX[np.argwhere(np.isnan(newX) == 0)[0]]
        if np.argwhere(mynans)[-1] == len(X)-1:
            newX[-1] = newX[np.argwhere(np.isnan(newX) == 0)[-1]]
        mynans = np.isnan(newX)
        f = interpolate.PchipInterpolator(
            np.where(mynans == 0)[0], newX[mynans == 0], extrapolate=False)
    else:
        f = interpolate.interp1d(np.where(mynans == 0)[
                                 0], newX[mynans == 0], method, fill_value="extrapolate")

    justnans[mynans] = f(np.squeeze(np.where(mynans)))

    newX[mynans] = justnans[mynans]
    return newX
def get_f0(x,
           sr,
           method:str='praatac', 
           hopSize:float=0.01,
           minPitch:float=75,
           maxPitch:float=600,
           interpUnvoiced:None|str="linear",

           outFilter:str='iir',
           outFiltType:str='low',
           outFiltCutOff:list|npt.NDArray=[None], 
           outFiltLen:int=6,  
           outFiltPolyOrd:int=3,
           
           minMaxQuant:None|list|npt.NDArray=None,maxCandNum:int=15,
           veryAccurate:bool=False,
           silenceThresh:float=0.03,
           voicingThresh:float=0.45,
           octaveCost:float=0.01,
           octaveJumpCost:float=0.35,
           voicedUnvoicedCost:float=0.14,

           pyinframe_length:int=2048,
           pyinwin_length:int=None,
           n_thresholds:int=100, 
           beta_parameters:tuple=(2, 18), 
           boltzmann_parameter:int=2, 
           resolution:float=0.1, 
           max_transition_rate:float=35.92, 
           switch_prob:float=0.01, 
           no_trough_prob:float=0.01, 
           pyinfill_na:float=np.nan, 
           pyincenter:bool=True, 
           pyinpad_mode:str='constant'
           ):
    """
    Compute f0 of input audio signal x sampled at frequency sr with step size 
    equal to hopSize secs according to three different methods: praat auto correlation,
    praat cross correlation and pyin as implemented in Librosa.
    
    Optionally the minimum and maximum f0 parameters can be adjusted after a first 
    estimation of the f0. This is done by setting them equal to two predetermined 
    quantiles (as defined in minMaxQuant) of the initial distribution of f0 vals.
    
    If interpUnvoiced is not None it must be a string indicating the method used 
    to interpolate the nan values corresponding to unvoiced portions of signal.
     
    If outFilter is not None, it must be a string indicatinf the kind of filter
    used to post process the f0 signal. Filter's parameters are provided via: 
    outFilter, outFiltCutOff, outFiltLen, outFiltPolyOrd.
    
    Input
    ----------
    x : one dimensional np array
        Input signal.
    sr : float
        sampling rate.
    method : str, optional
        f0 computation method. One among praatac, praatcc and pyin. The default is 'praatac'.
    hopSize : float, optional
        analysis hop size in seconds. The default is 0.01.
    minPitch : float, optional
        minimum allowed f0 value. The default is 75.
    maxPitch : float, optional
        Maximum allowed pitch. The default is 600.
    minMaxQuant : None|list|npt.NDArray of floats <1, optional
        either None or a sequence of two positive floats monotonically increasing. 
        If not None f0 is estimated a second time by using as minimum and maximum pitch
        the quantiles of the observed distribution of f0 values indicated by minMaxQuant. 
        The default is None.
    interpUnvoiced : None or string, optional
        If not None the string indicates the method to be used in interpolating
        unvoiced portions. Possible methods are 'pchip' and methods allowed by interp1d. 
        The default is False.
        
    ----------------------- post-processing filter's specific parameters
    outFilter : string or None, optional
        filter to apply after computation of f0. One among None (no filter), 
        iir (infinite response filter), fir (finite inpulse respnse filter), 
        sg (Savitsky Golay low pass filter). 
        If this is not none it replaces the low pass filter applied to the total 
        amount of MFCCs change in the original Goldstein's (2019) formulation.
        The default is 'iir'.
    outFiltCutOff, list or np array with max 2 positive, monotonically increasing elements: 
        cut off frequency/frequencies in Hertz. If it contains one value, this 
        will be the cut-off of a low pass filter, if two values theese will be 
        the cut-offs of a band pass filter. NOT USED WHEN outFilter=sg.
        The default is [None].
    outFiltLen, positive integer: 
        length of the filter in number of samples. When filter is sg (Savitsky Golay), 
        outFiltLen represents the length of the time window. The default is 6.        
    outFiltPolyOrd (positive integer): order of the polinomial used by sg 
        (Savitsky Golay) filter. The default is 3.
        
        ----------------------- PRAAT's specific parameters
    maxCandNum : int, optional
        maximum number of candidates (praat). The default is 15.
    veryAccurate : bool, optional
        Accuracy of the analysis (praat). The default is False.
    silenceThresh : positive float <1, optional
        Silence threshold (praat). The default is 0.03.
    voicingThresh : positive float <1, optional
        Voicing threshold (praat). The default is 0.45.
    octaveCost : float, optional
        Cost of selecting higher octaves (praat). The default is 0.01.
    octaveJumpCost : float, optional
        Cost for jumping from one oectave to the other (praat). The default is 0.35.
    voicedUnvoicedCost : float, optional
        Cost for switching between voiced and unvoiced (praat) . The default is 0.14.
        
        ----------------------- PYIN's specific parameters
    pyinframe_length : int, optional
        see frame_length in librosa.pyin for DESCRIPTION. The default is 2048.
    pyinwin_length : int, optional
       see win_length in librosa.pyin for DESCRIPTION. The default is None.
    n_thresholds : int, optional
        see n_thresholds in librosa.pyin for DESCRIPTION. The default is 100.
    beta_parameters : tuple, optional
        see beta_parameters in librosa.pyin for DESCRIPTION. The default is (2, 18).
    boltzmann_parameter : int, optional
        see boltzmann_parameter in librosa.pyin for DESCRIPTION. The default is 2.
    resolution : float, optional
        see resolution in librosa.pyin for DESCRIPTION. The default is 0.1.
    max_transition_rate : float, optional
        see max_transition_rate in librosa.pyin for DESCRIPTION. The default is 35.92.
    switch_prob : float, optional
        see switch_prob in librosa.pyin for DESCRIPTION. The default is 0.01.
    no_trough_prob : float, optional
        see no_trough_prob in librosa.pyin for DESCRIPTION. The default is 0.01.
    pyinfill_na : float, optional
        see pyinfill_na in librosa.pyin for DESCRIPTION. The default is np.nan.
    pyincenter : bool, optional
        see center in librosa.pyin for DESCRIPTION. The default is True.
    pyinpad_mode : str, optional
        see pad_mode in librosa.pyin for DESCRIPTION. The default is 'constant'.
    
    Output
    -------
    f0 : np.array
        sequence of f0 values.
    f0t : np.array
        sequence of time stamps.

    """
    if (interpUnvoiced is None) & (outFilter is not None):
        raise Exception(inspect.cleandoc("""Post processing filters should be applied (outFiltes is not None) \
        but unvoiced regions are not interpolated (interpUnvoiced is None).
        Cannot filter f0 signal with gaps due to unvoiced regions"""))

    if (method=='praatac') |(method=='praatcc'):
        if method=='praatac':
            myMethStr="To Pitch (ac)"
        else:
            myMethStr="To Pitch (cc)"

        xObj=parselmouth.Sound(values= x,
                          sampling_frequency = sr, 
                          start_time = 0.0)
        
        f0obj= call(xObj, myMethStr, hopSize, minPitch, maxCandNum,veryAccurate,
             silenceThresh, voicingThresh, octaveCost,octaveJumpCost,voicedUnvoicedCost,maxPitch)
        
        if minMaxQuant is not None:
            
            f0=f0obj.selected_array['frequency']
            
            f0 = f0[f0 > 20]
            quants = np.quantile(f0, [minMaxQuant[0], minMaxQuant[1]])
            
            f0obj = f0obj= call(xObj, "To Pitch (ac)", hopSize, quants[0], maxCandNum,veryAccurate,
                 silenceThresh, voicingThresh, octaveCost,octaveJumpCost,voicedUnvoicedCost,quants[1])
            
        f0=f0obj.selected_array['frequency']   
        f0[f0<=20]=np.nan
        f0t=np.arange(len(f0))*hopSize
    
    if method=='pyin':
        hop_length=int(hopSize*sr)
        f0,_voiceFlag,voiceProb=pyin(x, fmin=minPitch, fmax=maxPitch, sr=sr, frame_length=pyinframe_length, 
                     win_length=pyinwin_length, hop_length=hop_length, n_thresholds=n_thresholds, 
                     beta_parameters=beta_parameters, boltzmann_parameter=boltzmann_parameter,
                     resolution=resolution, max_transition_rate=max_transition_rate,
                     switch_prob=switch_prob, no_trough_prob=no_trough_prob, fill_na=pyinfill_na,
                     center=pyincenter, pad_mode=pyinpad_mode)
    
        if minMaxQuant is not None:
                        
            f0 = f0[np.isnan(f0)==0]
            quants = np.quantile(f0, [minMaxQuant[0], minMaxQuant[1]])
            
            f0,_voiceFlag,voiceProb=pyin(x, fmin=quants[0], fmax=quants[1], sr=sr, frame_length=pyinframe_length, 
                    win_length=pyinwin_length, hop_length=hop_length, n_thresholds=n_thresholds, 
                    beta_parameters=beta_parameters, boltzmann_parameter=boltzmann_parameter,
                    resolution=resolution, max_transition_rate=max_transition_rate,
                    switch_prob=switch_prob, no_trough_prob=no_trough_prob, fill_na=pyinfill_na,
                    center=pyincenter, pad_mode=pyinpad_mode)
         
        f0t=np.arange(len(f0))*hopSize

    if interpUnvoiced is not None:
        
       f0=interp_NAN(f0,interpUnvoiced)
        
    if outFilter is not None:
        f0=applyFilter(f0,1/hopSize,filt=outFilter,cutOff=outFiltCutOff, filtLen=outFiltLen, filtType=outFiltType,polyOrd=outFiltPolyOrd)
    
    return f0,f0t
