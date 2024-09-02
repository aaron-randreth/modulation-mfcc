from typing import override

from math import sqrt

from scipy import signal
from scipy.signal import argrelextrema, find_peaks

import librosa
import numpy as np
import numpy
import numpy.typing as npt
import numpy.typing as npt
import numpy as np
from scipy.signal import hilbert, butter, filtfilt, sosfiltfilt, savgol_filter, firwin,\
    find_peaks, decimate 
from scipy import interpolate
from scipy.stats import zscore
from parselmouth.praat import call
from librosa.feature import rms,mfcc as mfccSpectr
from librosa.core import load as audioLoad
from librosa.core.convert import fft_frequencies
from librosa import pyin,stft
import parselmouth
import copy

import inspect
import pyqtgraph as pg
#scripts par Leonardo Lancia
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

def get_amplitude(
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


def load_channel(
    file_path: str, signal_sample_rate: float = 10_000, channel_nb: int = 0
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Return the audio data of single channel of a file using librosa.

    Parameters
    ----------

    signal_sample_data: float, optional
        The sample rate of the file.

    channel_nb: int, optional
        The channel to return. (the default is the first one)

    Return
    ------

    audio_data: ndarray
        The data of the chosen channel.

    """
    audio_data, _ = librosa.load(file_path, sr=signal_sample_rate, mono=False)

    # if audio_data.ndim > 1:
        # return audio_data[channel_nb, :]

    return audio_data

def get_MFCCS_change(
                     audioIn:str|npt.NDArray, 
                     sigSr:float,
                     /,*,
                     channelN:int=0, 
                     tStep:float=0.001, 
                     winLen:float=0.025, 
                     n_mfcc:int=13, 
                     n_fft:int=512, 
                     minFreq:int=100, 
                     maxFreq:int= 10000, 
                     removeFirst:int=1, 
                     filtCutoff:int=12, 
                     filtOrd:int=6, 
                     diffMethod:str='grad', 
                     outFilter:str='iir',
                     outFiltType:str='low',
                     outFiltCutOff:list|npt.NDArray=[None], 
                     outFiltLen:int=6,  
                     outFiltPolyOrd:int=3
                     ):   
    
    """ 
    
    Computes the amount of change in the MFCCs over time
    
    Input
    -------
    
    audioIn (str or np array): input audio, if a string it indicates a file path, 
            if a np array of floats it represents an audio signal
    
    sigSr (default=10000): sampling frequency for the analysis
    
    channelN (default(default=0): selet the channel number for multichannel audio files
    
    tStep (default=0.005): analysis time step in ms

    winLen (default=0.025): analysis window length in ms

    n_mfcc (default=13): number of MFCCs to compute (the first one may then be removed via reoveFirst)

    n_fft (default=512): number of points for the FFT

    minFreq (default=100): smallest spectral frequency considered  

    maxFreq (default=8000): higest spectral frequency considered

    removeFirst (default=1): if one, the first cepstral corefficient is discarded

    filtCutoff (default=12): bandpass fitler freq.in Hz

    filtOrd (default=6): bandpass filter order

    diffMethod(default='grad'): method to compute velocity either central difference (grad) or Savitsky-Golay 
        with poly order =2 and win len = 3
    
    outFilter (string or None, default= None): filter to apply after computation
            of deltaMFCC. One among None (no filter), iir (infinite response filter), 
            fir (finite inpulse respnse filter), sg (Savitsky Golay low pass filter). 
            If this is not none it replaces the low pass filter applied to the total 
            amount of MFCCs change in the original Goldstein's (2019) formulation .
    
    outFiltCutOff (list or np array with max two positive, monotonically increasing 
            arguments): cut off frequency/frequencies in Hertz. If it contains one 
            value, this will be the cut-off of a low pass filter, if two values
            these will be the cut-offs of a band pass filter. NOT USED WHEN FILT=sg
    
    outFiltLen (positive integer): length of the filter in number of samples. 
        When filter is sg (Savitsky Golay), filt len represents the length of the 
        time window
        
    outFiltPolyOrd (positive integer): order of the polinomial used by sg (Savitsky Golay) filter
 
    
    Ouput
    -------    
        totChange: Amount of change over time
        
        T: time stamps for each value   
    """
    if type(audioIn)==str: # if audioIn represents a file name open it with the desired sampling rate
        myAudio, _ = audioLoad(audioIn,sr=sigSr, mono=False)
    else:
        myAudio=audioIn
        
    if len(np.shape(myAudio))>1:# exstract desired channel if signal is multichannel
        y=myAudio[channelN,:]
    else:
        y=myAudio
    
    win_length=int(winLen*sigSr)# get window length in frame numbers
    
    hop_length=int(tStep*sigSr)# get hop length in frame numbers
    
    # launch Librosa MFCC routine
    myMfccs=mfccSpectr( y=y, sr=sigSr, n_mfcc=n_mfcc, win_length=win_length, hop_length=hop_length,n_fft=n_fft,fmin=minFreq,fmax=maxFreq)
    
    # obtain time anchors
    T=np.round(np.multiply(np.arange(1,np.shape(myMfccs)[1]+1),tStep)+winLen/2,4)
    
    # remove first component (it's amplitude)
    if removeFirst:
       
        myMfccs=myMfccs[1:,:]
     
    # lop-pass filter MFCCs
    cutOffNorm = filtCutoff / ((1/tStep) / 2)
    
    sos = butter(filtOrd, cutOffNorm, btype='low', output='sos')
    
    filtMffcs=sosfiltfilt(sos, myMfccs)
        
    # compute derivative
    if diffMethod=='grad':# if use gradient
    
        myDiff=np.gradient(filtMffcs,axis=1)
    
    else:# if use Savitsky Golay differentiator
    
        myDiff = savgol_filter(
            filtMffcs, 3, 2, deriv=1, axis=1, mode='interp')
        
    #Get square root od summed squared differences
    totChange=np.sqrt(np.sum(myDiff**2,0))/np.shape(myMfccs)[0]
    
    if outFilter is None: # if no post processing filter is applied, 
                          # apply Goldstein's low pass filter
        
        #low pass filter total amount of change
        totChange=sosfiltfilt(sos,totChange)
        
    else: # otherwise use custom low or band-pass filter (see DOC of function apply filter)
        
        totChange=applyFilter(totChange,1/tStep,filt=outFilter,filtType=outFiltType,cutOff=outFiltCutOff, filtLen=outFiltLen, polyOrd=outFiltPolyOrd)
    
    return totChange, T
