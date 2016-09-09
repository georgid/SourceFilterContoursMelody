'''
Created on Aug 26, 2016

@author: joro
'''

import numpy as np
from HarmonicSummationSF import calculateSpectrum
# from essentia.standard import HarmonicModelAnal
from essentia.standard import *
import essentia.streaming as es
import sys
from essentia import Pool
import os
import numpy
from vocalVariance import extractMFCCs, compute_var_mfccs
import traceback
from matplotlib import pyplot








def extract_vocal_var(fftgram, idx_start, contour_f0s,  NtimbreFeat,    options):
    '''
    extract vocal variance from contours_f0 and fftgram for whole audio
    1. extract harmonic partials
    2. resynthesize audio (needed because mfcc are extracted from higher spectral resolution)
    3. extract mfccs-  
    '''
    
    hfreqs, magns, phases = compute_harmonic_magnitudes(contour_f0s, fftgram, idx_start, options)
    audio_contour, spectogram_contour = harmonic_magnitudes_to_audio(hfreqs, magns, phases, options)
#     awrite = MonoWriter (filename = 'test.', sampleRate = 44100);
    
    mfccs_array = extractMFCCs(audio_contour)
    vv_array = compute_var_mfccs(mfccs_array,  NtimbreFeat, options)

    return vv_array


def compute_timbre_features(contours_bins_SAL, contours_start_times_SAL, fftgram, times_recording, options):
    
    NContours = len(contours_bins_SAL)
    NtimbreFeat = 5
    
    contours_f0 = []
    a = options.stepNotes / 1200.0
    for  curr_contour in contours_bins_SAL: # convert from cent bins to f0
            contour_f0 = options.minF0 * np.power(2, np.array(curr_contour) * a)
            contours_f0.append(contour_f0)
    
    contourTimbre =  np.zeros([NContours, NtimbreFeat]) # compute timbral features
    for i in range(NContours):
            lcontour = len(contours_f0[i])
            if lcontour > 0:
#                 print 'working on contour {}...'.format(i)
                
                times_contour, idx_start = get_ts_contour(contours_f0[i], contours_start_times_SAL[i], times_recording, options)
                
                vv_array = extract_vocal_var(fftgram, idx_start, contours_f0[i],  NtimbreFeat,   options)
                
                # take median over features
                median_timbre_features = numpy.median(vv_array, axis = 0)
                
                contourTimbre[i,:] = median_timbre_features.T
    
    if (options.plotting):
        import pylab as plt
        plt.imshow(contourTimbre)

    return contourTimbre



def compute_harmonic_magnitudes(contour_f0s,  fftgram, idx_start, options):
    '''
    Compute for each frame harm amplitude
    get harmonic partials form original spectrum
    
    Params:
    fftgram - fftgram of whole audio file
    times - ts of whole audio
    
 
    hfreq - harmonics  of contour
    magns -  magns of contour
    '''
    
    run_harm_model_anal = HarmonicModelAnal(nHarmonics=30)
    
    # TODO: sanity check: times == len(fftgram) and contour_start_time_SAL in times
   
   
    pool = Pool()
    

    for i, contour_f0 in enumerate(contour_f0s):
        
        fft = fftgram[idx_start + i]
        # convert to freq : 
        hfreq, magn, phase = run_harm_model_anal(fft, contour_f0)
       
        
        pool.add('phases', phase)
        pool.add('hfreqs', hfreq)
        pool.add('magns', magn)
       

    return pool['hfreqs'], pool['magns'], pool['phases']


def get_ts_contour(contour_f0s, contour_start_time, times, options):
    '''
    Params:
    fftgram - fftgram of whole audio file
    times - ts of whole audio
    
       return:
    ts of contour
    '''
    
      #### at which timestamp starts contour from whole audio? 
    #      there could be some inprecision in the timestamp
    time_interval_min = float(options.hopsizeInSamples) / options.Fs /2.0
    idx_start_where = np.where(abs( times - contour_start_time) < time_interval_min )
    if len(idx_start_where[0]) != 1:
        sys.exit('there should be one timestamp with this pitch')
    idx_start = idx_start_where[0][0]
    
    len_contour = len(contour_f0s)
    times_contour =   contour_start_time +  numpy.arange(len_contour) *  float(options.hopsizeInSamples) / options.Fs
    return times_contour, idx_start   

def harmonic_magnitudes_to_audio (hfreqs, magns, phases,  options):
    '''
    Compute for each frame harm amplitude
    convert cent bins to herz
    get harmonic partials form original spectrum
    
    Params:
    
    hfreq - harmonics  of contour
    magns -  magns of contour
    
    return:
    spectogram contour

    out_audio_contour - audio of harmonics for a contour
    '''
    
   
    
    pool = Pool()
    
    run_sine_model_synth = SineModelSynth( hopSize=options.hopsizeInSamples, sampleRate = options.Fs) 
    run_ifft = IFFT(size = options.windowsizeInSamples);
    run_overl = OverlapAdd (frameSize = options.windowsizeInSamples, hopSize = options.hopsizeInSamples);
    out_audio_contour = np.array(0)
    
    for hfreq, magn, phase in zip(hfreqs, magns, phases):
        
        spectrum, audio_frame = harmonics_to_audio(hfreq, magn, phase, run_sine_model_synth, run_ifft, run_overl )
        out_audio_contour = np.append(out_audio_contour, audio_frame)
        
        pool.add('spectrum', spectrum)

       
    return  out_audio_contour, pool['spectrum']


def compute_harm_variation(hfreq, magns):
    '''
    find frequencies of first 4 formants
    variance of frequency of these formants
    human voice     
    '''
    pass

def compute_amplitude_variation(hfreq, magns):
    '''
    
    variance of amplitudes harmonics
     
    '''
    varianceLength = 1.0 # 1 second
    
    mfccs_array = mfccs_array[:, 1:num_mfccs_var+1]
    
    num_mfccs= mfccs_array.shape[1]       
    num_frames = len(magns)
    
   
    # variance num frames
    numFrVar = int(math.floor(options.Fs * varianceLength / _frameSize))
    vocal_var_array = np.zeros(mfccs_array.shape)
    
    for i in range(0, num_frames):
        startIdx = max(0,i-numFrVar)
        endIdx = min( num_frames-1, i + numFrVar )
    
        # iterate over mfccs
        for coeff in range(num_mfccs):
            mfcc_slice = mfccs_array [ startIdx : endIdx + 1, coeff ]
            vocal_var_array[i, coeff] = np.var( mfcc_slice )

def harmonics_to_audio(hfreq, magns, phases, run_sine_model_synth,  run_ifft, run_overl):
    '''
    
    convert to spectrum for a whole contour
    see tutotrial
    https://github.com/MTG/essentia/blob/2bc1deba4d49ed8e025b4c2b45d0d00c0ca2ec49/src/examples/python/musicbricks-tutorials/2-sinemodel_analsynth.py
    '''
    
    fft_harm = run_sine_model_synth(   magns, hfreq, phases)
    
    # go back to audio
    audio_out = run_overl(run_ifft(fft_harm))
    
    spectrum = abs(fft_harm)
    
    return spectrum, audio_out



    
def createDataFrameWithExtraFeatures(contours_start_times_SAL,contour_bins,timbreFeatures,contourTonalInfo):
    """ Create DataFrame with additional features (not salience, pitch or vibrato)
    Parameters
    ----------
    contour_bins,timbreFeatures [optional] matrix Ncontours*Nfeatures

    Returns
    -------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    """
    from pandas import DataFrame,concat
    headers = []
    extraFeatures = None
    if timbreFeatures is not None:
        NFeatures = timbreFeatures.shape[1]
        headers = ['timbre'+str(id) for id in range(0,NFeatures)]
        extraFeatures = concat([extraFeatures,DataFrame(timbreFeatures, columns=headers)],axis=1)
    if contourTonalInfo is not None:
        NFeatures = contourTonalInfo.shape[1]
        headers = ['tonal'+str(id) for id in range(0,NFeatures)]
        extraFeatures = concat([extraFeatures,DataFrame(contourTonalInfo,columns=headers)],axis=1)

    return extraFeatures
