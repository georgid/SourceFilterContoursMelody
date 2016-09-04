'''
Created on Aug 26, 2016

@author: joro
'''

import numpy as np
from src.HarmonicSummationSF import calculateSpectrum
from essentia.standard import HarmonicModelAnal
from essentia.standard import *
import essentia.streaming as es
import sys
from essentia import Pool
import os
import numpy
from src.vocalVariance import extractMFCCs, extractVocalVar
import traceback

def compute_timbre_features(contours_bins_SAL,contours_start_times_SAL, fftgram, times, options):
    
    NContours = len(contours_bins_SAL)
    NtimbreFeat = 5
    
    contourTimbre =  np.zeros([NContours, NtimbreFeat])
    a = options.stepNotes / 1200.0
    for i, curr_contour in enumerate(contours_bins_SAL):
            contour_f0 = options.minF0 * np.power(2, np.array(curr_contour) * a)
            contours_bins_SAL[i] = contour_f0
    
    for i in range(NContours):
            lcontour = len(contours_bins_SAL[i])
            if lcontour > 0:
                print 'working on contour {}...'.format(i)
                times_contour, spectogram_harm,  hfreq, magns = compute_harmonic_magnitudes(contours_bins_SAL[i], contours_start_times_SAL[i], fftgram, times, options )
                mfccs_array = extractMFCCs(spectogram_harm)
                vv_array = extractVocalVar(mfccs_array, 2048, NtimbreFeat, options)
                
                # take median over features
                median_timbre_features = numpy.median(vv_array, axis = 0)
                
                contourTimbre[i,:] = median_timbre_features.T
    
    if (options.plotting):
        import pylab as plt
        plt.imshow(contourTimbre)

    return contourTimbre



def compute_harmonic_magnitudes(contour_f0s, contour_start_time, fftgram, times, options ):
    '''
    Compute for each frame harm amplitude
    convert cent bins to herz
    get harmonic partials form original spectrum
    '''
    
    run_harm_model_anal = HarmonicModelAnal(nHarmonics=30)
    
    # TODO: sanity check: times == len(fftgram) and contour_start_time_SAL in times
   
   
    #### at which timestamp starts contour from whole audio? 
    #      there could be some inprecision in the timestamp
    time_interval_min = float(options.hopsizeInSamples) / options.Fs /2.0
    idx_start_where = np.where(abs( times - contour_start_time) < time_interval_min )
    if len(idx_start_where[0]) != 1:
        sys.exit('there should be one timestamp with this pitch')
    idx_start = idx_start_where[0][0]
    
    pool = Pool()
    
    for i, contour_f0 in enumerate(contour_f0s):
        
        fft = fftgram[idx_start + i]
        # convert to freq : 
        hfreq, magns, phases = run_harm_model_anal(fft, contour_f0)
        spectrum = harmonics_to_spectrum(hfreq, magns, phases, options)
        pool.add('spectrum', spectrum)
        pool.add('hfreq', hfreq)
        pool.add('magns', magns)
       
    len_contour = len(contour_f0s)
    times_contour =   contour_start_time +  numpy.arange(len_contour) *  float(options.hopsizeInSamples) / options.Fs
    return times_contour, pool['spectrum'],  pool['hfreq'], pool['magns']

def compute_harm_variation(hfreq, magns):
    '''
    find frequencies of first 4 formants
    variance of frequency of these formants
    human voice     
    '''

def compute_amplitude_variation(hfreq, magns):
    '''
    
    variance of amplitudes harmonics
     
    '''

def harmonics_to_spectrum(hfreq, magns, phases, options):
    '''
    convert to spectrum
    see tutotrial
    https://github.com/MTG/essentia/blob/2bc1deba4d49ed8e025b4c2b45d0d00c0ca2ec49/src/examples/python/musicbricks-tutorials/2-sinemodel_analsynth.py
    '''
    run_sine_model_synth = SineModelSynth( hopSize=options.hopsizeInSamples, sampleRate = options.Fs) 
    fft = run_sine_model_synth(   magns, hfreq, phases)
    spectrum = abs(fft)
    
    return spectrum



    
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


if __name__ == '__main__':
    pass