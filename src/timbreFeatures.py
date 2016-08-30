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
from docutils.nodes import math
import numpy
from src.vocalVariance import extractMFCCs, extractVocalVar

def compute_timbre_features(contours_bins_SAL,contours_start_times_SAL, fftgram, times, options):
    
    NContours = len(contours_bins_SAL)
    NtimbreFeat = 5
    
    contourTimbre =  np.zeros([NContours, NtimbreFeat])
    
    
    for i in range(NContours):
            lcontour = len(contours_bins_SAL[i])
            if lcontour > 0:
                times_contour, spectogram_harm,  hfreq, magns = compute_harmonic_magnitudes(contours_bins_SAL[i], contours_start_times_SAL[i], fftgram, times, options )
                mfccs_array = extractMFCCs(spectogram_harm)
                vv_array = extractVocalVar(mfccs_array, 2048, options)
                
                # take median over features
                median_timbre_features = numpy.median(vv_array, axis = 0)
                
                contourTimbre[i,:] = median_timbre_features
    
    if (options.plotting):
        import pylab as plt
        plt.imshow(contourTimbre)

    return contourTimbre


def compute_harmonic_magnitudes(contour_bins_SAL, contour_start_time_SAL, fftgram, times, options ):
    '''
    Compute for each frame harm amplitude
    convert cent bins to herz
    get harmonic partials form original spectrum
    '''
    
    run_harm_model_anal = HarmonicModelAnal()
    
    # TODO: sanity check: times == len(fftgram) and contour_start_time_SAL in times
    len_contour = len(contour_bins_SAL)
   
    #### at which timestamp starts contour from whole audio? 
    #      there could be some inprecision in the timestamp
    time_interval_min = float(options.hopsizeInSamples) / options.Fs /2.0
    idx_start_where = np.where(abs( times - contour_start_time_SAL) < time_interval_min )
    if len(idx_start_where[0]) != 1:
        sys.exit('there should be one timestamp with this pitch')
    idx_start = idx_start_where[0][0]
    
    pool = Pool()
    
    for i, idx in enumerate(range(idx_start, idx_start + len_contour)):
        
        fft = fftgram[idx]
        # convert to freq : 
        f0 = options.minF0 * pow(2, contour_bins_SAL[i] * options.stepNotes / 1200.0)
        hfreq, magns, phases = run_harm_model_anal(fft, f0)
        spectrum = harmonics_to_spectrum(hfreq, magns, phases, options)
        pool.add('spectrum', spectrum)
        pool.add('hfreq', hfreq)
        pool.add('magns', magns)
       
    
    times_contour =   contour_start_time_SAL +  numpy.arange(len_contour) *  float(options.hopsizeInSamples) / options.Fs
    return times_contour, pool['spectrum'],  pool['hfreq'], pool['magns']


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

def loadContour():
    import contour_classification.contour_utils as cc
    contour_fpath = 'recording.ctr'
    cdat = cc.load_contour_data(contour_fpath, normalize=False)
    print cdat

if __name__ == '__main__':
    pass