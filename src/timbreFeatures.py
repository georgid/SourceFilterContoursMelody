'''
Created on Aug 26, 2016

@author: joro
'''

import numpy as np
from HarmonicSummationSF import calculateSpectrum
from Parameters import Parameters
from fluctogram import extact_pseudo_fluctogram
if Parameters.extract_timbre:
    from essentia.standard import HarmonicModelAnal
from essentia.standard import *
import essentia.streaming as es
import sys
from essentia import Pool
import os
import numpy
from vocalVariance import extractMFCCs, compute_var_mfccs
import traceback
if Parameters.with_MATPLOTLIB:
    from matplotlib import pyplot
import csv

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../smstools/software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../smstools/workspace/'))

try:
    from harmonicModel_function import resynthesize
except:
    print 'sms.tools.harmonicModel not available'







def extract_vocal_var(fftgram, idx_start, contour_f0s,  NtimbreFeat,    options):
    '''
    extract vocal variance from contours_f0 and fftgram for whole audio
    1. extract harmonic partials
    2. resynthesize audio (needed because mfcc are extracted from higher spectral resolution)
    3. extract mfccs-  
    '''
    
    hfreqs, magns, phases = compute_harmonic_magnitudes(contour_f0s, fftgram, idx_start, options)
    audio_contour, spectogram_contour = harmonic_magnitudes_to_audio(hfreqs, magns, phases, options)
    
    mfccs_array = extractMFCCs(audio_contour)
    vv_array = compute_var_mfccs(mfccs_array,  NtimbreFeat, options)

    return vv_array, audio_contour



def contour_to_audio(contours_bins_SAL, contours_start_times_SAL, fftgram, times_recording, options):
    ''' 
    resynthesize contours to audio
    for listening to or extrawcting features externally
    use harmonic modeling
    '''
    NContours = len(contours_bins_SAL)
    spectogram_contours = []
    ### convert cent bins to herz
    contours_f0 = []
    a = options.stepNotes / 1200.0
    for  curr_contour in contours_bins_SAL: # convert from cent bins to f0
            contour_f0 = options.minF0 * np.power(2, np.array(curr_contour) * a)
            contours_f0.append(contour_f0)
            
            
    for i in range(NContours):
            times_contour, idx_start = get_ts_contour(contours_f0[i], contours_start_times_SAL[i], times_recording, options)

            contour_URI = options.contours_output_path + options.track + '_' + str(i)
            hfreqs, magns, phases = compute_harmonic_magnitudes(contours_f0[i], fftgram, idx_start, options)
            audio_contour, spectogram_contour = harmonic_magnitudes_to_audio(hfreqs, magns, phases, options)
            spectogram_contours.append(spectogram_contour)
            
            if not os.path.isfile(contour_URI  + '.wav'):
                resynthesize(hfreqs, magns, phases, 44100, 128, contour_URI  + '.wav')
    return  spectogram_contours 
            

def compute_timbre_features(contours_bins_SAL, contours_start_times_SAL, fftgram, times_recording, options):
    '''
    compute timbre features for all contours
    
    return: numpy array of timbral features 
    '''
    NContours = len(contours_bins_SAL)
    
    contours_f0 = []
    a = options.stepNotes / 1200.0
    for  curr_contour in contours_bins_SAL: # convert from cent bins to f0
            contour_f0 = options.minF0 * np.power(2, np.array(curr_contour) * a)
            contours_f0.append(contour_f0)
    
    contourTimbre =  np.zeros([NContours, Parameters.dim_timbre]) # compute timbral features
    for i in range(NContours):
            lcontour = len(contours_f0[i])
            if lcontour > 0:
#                 print 'working on contour {}...'.format(i)
                
                times_contour, idx_start = get_ts_contour(contours_f0[i], contours_start_times_SAL[i], times_recording, options)
           
                
#                 if Parameters.useVV_for_classification:
#                     timbre_feature, audio = extract_vocal_var(fftgram, idx_start, contours_f0[i],  Parameters.dim_timbre,   options)
                if Parameters.use_fluct_for_classification:
                    timbre_feature = extact_pseudo_fluctogram(contours_bins_SAL[i], options)
                
                # take median over features
                
                median_timbre_features = numpy.median(timbre_feature, axis = 0)
                contourTimbre[i,:] = median_timbre_features
    
#     if (options.plotting):
#         import pylab as plt
#         plt.imshow(contourTimbre)

    return contourTimbre

def load_timbre_features(contour_data_frame, options):
    '''
    load timbre features externally computed in MATLAB
    '''
    NContours = contour_data_frame.shape[0]
    contourTimbre =  np.zeros([NContours, Parameters.dim_timbre]) # compute timbral features
    for i in range(NContours):
        # load SVD-lenher extracted
        contour_URI = os.path.join(options.contours_output_path, Parameters.features_MATLAB_URI, options.track + '_' + str(i)) 
        timbre_feature = np.empty((0, Parameters.dim_timbre))
        try:    
            with open(contour_URI + '.arff') as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    curr_feature = np.array(row).astype(np.float).reshape(1,len(row))
                    if np.isnan(curr_feature).any():
                        curr_feature = np.nan_to_num(curr_feature) # workaround for NaNs -> to zero
                    
                    timbre_feature = np.append(timbre_feature, curr_feature, axis=0)
                    
        except: # if file not generated for some reason, use zeros as workaround 
            timbre_feature = np.zeros((0, Parameters.dim_timbre))
        
        median_timbre_features = numpy.median(timbre_feature, axis = 0)
        contourTimbre[i,:] = median_timbre_features
    

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
        
        if idx_start + i > len(fftgram) - 1:
                    sys.exit('idx start is {} while len ffmtgram is {}'.format(idx_start, len(fftgram) ) )
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
    
    run_sine_model_synth = SineModelSynth( hopSize=512, sampleRate = options.Fs) 
    run_ifft = IFFT(size = options.windowsizeInSamples);
    run_overl = OverlapAdd (frameSize = options.windowsizeInSamples, hopSize = 512, gain = 1./options.windowsizeInSamples );
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
