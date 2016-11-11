'''
Created on Aug 27, 2016

test compute harmonic spectrum and magnitudes
visualize  
@author: joro
'''

import sys
import os
import essentia.standard
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from timbreFeatures import get_ts_contour, extract_vocal_var,\
    compute_timbre_features, load_timbre_features
from main_contour_extraction import load_contour
from timbreFeatures import compute_harmonic_magnitudes, compute_harmonic_magnitudes
import json
from HarmonicSummationSF import calculateSpectrum
import numpy as np
import parsing
import os
from matplotlib import pyplot
from vocalVariance import extractMFCCs, compute_var_mfccs
from contour_classification.experiment_utils import get_data_files
from contour_classification.contour_utils import contours_from_contour_data,\
    plot_contours, plot_contours_interactive
from Parameters import Parameters


def test_vocal_variance(track, options):
    '''
    test computing of harmonic amplitudes with essentia
    load the salience bins, saved as pandas dataframes. extract complex fft spectrum and compute harmonic amplitudes  
    If they have already the timbre features, they are recomputed here.  
    '''
    
    
        
    _, fftgram = calculateSpectrum(track + '.wav', options.hopsizeInSamples)
    timestamps_recording = np.arange(len(fftgram)) * float(options.hopsizeInSamples) / options.Fs
        
    contour_data_frame, adat = get_data_files(track, meltype=1)
    c_times, c_freqs, _ = contours_from_contour_data(contour_data_frame)
    
    for (times, freqs) in zip(c_times.iterrows(), c_freqs.iterrows()): # for each contour
        row_idx = times[0]
        times = times[1].values
        freqs = freqs[1].values

        # remove trailing NaNs
        times = times[~np.isnan(times)]
        freqs = freqs[~np.isnan(freqs)]
        

        # compute harm magns

        times_contour, idx_start = get_ts_contour(freqs, times[0], timestamps_recording, options)
        print 'contour len: {}'.format(times[-1] - times[0])
        vv_array = extract_vocal_var(fftgram, idx_start, freqs, Parameters.dim_timbre,    options)                
        
#         save_harmonics(times, hfreqs, test_track)
        # plot spectrogram per contour
#         pyplot.imshow(vv_array)
#         pyplot.show()
        return contour_data_frame, vv_array






def compare_to_matlab_features(track, options):
    '''
    test computing of harmonic amplitudes with essentia
    load the salience bins, saved as pandas dataframes. extract complex fft spectrum and compute harmonic amplitudes  
    If they have already the timbre features, they are recomputed here.  
    '''
    
    contour_data_frame, adat = get_data_files(track, meltype=1)
    contour_output_path = '/home/georgid/Documents/iKala/Conv_mu-1_G-0_LHSF-0_pC-27.56_pDTh-0.9_pFTh-0.9_tC-100_mD-200_vxTol-0.2_LEH_300_100_300_5_1_5/'
    
    
    
    
    wav_URI = os.path.join(os.path.dirname(__file__), track + '.wav')
    print wav_URI
    _, fftgram = calculateSpectrum(wav_URI , options.hopsizeInSamples)
    timestamps_recording = np.arange(len(fftgram)) * float(options.hopsizeInSamples) / options.Fs
        
    c_times, c_freqs, _ = contours_from_contour_data(contour_data_frame)
    

    
    for i, (times, freqs) in zip(c_times.iterrows(), c_freqs.iterrows()): # for each contour
        row_idx = times[0]
        times = times[1].values
        freqs = freqs[1].values

        # remove trailing NaNs
        times = times[~np.isnan(times)]
        freqs = freqs[~np.isnan(freqs)]
        

        # compute harm magns

        times_contour, idx_start = get_ts_contour(freqs, times[0], timestamps_recording, options)
        print 'contour len: {}'.format(times[-1] - times[0])
        
        vv_array, audio = extract_vocal_var(fftgram, idx_start, freqs, Parameters.dim_vv,    options)  
#         contourTimbre = compute_timbre_features(contours_bins_SAL, contours_start_times_SAL, fftgram, timestamps_recording, options)


        # plot spectrogram per contour
        pyplot.imshow(vv_array, interpolation='none')
        pyplot.show()
        

        ########################### matlab 
        
        # save  audio as file. 
        # extract  feature with matlab
        # visualize in matlab.      
        print 'len : {}'.format(len(audio)/44100.0)
        


    
def save_harmonics(times_contour, hfreqs, outFile_name):
        '''
        save harmonics para visualisation in SV
        '''
        ##### plot first 5 harmonics
        len_contour = len(times_contour, hfreqs)
        output_path = '.'
        for i in range(5):
                harm_series = hfreqs[:,i]
                if len_contour != len(harm_series):
                    sys.exit('not equal size harm series and pitch')
                est_partial_and_ts  = zip(times_contour, harm_series)
                outFileURI = os.path.join(output_path , outFile_name + '._' +  str(i) + '_pitch_onlyVocal.csv')
                writeCsv(outFileURI, est_partial_and_ts) 
    
        # TODO: open in sonic visualiser

     
  



def writeCsv(fileURI, list_, withListOfRows=1):
    '''
    TODO: move to utilsLyrics
    '''
    from csv import writer
    fout = open(fileURI, 'wb')
    w = writer(fout)
    print 'writing to csv file {}...'.format(fileURI)
    for row in list_:
        if withListOfRows:
            w.writerow(row)
        else:
            tuple_note = [row.onsetTime, row.noteDuration]
            w.writerow(tuple_note)
    
    fout.close()


if __name__ == '__main__':
    
    track = Parameters.test_track
    args, options = parsing.parseOptions(sys.argv)
    
    test_vocal_variance(track, options)
    
    

    
    