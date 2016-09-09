'''
Created on Aug 27, 2016

test compute harmonic spectrum and magnitudes
visualize  
@author: joro
'''

import sys
import os
from src.timbreFeatures import get_ts_contour, extract_vocal_var
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
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

OLAP_THRESH = 0.5




def test_compute_harmonic_ampl_2(args):
    '''
    test computing of harmonic amplitudes with essentia
    load the salience bins, saved as pandas dataframes. extract complex fft spectrum and compute harmonic amplitudes  
    If they have already the timbre features, they are recomputed here.  
    '''
    
    args, options = parsing.parseOptions(args)
    to_mfcc_coeff = 5
    
    _, fftgram = calculateSpectrum(Parameters.test_track+ '.wav', options.hopsizeInSamples)
    timestamps_recording = np.arange(len(fftgram)) * float(options.hopsizeInSamples) / options.Fs
        
    contour_data_frame, adat = get_data_files(Parameters.test_track, meltype=1)
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
        vv_array = extract_vocal_var(fftgram, idx_start, freqs, to_mfcc_coeff,    options)                
        
#         save_harmonics(times, hfreqs, test_track)
        # plot spectrogram per contour
        pyplot.imshow(vv_array)
        pyplot.show()
        
        



    
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
    
    
    test_compute_harmonic_ampl_2(sys.argv)
    
#     test_vocal_variance(sys.argv)
    