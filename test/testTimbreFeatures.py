'''
Created on Aug 27, 2016

test compute harmonic spectrum and magnitudes
visualize  
@author: joro
'''
from src.timbreFeatures import compute_harmonic_magnitudes,\
    compute_harmonic_magnitudes
import json
from src.HarmonicSummationSF import calculateSpectrum
import numpy as np
import sys
from src import parsing
import os
from matplotlib import pyplot
from src.vocalVariance import extractMFCCs, extractVocalVar
from src.contour_classification.experiment_utils import get_data_files
from src.contour_classification.contour_utils import contours_from_contour_data,\
    plot_contours, plot_contours_interactive
from src.contour_classification.run_contour_training_melody_extraction import contours_path,\
    load_labeled_contours
OLAP_THRESH = 0.5

test_track = '10161_chorus'

def test_compute_harmonic_ampl_2(args):
    '''
    test computing of harmonic amplitudes with essentia
    load all contours saved as pandas dataframes. If they have already the timbre features, they are recomputed here.  
    '''
    
    args, options = parsing.parseOptions(args)
    
    
    _, fftgram = calculateSpectrum(test_track+ '.wav', options.hopsizeInSamples)
    timestamps_recording = np.arange(len(fftgram)) * float(options.hopsizeInSamples) / options.Fs
        
    contour_data_frame, adat = get_data_files(test_track, meltype=1)
    c_times, c_freqs, _ = contours_from_contour_data(contour_data_frame)
    
    for (times, freqs) in zip(c_times.iterrows(), c_freqs.iterrows()): # for each contour
        row_idx = times[0]
        times = times[1].values
        freqs = freqs[1].values

        # remove trailing NaNs
        times = times[~np.isnan(times)]
        freqs = freqs[~np.isnan(freqs)]
        
        #plot contours

        # compute harm magns
        _, spectogram_contour, hfreqs, magns =  compute_harmonic_magnitudes(freqs, times[0], fftgram, timestamps_recording, options )
        

        
#         save_harmonics(times, hfreqs, test_track)
        # plot spectrogram per contour
        pyplot.imshow(spectogram_contour)
        pyplot.show()
        
        
        mfccs_array = extractMFCCs(spectogram_contour)
        vv_array = extractVocalVar(mfccs_array, 2048, 5, options)



    
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
    