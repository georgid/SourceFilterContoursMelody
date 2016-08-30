'''
Created on Aug 27, 2016

@author: joro
'''
from src.timbreFeatures import compute_harmonic_magnitudes
import json
from src.HarmonicSummationSF import calculateSpectrum
import numpy
import sys
from src import parsing
import os
from matplotlib import pyplot
from src.vocalVariance import extractMFCCs, extractVocalVar




def test_compute_harmonic_ampl(args):
    '''
    test computing of harmonic amplitudes with essentia
    read the saliences of a contour from serizlized json file   
    '''
    
    contour_bins_SAL, contour_start_time_SAL, fftgram, times, options, wavFile = load_contours(args)
    
    times_contour, spectogram_harm, hfreqs, magns =  compute_harmonic_magnitudes(contour_bins_SAL, contour_start_time_SAL, fftgram, times, options )    
    
    ##### plot first 5 harmonics
    len_contour = len(times_contour)
    output_path = '.'
    for i in range(5):
            harm_series = hfreqs[:,i]
            if len_contour != len(harm_series):
                sys.exit('not equal size harm series and pitch')
            est_partial_and_ts  = zip(times_contour, harm_series)
            outFileURI = os.path.join(output_path , wavFile + '._' +  str(i) + '_pitch_onlyVocal.csv')
            writeCsv(outFileURI, est_partial_and_ts) 

    # TODO: open in sonic visualiser

def test_vocal_variance(args):
    
    contour_bins_SAL, contour_start_time_SAL, fftgram, times, options, wavFile = load_contours(args)
    
    times_contour, spectogram_harm, hfreqs, magns =  compute_harmonic_magnitudes(contour_bins_SAL, contour_start_time_SAL, fftgram, times, options )    

    mfccs_array = extractMFCCs(spectogram_harm)
    vv_array = extractVocalVar(mfccs_array, 2048, options)
    return vv_array


def load_contours(args):
    '''
    load from serialized array of contour salience bins by json
    '''
    
    args, options = parsing.parseOptions(args)
    
    with open('10161_chorus.contour_bins.txt', 'r') as fh:
        contour_bins_SAL = json.load(fh)
    contour_start_time_SAL = contour_bins_SAL[0]
    contour_bins_SAL = contour_bins_SAL[1:]
    wavFile = '10161_chorus.wav'
    sampleRate = 44100
    spectogram, fftgram = calculateSpectrum(wavFile, options.hopsizeInSamples)
    times = numpy.arange(len(fftgram)) * float(options.hopsizeInSamples) / sampleRate
    options.minF0 = 55
    options.stepNote = 10
    return contour_bins_SAL, contour_start_time_SAL, fftgram, times, options, wavFile



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
#     test_compute_harmonic_ampl(sys.argv)
    test_vocal_variance(sys.argv)
    