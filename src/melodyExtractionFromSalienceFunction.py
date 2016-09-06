from timbreFeatures import compute_timbre_features,\
    createDataFrameWithExtraFeatures
from HarmonicSummationSF import calculateSpectrum, calculateSF
import json
from Parameters import Parameters
__author__ = 'juanjobosch'

import sys, os
from essentia import *
from essentia.standard import *
import contourExtraction as ce

   


    
def MEFromFileNumInFolder(salsfolder, outfolder, fileNum, options):
    """ Auxiliar function, to extract melody from a folder with precomputed and saved saliences (*.Msal)
        Parameters
    ----------
    salsfolder: Folder containing saved saliences
    outfolder: melody extraction output folder
    fileNum: number of the file [1:numfiles]
    options: set of options for melody extraction

    No return
    """
    from os.path import join, basename
    import glob

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    fn = glob.glob(salsfolder + '*.Msal*')[fileNum - 1]
    bn = basename(fn)
    outputfile = join(outfolder, bn[0:bn.find('.Msal')] + '.pitch')
    wavFile = join(salsfolder, bn[0:bn.find('.Msal')] + '.wav')
    MEFromSFFile(fn, outputfile, wavFile, options)


def loadSFFile(fn):
    """ Auxiliar function to load a previouslly saved salience function (*.Msal)
        Parameters
    ----------
    fn: filename

        Returns
    ----------
    times: set of times for the frames of the salience function
    SF: Pitch salience function
    """
    from os.path import splitext
    from numpy import loadtxt
    from scipy.io import loadmat

    if splitext(fn)[-1] == '.mat':
        loaded = loadmat(fn)
        mat = loaded.get('timesAndSF')
    else:
        try:
            mat = loadtxt(fn)
        except:
            mat = loadtxt(fn, delimiter=',')
            # load as text file

    times = mat[:, 0]
    SF = mat.T

    return times, SF


def MEFromSFFile(fn, outputfile, wavFile, options):
    """ Computes Melody extractino from a Salience function File
        Parameters
    ----------
    fn: salience function filename
    outputfile: output filename
    options: set of options for melody extraction

    No returns

    """
    from numpy import column_stack, savetxt

    times, SF = loadSFFile(fn)
    times, pitch = MEFromSF(times, SF, wavFile, options)
    savetxt(outputfile, column_stack((times.T, pitch.T)), fmt='%-7.5f', delimiter=",")


def MEFromSF(times, SF, fftgram, options):
    """ Computes Melody extractino from a Salience function
        Parameters
    ----------
    times: set of times for each frame of the salience function
    SF: Pitch salience function
    options: set of options for melody extraction
    E.g.
    options.saveContours = True : to save contours as a dataframe for contour classification
    options.PCS = True : to run melody extraction based on Pitch Contour Selection (MIREX2015, MIREX2016, SMC2016, ISMIR2016(C2) )

    Returns:
    ----------
    times: set of times for each frame of the estimated melody
    pitch: set of pitches of the estimated melody
    """

    Fs = options.Fs
    hopsize = options.hopsizeInSamples
    stepNotes = options.stepNotes
    
    Nbins = SF.shape[0]

    try:
        voiceVibrato = options.voiceVibrato
    except:
        # Default: use of vibrato = False
        voiceVibrato = False

    voicingTolerance = options.voicingTolerance

    import numpy as np

    # Initialise methods:

    # Initialise Pitch contour selection: from contours, extracting melody using salamon2012 as mplemented in Essentia

    run_pitch_contours_melody = PitchContoursMelody(guessUnvoiced=True,
                                                    binResolution=int(stepNotes),
                                                    hopSize=int(hopsize), voicingTolerance=int(voicingTolerance),
                                                    voiceVibrato=voiceVibrato,
                                                    referenceFrequency=options.minF0,
                                                    minFrequency=options.minF0)

    # Computes peaks from salience function

    run_pitch_salience_function_peaks = PitchSalienceFunctionPeaks(binResolution=int(stepNotes),
                                                                   referenceFrequency=options.minF0,
                                                                   minFrequency=options.minF0)

    # Extracts contours from salience function peaks

    run_pitch_contours = PitchContours(hopSize=int(hopsize), binResolution=int(stepNotes),
                                       peakDistributionThreshold=options.peakDistributionThreshold,
                                       peakFrameThreshold=options.peakFrameThreshold,
                                       minDuration=options.minDuration,
                                       timeContinuity=options.timeContinuity,
                                       pitchContinuity=options.pitchContinuity)

    pool = Pool()

    # For all frames, compute salience peaks, and save their salience and bin
    for index in range(0, np.size(times) - 1, 1):
        # The vector should be of size 600 if we have 10 bins/semitone (total 6000)
        SALsalience_peaks_bins, SALsalience_peaks_saliences = run_pitch_salience_function_peaks(
            np.array(np.append((np.array(SF[1:601, index])), np.zeros(max(0, 600 - Nbins))), 'float32'))
        if (len(SALsalience_peaks_bins) == 0):
            SALsalience_peaks_bins = [1.0]
            SALsalience_peaks_saliences = [1e-15]
        pool.add('allframes_SALsalience_peaks_saliences', SALsalience_peaks_saliences)
        pool.add('allframes_SALsalience_peaks_bins', SALsalience_peaks_bins)

    # Create contours using previouslly computed peaks
    contours_bins_SAL, contours_saliences_SAL, contours_start_times_SAL, durationSAL = run_pitch_contours(
        pool['allframes_SALsalience_peaks_bins'],
        pool['allframes_SALsalience_peaks_saliences'])

    NContours = len(contours_bins_SAL)
    print 'NContours %d' % NContours
    pitch = np.zeros(len(times))
    
#     print contours_bins_SAL[0]
#     import json
#     with open('../test/contour_bins.txt', 'w') as outfile:
#         contour_bins_test = contours_bins_SAL[0]
#         contour_bins_test.insert(0, contours_start_times_SAL[0])
#         json.dump(contour_bins_test, outfile)
    contourTimbre = None 
    if Parameters.useTimbre:
        try:
             contourTimbre = compute_timbre_features(contours_bins_SAL, contours_start_times_SAL, fftgram, times, options)
        except:
            print "Error computing timbre features"

    
    options.saveContours = True

    if (NContours > 0):

        if options.extractionMethod == 'PCS':
            # Extract melody from contours using Pitch Contour Selection
            allpitch, confidence = run_pitch_contours_melody(contours_bins_SAL,
                                                             contours_saliences_SAL,
                                                             contours_start_times_SAL,
                                                             durationSAL)

            # We convert the allpitch (always positive) to a sequence of positive
            # and negative pitches, depending on the confidence, which is a measure
            # of the voicing. We add 0 to avoid negative zeros (-0.0)
            pitch = allpitch * (-1 + 2 * (confidence > 0)) + 0
            L = min(len(pitch), len(times))
            pitch = pitch[0:L]
            times = times[0:L]
            
        # If contour need to be saved for pitch contour classification, we compute the the contour data
        if options.saveContours:
            
            try:
                extraFeatures = createDataFrameWithExtraFeatures(contours_start_times_SAL, contours_bins_SAL, contourTimbre, contourTonalInfo= None)
                contour_data = ce.compute_contour_data(contours_bins_SAL, contours_saliences_SAL,
                                                       contours_start_times_SAL, stepNotes, options.minF0,
                                                       options.hopsize, extra_features=extraFeatures)
                picklefile = options.pitch_output_file + Parameters.CONTOUR_EXTENSION
                from pickle import dump
                with open(picklefile, 'wb') as handle:
                    dump(contour_data, handle)
                print 'stored contours as ' + picklefile 
            except:
                print "Error computing contour data"
    return times, pitch



    