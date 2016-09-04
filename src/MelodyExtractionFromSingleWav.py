#!/usr/bin/env python
from src.HarmonicSummationSF import calculateSpectrum
__author__ = "Juan Jose Bosch"
__email__ = "juan.bosch@upf.edu"

import sys

from numpy import savetxt, max, column_stack

import SourceFilterModelSF
import combineSaliences
import melodyExtractionFromSalienceFunction
from HarmonicSummationSF import calculateSF


def process(args):
    # options
    mu = 1
    G = 0
    doConvolution = True
    fileName = args[0]
    combmode = 13

    # Compute HF0 (SIMM with source-filter model)
    if combmode > 0:
        timesHF0, HF0, options = SourceFilterModelSF.main(args)

    # In order to have the same structure as the Harmonic Summation Salience Function
    HF0 = HF0[1:, :]


    # -------------------------

    if options.extractionMethod == "BG1":
        # Options MIREX 2016:  BG1
        options.pitchContinuity = 27.56
        options.peakDistributionThreshold = 1.3
        options.peakFrameThreshold = 0.7
        options.timeContinuity = 100
        options.minDuration = 100
        options.voicingTolerance = 1
        options.useVibrato = False

    if options.extractionMethod == "BG2":
        # Options MIREX 2016:  BG2
        options.pitchContinuity = 27.56
        options.peakDistributionThreshold = 0.9
        options.peakFrameThreshold = 0.9
        options.timeContinuity = 100
        options.minDuration = 100
        options.voicingTolerance = 0.2
        options.useVibrato = False

    if combmode != 4 and combmode != 5:
        # Computing Harmonic Summation salience function
        hopSizeinSamplesHSSF = int(min(options.hopsizeInSamples, 0.01 * options.Fs))
        spectogram, fftgram = calculateSpectrum(fileName, hopSizeinSamplesHSSF)
        timesHSSF, HSSF = calculateSF(spectogram,  hopSizeinSamplesHSSF)
    else:
        print "Harmonic Summation Salience function not used"

    # Combination mode used in MIREX, ISMIR2016, SMC2016
    if combmode == 13:
        times, combSal = combineSaliences.combine3MIREX(timesHF0, HF0, timesHSSF, HSSF, G, mu, doConvolution)

    combSal = combSal / max(combSal)

    print("Extracting melody from salience function")
    times, pitch, dummy, dummy = melodyExtractionFromSalienceFunction.MEFromSF(times, combSal, options)

    # Save output file
    savetxt(options.pitch_output_file, column_stack((times.T, pitch.T)), fmt='%-7.5f', delimiter="\t")
    print("Output file written")


def main(args):
    process(args)


if __name__ == '__main__':
    import time

    start_time = time.time()

    main(sys.argv[1:])
    print("Processing time: --- %s seconds ---" % (time.time() - start_time))
