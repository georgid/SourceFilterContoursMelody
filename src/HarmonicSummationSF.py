__author__ = 'jjb'

from essentia.standard import *
from essentia import *


def calculateSpectrum(filename, hopsizeFrames):
    '''
    Computes spectrum
    Parameters
    ----------
    filename: Name of the file
    hopsizeFrames: size of the hop in frames
    '''
    
    hopSize = int(hopsizeFrames)
    frameSize = 2048

    # Setting the algorithms
    run_windowing = Windowing(type='hann', zeroPadding=3 * frameSize)
    run_spectrum = Spectrum(size=frameSize * 4)
    run_FFT = FFT(size=frameSize * 4)
    
    pool = Pool();
    
    fft_array = []
    
    # Now we are ready to start processing.
    # 1. Load audio and pass it through the equal-loudness filter
    audio = MonoLoader(filename=filename)()
    audio = EqualLoudness()(audio)

    # 2. Cut audio into frames and compute for each frame:
    #    spectrum -> spectral peaks -> pitch salience function
    # With startFromZero = False, the first frame is centered at time = 0, instead of half the fremesize
    for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize, startFromZero=False):
        frame = run_windowing(frame)
        spectrum = run_spectrum(frame)
        fft = run_FFT(frame)
        pool.add('spectrum', spectrum)
        fft_array.append( fft) 
    
    return pool['spectrum'], fft_array

    
def calculateSF(spectogram, hopSizeFrames):
    """ Computes the salience function based on harmonic summation
    


    Returns
    -------
    times: list of times of each of the frames of the salience function
    salience: Harmonic summation salience function
    """
    from numpy import arange
    sampleRate = 44100
    run_spectral_peaks = SpectralPeaks(minFrequency=1,
                                       maxFrequency=20000,
                                       maxPeaks=100,
                                       sampleRate=sampleRate,
                                       magnitudeThreshold=0,
                                       orderBy="magnitude")
    run_pitch_salience_function = PitchSalienceFunction()

    pool = Pool();

    for spectrum in spectogram:
        peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
        salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
        pool.add('allframes_salience', salience)

    salience = pool['allframes_salience']
    hopSize = int(hopSizeFrames)
    times = arange(len(pool['allframes_salience'])) * float(hopSize) / sampleRate

    return times, salience
