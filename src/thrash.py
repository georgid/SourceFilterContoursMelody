'''
Created on Sep 2, 2016

@author: georgid
'''


def compute_harmonic_magnitudes_old(contour_bins_SAL, contour_start_time_SAL, fftgram, times, options ):
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
    run_sine_model_synth = SineModelSynth( hopSize_block=options.hopsizeInSamples, sampleRate = options.Fs) 
    run_ifft = IFFT(size = options.windowsizeInSamples);
    run_overl = OverlapAdd (frameSize_block = options.windowsizeInSamples, hopSize_block = options.hopsizeInSamples);
    
    for i, idx in enumerate(range(idx_start, idx_start + len_contour)):
        
        fft = fftgram[idx]
        # convert to freq : 
        f0 = options.minF0 * pow(2, contour_bins_SAL[i] * options.stepNotes / 1200.0)
        hfreq, magns, phases = run_harm_model_anal(fft, f0)
        spectrum, audio_frame = harmonics_to_audio(hfreq, magns, phases, run_sine_model_synth, run_ifft, run_overl )
        pool.add('spectrum', spectrum)
        pool.add('hfreq', hfreq)
        pool.add('magns', magns)
       
    
    times_contour =   contour_start_time_SAL +  numpy.arange(len_contour) *  float(options.hopsizeInSamples) / options.Fs
    return times_contour, pool['spectrum'],  pool['hfreq'], pool['magns']


def loadContour():
    import contour_classification.contour_utils as cc
    contour_fpath = 'recording.ctr'
    cdat = cc.load_contour_data(contour_fpath, normalize=False)
    print cdat



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
    times = np.arange(len(fftgram)) * float(options.hopsizeInSamples) / sampleRate
    options.minF0 = 55
    options.stepNote = 10
    return contour_bins_SAL, contour_start_time_SAL, fftgram, times, options, wavFile



def test_compute_harmonic_ampl(args):
    '''
    test computing of harmonic amplitudes with essentia
    read the saliences of a contour from serizlized json file   
    '''
    
    test_track = '10161_chorus'
    
    contour_bins_SAL, contour_start_time_SAL, fftgram, times, options, wavFile = load_contours(args)
    contour_data_frame, adat = get_data_files(test_track, meltype=1)
    c_times, c_freqs, _ = contours_from_contour_data(contour_data_frame)
    times_contour, spectogram_harm, hfreqs, magns =  compute_harmonic_magnitudes(contour_bins_SAL, contour_start_time_SAL, fftgram, times, options )    

#     pyplot.imshow(spectogram_harm)
#     pyplot.show()

#   try:
#         a = options.stepNotes / 1200.0
# 
#     except  Exception:
#         print(traceback.format_exc())


if __name__ == '__main__':
    pass