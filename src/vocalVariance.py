'''
Created on Jun 23, 2016

feature vocal variance

@author: georgid
'''


from essentia.standard import *
import math
import numpy as np

from Parameters import Parameters
if Parameters.with_MATPLOTLIB:
    from matplotlib.pyplot import imshow, show
    from matplotlib import pyplot
# from cante.extrBarkBands import extrBarkBands
from essentia import Pool

    # params as suggested by BErnhard Lehner
num_mfccs = 29
numberBands = 30
highFrequencyBound = 11025
frameSize_block = 0.7 # 
hopSize_block = 0.1 # s
varianceWindow = 5 * hopSize_block  # variance on the one side in seconds
plotting = False
# plotting = True

def extractMFCCs(audio):
    '''
    extract mfccs from spectromra
    '''
    
    ######## compute MFCCs
    #     maybe set highFrequencyBound=22100
    frameSizeInSamples = int(round(44100 * frameSize_block))
    hopSizeInSamples = int(round(44100 * hopSize_block))
    inputSpectrumSize = frameSizeInSamples / 2 + 1
    
#     inputSpectrumSize = 1025
    mfcc = MFCC(numberCoefficients=num_mfccs, numberBands=numberBands, highFrequencyBound = highFrequencyBound, inputSize=inputSpectrumSize)
    w = Windowing(type = 'hann')
    spectrum = Spectrum()
    mfccs_array = []
    pool = Pool()
    
    audio = essentia.array(audio)
    for frame in FrameGenerator(audio, frameSize = frameSizeInSamples, hopSize = hopSizeInSamples):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        pool.add('mfcc', mfcc_coeffs)
    
    
#     mfccs_array = np.zeros( (len(spectogram), num_mfccs) )
#     for i,spectrum in enumerate(spectogram):
#      
#         mfcc_bands, mfcc_coeffs = mfcc( spectrum )
#         mfccs_array[i] = mfcc_coeffs
       
    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    
#     mfccs_T = essentia.array(pool['mfcc']).T
#     # and plot
#     imshow(mfccs_T, aspect = 'auto', interpolation='none')
#     show() # unnecessary if you started "ipython --pylab" 

    return pool['mfcc']



def compute_var_mfccs(mfccs_array, hl_mfcc_coeff, options):
    '''
    vocal variance
    see Lehner et al. - On the reduction of false positives in singing voice detection (2.3)
    
    params:
    hl_mfcc_coeff - from 1 to to to  mfcc coefficient
    '''
    
    mfccs_array = mfccs_array[:, 1:hl_mfcc_coeff+1]
    
    num_mfccs= mfccs_array.shape[1]       
    num_frames = len(mfccs_array)
    
   
    # variance num frames
    numFrVar = int(math.floor(varianceWindow / hopSize_block))
    vocal_var_array = np.zeros(mfccs_array.shape)
    
    
    for i in range(0, num_frames):
        startIdx = max(0,i-numFrVar)
        endIdx = min( num_frames-1, i + numFrVar )
    
        # iterate over mfccs
        for coeff in range(num_mfccs):
            mfcc_slice = mfccs_array [ startIdx : endIdx + 1, coeff ]
            vocal_var_array[i, coeff] = np.var( mfcc_slice )
       
     
    # and plot
    if plotting:
        # transpose to have it in a better shape
        # we need to convert the list to an essentia.array first (== numpy.array of floats)
        vocal_var_T = essentia.array(vocal_var_array).T
        imshow(vocal_var_T, aspect = 'auto', interpolation='none')  
        pyplot.show()
        
    return vocal_var_array