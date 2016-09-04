'''
Created on Jun 23, 2016

feature vocal variance

@author: georgid
'''


from essentia.standard import *
import math
import numpy as np
from matplotlib.pyplot import imshow, show
from matplotlib import pyplot
# from cante.extrBarkBands import extrBarkBands


varianceLength = 1 # in sec
plotting = False
# plotting = True

def extractMFCCs(spectogram):
    '''
    extract mfccs fro spectromra
    '''
    # 30 as suggested by BErnhard Lehner
    num_mfccs = 30
    
    ######## compute MFCCs
    #     maybe set highFrequencyBound=22100
    mfcc = MFCC(numberCoefficients=num_mfccs, numberBands=30)
    
    mfccs_array = np.zeros( (len(spectogram), num_mfccs) )
    
    for i,spectrum in enumerate(spectogram):
    
        mfcc_bands, mfcc_coeffs = mfcc( spectrum )
        mfccs_array[i] = mfcc_coeffs
       
    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    
#     mfccs_T = essentia.array(mfccs_array).T
#     # and plot
#     imshow(mfccs_T, aspect = 'auto', interpolation='none')
#     show() # unnecessary if you started "ipython --pylab" 

    return mfccs_array


def extractVocalVar(mfccs_array, _frameSize, num_mfccs_var, options):
    '''
    vocal variance
    see Lehner et al. - On the reduction of false positives in singing voice detection (2.3)
    '''
    
    mfccs_array = mfccs_array[:, 1:num_mfccs_var+1]
    
    num_mfccs= mfccs_array.shape[1]       
    num_frames = len(mfccs_array)
    
   
    # variance num frames
    numFrVar = int(math.floor(options.Fs * varianceLength / _frameSize))
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