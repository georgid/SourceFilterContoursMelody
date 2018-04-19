'''
Created on Sep 29, 2016

@author: georgid
'''

import numpy as np
import math


varianceWindow = 0.4 # variance on the one side in seconds

def extact_pseudo_fluctogram(contour_bins_SAL, options):
    '''
    fluctogram variatiopn from bin_saliences
    
    Parameters
    -----------------
    
    contour_bins_SAL: array list 
        represents a melody contour - a bin represents a 10-cent-frequency 
    '''
    
    num_frames = len(contour_bins_SAL)
    delta_bins = np.zeros(( num_frames ))
    num_bins = 5 # 10 bins mean 1 semitone 
    
    # create  deltas btw consecutive bins (1-order)
    for i in range(num_frames - 1):
        delta = contour_bins_SAL[i+1] - contour_bins_SAL[i]
        if abs(delta) <=  num_bins: # care about only sub-semitone fluctuations
            delta_bins[i] = delta
    
    fluct = np.zeros(delta_bins.shape)
    
    # compute variance of fluctuations   
    numFrVar = int(math.floor(varianceWindow / options.hopsize))
    for i in range(0, num_frames):
        startIdx = max(0, i-numFrVar)
        endIdx = min( num_frames-1, i + numFrVar )
        
        delta_bins_slice = delta_bins [ startIdx : endIdx + 1 ]
        fluct[i] = np.var( delta_bins_slice )
    
    return fluct



