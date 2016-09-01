'''
Created on Aug 31, 2016

@author: georgid
'''

import contour_utils as cc
import numpy as np
import pandas as pd

def contours_to_vocal(contour_data, prob_thresh=0.5):
    '''
    mark regions as vocal or non-vocal based on classified contours
    if there is at least one melodic contour at a frame, consider it vocal
    needed for final result

    Parameters
    ----------
    contour_data : DataFrame or dict of DataFrames
        DataFrame containing labeled features.
    prob_thresh : float
        Threshold that determines positive class

    Returns
    -------
    mel_output : Series
        Pandas Series with time stamp as index and f0 as values
    '''

    contour_threshed = contour_data[contour_data['mel prob'] >= prob_thresh]

    if len(contour_threshed) == 0:
        print "Warning: no contours above threshold."
        contour_times, _, _ = \
            cc.contours_from_contour_data(contour_data, n_end=4)

        hopsizeInSamples = 128.0
        step_size = hopsizeInSamples/44100.0  # contour time stamp step size
        mel_time_idx = np.arange(0, np.max(contour_times.values.ravel()) + 1,
                                 step_size)
        mel_output = pd.Series(np.zeros(mel_time_idx.shape),
                               index=mel_time_idx)
        return mel_output

    # get separate DataFrames of contour time, frequency, and probability
    contour_times, contour_freqs, _ = \
        cc.contours_from_contour_data(contour_threshed, n_end=4)

    # make frequencies below probability threshold negative
    #contour_freqs[contour_data['mel prob'] < prob_thresh] *= -1.0

    probs = contour_threshed['mel prob']
    contour_probs = pd.concat([probs]*contour_times.shape[1], axis=1,
                              ignore_index=True)

    contour_num = pd.DataFrame(np.array(contour_threshed.index))
    contour_nums = pd.concat([contour_num]*contour_times.shape[1], axis=1,
                             ignore_index=True)

    avg_freq = contour_freqs.mean(axis=1)

    # create DataFrame with all unwrapped [time, frequency, probability] values.
    mel_dat = pd.DataFrame(columns=['time', 'f0', 'probability', 'c_num'])
    mel_dat['time'] = contour_times.values.ravel()
    mel_dat['f0'] = contour_freqs.values.ravel()
    mel_dat['probability'] = contour_probs.values.ravel()
    mel_dat['c_num'] = contour_nums.values.ravel()

    # remove rows with NaNs
    mel_dat.dropna(inplace=True)

    # sort by probability then by time
    # duplicate times with have maximum probability value at the end
    mel_dat.sort(columns='probability', inplace=True)
    mel_dat.sort(columns='time', inplace=True)

    hopsizeInSamples = 128.0
    # compute evenly spaced time grid for output
    step_size = hopsizeInSamples/44100.0  # contour time stamp step size
    mel_time_idx = np.arange(0, np.max(mel_dat['time'].values) + 1, step_size)

    # find index in evenly spaced grid of estimated time values
    old_times = mel_dat['time'].values
    reidx = np.searchsorted(mel_time_idx, old_times)
    shift_idx = (np.abs(old_times - mel_time_idx[reidx - 1]) < \
                 np.abs(old_times - mel_time_idx[reidx]))
    reidx[shift_idx] = reidx[shift_idx] - 1
    
    mel_dat['reidx'] = reidx
    
            # initialize output melody
    mel_output = pd.Series(np.zeros(mel_time_idx.shape), index=mel_time_idx)
    
    # dummy frequency of 1, on all for which there is a detected contour
    for i in mel_dat['reidx']:
        mel_output.set_value(mel_output.index[i],1.0)

    return mel_output
    
    