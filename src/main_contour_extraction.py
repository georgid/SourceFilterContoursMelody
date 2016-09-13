'''
Created on Sep 2, 2016

@author: georgid
'''
import os
import json
from melodyExtractionFromSalienceFunction import MEFromSF
from HarmonicSummationSF import calculateSpectrum, calculateSF
import sys
import pandas as pd
from Parameters import Parameters
import numpy as np
from contour_classification.experiment_utils import get_data_files
from contour_classification.contour_utils import contours_from_contour_data
from timbreFeatures import compute_timbre_features
from melodyExtractionFromSalienceFunction import saveContours
import parsing
from timbreFeatures import contour_to_audio

# mel_type = 1
#     #  for iKala
# if Parameters.datasetIKala:
#         with open('contour_classification/melody_trackids_iKala.json', 'r') as fhandle:
#             track_list = json.load(fhandle)
# tracks = track_list['tracks']



def create_contours_and_store(tracks, contours_output_path):
    import parsing
    
    (args,options) = parsing.parseOptions(sys.argv)
    
    options.pitchContinuity = 27.56
    options.peakDistributionThreshold = 1.3
#     options.peakDistributionThreshold = 0.9
    options.peakFrameThreshold = 0.7
#     options.peakFrameThreshold = 0.9
    options.timeContinuity = 100

#     options.timeContinuity = 50     # medley DB
    options.minDuration = 100
    options.voicingTolerance = 1
    options.useVibrato = False
    
    options.Fs = 44100
    options.extractionMethod = 'PCC'
    options.plotting = False
    
    
    

           
    for fileName in tracks:

        options.pitch_output_file    = contours_output_path + fileName
        wavfile_  = Parameters.iKala_wav_URI + fileName + '.wav'
        spectogram, fftgram = calculateSpectrum(wavfile_, options.hopsizeInSamples)
        timesHSSF, HSSF = calculateSF(spectogram,  options.hopsizeInSamples)
        HSSF = HSSF.T
        print("Extracting melody from salience function ")
        times, pitch = MEFromSF(timesHSSF, HSSF, fftgram, options)
        

def load_contour_and_extractTimbre_and_save(tracks, contours_output_path, options):
    
    '''
    toAudio - if True store resynthesized audio for each contour and return without contour extraction
    else: extract Timbre and save
    '''
    for track in tracks:
        _, fftgram = calculateSpectrum(Parameters.iKala_wav_URI + track + '.wav', options.hopsizeInSamples)
        timestamps_recording = np.arange(len(fftgram)) * float(options.hopsizeInSamples) / options.Fs
            
        contour_data_frame, adat = get_data_files(track, meltype=1)
        contours_start_times_df, contours_bins_df, contours_saliences_SAL_df = contours_from_contour_data(contour_data_frame)
        
        contours_bins_SAL = []
        contours_saliences_SAL = []
        for (freqs, saliences) in zip(contours_bins_df.iterrows(), contours_saliences_SAL_df.iterrows()) :
            freqs = freqs[1].values
            freqs = freqs[~np.isnan(freqs)] # remove trailing nans
            contours_bins_SAL.append(freqs)
            
            saliences = saliences[1].values
            saliences = saliences[~np.isnan(saliences)] # remove trailing nans
            contours_saliences_SAL.append(saliences)
            
        contours_start_times_SAL = contours_start_times_df.values[:,0]
        options.saveContours = True
        options.track = track
        
        options.pitch_output_file    = os.path.join(contours_output_path, track)
        
        if Parameters.extract_timbre: 
            contourTimbre = compute_timbre_features(contours_bins_SAL, contours_start_times_SAL, fftgram, timestamps_recording, options)
            saveContours(options, options.stepNotes, contours_bins_SAL, contours_saliences_SAL, contours_start_times_SAL, contourTimbre)
        if Parameters.to_audio: # simply resynth to audio
            options.contours_output_path = os.path.join(contours_output_path,  Parameters.features_MATLAB_URI )
            contour_to_audio(contours_bins_SAL, contours_start_times_SAL, fftgram, timestamps_recording, options)


def label_contours_and_store(output_contours_path, tracks):
    '''
    overlap all contours with ground truth and serialize to pandas csv
    '''
    mel_type = 1
    #  for iKala

    # mdb_files, splitter = eu.create_splits(test_size=0.15)
    import contour_classification.experiment_utils as eu
    ########################### 3.1 Contour labeling
    dset_contour_dict, dset_annot_dict = eu.compute_all_overlaps(tracks, meltype=mel_type)
    
    
      
    dset_contour_dict_labeled, _, _ = \
            eu.label_all_contours(dset_contour_dict, dset_contour_dict, \
                                  dset_contour_dict, olap_thresh=Parameters.OLAP_THRESH)
    
#     write to json 
    for key in dset_contour_dict_labeled:
        filename = os.path.join(output_contours_path, key + '.ctr.ovrl')
        dset_contour_dict_labeled[key].to_csv(filename, sep='\t', encoding='utf-8',  index=False)
    for key in dset_annot_dict:
        filename = os.path.join(output_contours_path, key + '.ctr.anno')
        dset_annot_dict[key].to_csv(filename, sep='\t', encoding='utf-8',  index=False)
    
    return dset_contour_dict_labeled, dset_annot_dict


def load_labeled_contours(tracks, contours_output_path):
    # import labeled contours
    dset_annot_dict = {}
    dset_contour_dict = {}
    for test_track in tracks:
        filename = os.path.join(contours_output_path, test_track + '.ctr.anno')
        dset_annot_dict[test_track] = pd.read_csv(filename, sep='\t', encoding='utf-8')
        filename = os.path.join(contours_output_path, test_track + '.ctr.ovrl')
        dset_contour_dict[test_track] = pd.read_csv(filename, sep='\t', encoding='utf-8')
    
    return dset_contour_dict, dset_annot_dict

if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        sys.exit('usage: {} <path-to-ikala> <create_contours=1> <extractTimbre>'.format(sys.argv[0]))
    path_ = sys.argv[1]
    Parameters.iKala_URI = path_

    args, options = parsing.parseOptions(sys.argv)
    whichStep_ = int(sys.argv[2]) # 
    Parameters.extract_timbre  = int(sys.argv[3])
    
    
#     Parameters.contour_URI += '/vv_hopS-0.5/'
    if whichStep_ == 1:
        if not os.path.exists(Parameters.contour_URI):
            os.mkdir(Parameters.contour_URI)
            
        create_contours_and_store(Parameters.tracks, Parameters.contour_URI)
    
    elif whichStep_ == 2:
       
        load_contour_and_extractTimbre_and_save(Parameters.tracks, Parameters.contour_URI, options)
    
        dset_contour_dict_labeled, dset_annot_dict = label_contours_and_store(Parameters.contour_URI, Parameters.tracks)

    

#     MEFromFileNumInFolder('../test/', 'output', 1 , options)
# #     MEFromFileNumInFolder(sys.argv[1], sys.argv[2], int(sys.argv[3]), options)
    