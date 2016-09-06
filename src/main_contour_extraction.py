'''
Created on Sep 2, 2016

@author: georgid
'''
import os
import json
from src.melodyExtractionFromSalienceFunction import MEFromSF
from src.HarmonicSummationSF import calculateSpectrum, calculateSF
import sys
import pandas as pd
from src.Parameters import Parameters

# mel_type = 1
#     #  for iKala
# if Parameters.datasetIKala:
#         with open('contour_classification/melody_trackids_iKala.json', 'r') as fhandle:
#             track_list = json.load(fhandle)
# tracks = track_list['tracks']



def create_contours_and_store(contours_path):
    import parsing
    
    (args,options) = parsing.parseOptions(sys.argv)
    
    options.pitchContinuity = 27.56
    options.peakDistributionThreshold = 1.3
    options.peakFrameThreshold = 0.7
    options.timeContinuity = 100
    options.minDuration = 100
    options.voicingTolerance = 1
    options.useVibrato = False
    
    options.Fs = 44100
    options.extractionMethod = 'PCC'
    options.plotting = False
    
    
    

    wav_path = "/home/georgid/Documents/iKala/Wavfile/" 
        
    for fileName in Parameters.tracks:

        options.pitch_output_file    = contours_path + fileName
        wavfile_  = wav_path + fileName + '.wav'
        spectogram, fftgram = calculateSpectrum(wavfile_, options.hopsizeInSamples)
        timesHSSF, HSSF = calculateSF(spectogram,  options.hopsizeInSamples)
        HSSF = HSSF.T
        print("Extracting melody from salience function ")
        times, pitch = MEFromSF(timesHSSF, HSSF, fftgram, options)
        

def compute_all_overlaps_and_store(output_contours_path):
    '''
    overlap all contours with ground truth and serialize to pandas csv
    '''
    mel_type = 1
    #  for iKala

    
    # mdb_files, splitter = eu.create_splits(test_size=0.15)
    import contour_classification.experiment_utils as eu
    ########################### 3.1 Contour labeling
    dset_contour_dict, dset_annot_dict = eu.compute_all_overlaps(Parameters.tracks, meltype=mel_type)
    
    
    OLAP_THRESH = 0.5    
    dset_contour_dict_labeled, _, _ = \
            eu.label_all_contours(dset_contour_dict, dset_contour_dict, \
                                  dset_contour_dict, olap_thresh=OLAP_THRESH)
    
#     write to json 
    for key in dset_contour_dict_labeled:
        filename = os.path.join(output_contours_path, key + '.ctr.ovrl')
        dset_contour_dict_labeled[key].to_csv(filename, sep='\t', encoding='utf-8',  index=False)
    for key in dset_annot_dict:
        filename = os.path.join(output_contours_path, key + '.ctr.anno')
        dset_annot_dict[key].to_csv(filename, sep='\t', encoding='utf-8',  index=False)
    
    return dset_contour_dict_labeled, dset_annot_dict


def load_labeled_contours(tracks, contours_path):
    # import labeled contours
    dset_annot_dict = {}
    dset_contour_dict = {}
    for test_track in tracks:
        filename = os.path.join(contours_path, test_track + '.ctr.anno')
        dset_annot_dict[test_track] = pd.read_csv(filename, sep='\t', encoding='utf-8')
        filename = os.path.join(contours_path, test_track + '.ctr.ovrl')
        dset_contour_dict[test_track] = pd.read_csv(filename, sep='\t', encoding='utf-8')
    
    return dset_contour_dict, dset_annot_dict

if __name__ == '__main__':
    
    contours_path = Parameters.iKala_annotation_URI
    create_contours_and_store(contours_path)
    dset_contour_dict_labeled, dset_annot_dict = compute_all_overlaps_and_store(contours_path)


    

#     MEFromFileNumInFolder('../test/', 'output', 1 , options)
# #     MEFromFileNumInFolder(sys.argv[1], sys.argv[2], int(sys.argv[3]), options)
    