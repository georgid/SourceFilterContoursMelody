'''
Created on Aug 31, 2016

@author: georgid
'''
import json
import os

subset = 'v'
 

class   Parameters(object):
    
    with_MATPLOTLIB = False
    datasetIKala  = True
    
    iKala_URI = '/home/georgid/Documents/iKala/'
    medleyDbURI =  '/home/georgid/Documents/medleyDB/'
    iKala_annotation_URI = iKala_URI + "/PitchLabel/"
    
    iKala_wav_URI = iKala_URI + "/Wavfile//"
    
    OLAP_THRESH = 0.5 
    
    
    useTimbre_for_classification = True
    use_SAL_for_classification = True
    read_features_from_MATLAB = True # load .arff files per contour extracted with Lehner's code matlab
    dim_timbre = 30 # MFCC from B. Lehner
    dim_timbre = 5 # vocal var

    extract_timbre = 0
    
    to_audio = False
    features_MATLAB_URI = 'SVD2015/varMFCC_29_30_0_0.5_0t/300_200_300/5_1_5/'
    features_MATLAB_URI = 'SVD2015/MFCC_29_30_0_0.5_0t/300_200_300/'
    
    if datasetIKala:
        
        contour_URI = iKala_URI + '/Conv_mu-1_G-0_LHSF-0_pC-27.56_pDTh-1.3_pFTh-0.7_tC-100_mD-100_vxTol-0.2/'
#         contour_URI = iKala_URI + '/Conv_mu-1_G-0_LHSF-0_pC-27.56_pDTh-0.9_pFTh-0.9_tC-100_mD-100_vxTol-0.2/'

#         contour_URI = iKala_URI + '/Conv_mu-1_G-0_LHSF-0_pC-27.56_pDTh-1.3_pFTh-0.9_tC-50_mD-100_vxTol-0.2/'

        
        dir_tracks = os.path.join( os.path.dirname(os.path.realpath(__file__)) , 'contour_classification/melody_trackids_iKala.json' )
#         dir_tracks = os.path.join( os.path.dirname(os.path.realpath(__file__)) , 'contour_classification/melody_trackids_iKala_subset.json' )
#         dir_tracks = os.path.join( os.path.dirname(os.path.realpath(__file__)) , 'contour_classification/melody_trackids_iKala_subset_small.json' )

        with open(dir_tracks, 'r') as fhandle:
            track_list = json.load(fhandle)
    else: # medley DB
        
        contour_URI = medleyDbURI + '/Conv_mu-1_G-0_LHSF-0_pC-27.56_pDTh-0.9_pFTh-0.9_tC-50_mD-100_vxTol-0.2/'
        contour_URI = medleyDbURI + '/Conv_mu-1_G-0_LHSF-0_pC-27.56_pDTh-1.3_pFTh-0.9_tC-50_mD-100_vxTol-0.2/'
        
        dir_tracks = os.path.join( os.path.dirname(os.path.realpath(__file__)) , 'contour_classification/melody_trackids.json' )
        dir_splits = os.path.join( os.path.dirname(os.path.realpath(__file__)) , 'contour_classification/v_i_splits.json' )

        if subset == 'all':
            with open(dir_tracks, 'r') as fhandle:
                track_list = json.load(fhandle)
        else:
            with open(dir_splits, 'r') as fhandle:
                vi_dict = json.load(fhandle)
            if subset == 'i':
                track_list = [k for k in vi_dict.keys() if vi_dict[k] == "i"]
            if subset == 'v':
                track_list = [k for k in vi_dict.keys() if vi_dict[k] == "v"]
                with open(dir_tracks, 'r') as fhandle:
                    track_list = json.load(fhandle)
            
    tracks = track_list['tracks']   
    
    test_track = '10161_chorus'
    
    CONTOUR_EXTENSION = '.pitch.ctr'
    
    CONTOUR_EXTENSION_OVRL = '.ctr.ovrl'
    
    with_plotting = False
    
    ###### not used
    harmonicTreshold = -70
    wSize = 2048
    nHarmonics = 30
    