'''
Created on Aug 31, 2016

@author: georgid
'''
import json
import os

dir_tracks_iKala = os.path.join( os.path.dirname(os.path.realpath(__file__)) , 'contour_classification/melody_trackids_iKala.json' )

class Parameters(object):
    datasetIKala  = True
    iKala_URI = '/home/georgid/Documents/iKala/'
    iKala_annotation_URI = iKala_URI + "/PitchLabel/"
    iKala_wav_URI = iKala_URI + "/Wavfile//"
    useTimbre = False
    
    if datasetIKala:
        with open(dir_tracks_iKala, 'r') as fhandle:
            track_list = json.load(fhandle)
    
    tracks = track_list['tracks']
    
    test_track = '10161_chorus'
    
    CONTOUR_EXTENSION = ''
    if useTimbre:
        CONTOUR_EXTENSION = '.timbre'
    CONTOUR_EXTENSION += '.ctr' 