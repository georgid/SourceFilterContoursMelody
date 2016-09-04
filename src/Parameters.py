'''
Created on Aug 31, 2016

@author: georgid
'''
import json
import os

dir_tracks_iKala = os.path.join( os.path.dirname(os.path.realpath(__file__)) , 'contour_classification/melody_trackids_iKala.json' )

class Parameters(object):
    datasetIKala  = True
    iKala_annotation_URI = "/home/georgid/Documents/iKala/PitchLabel/"
    useTimbre = True
    
    if datasetIKala:
        with open(dir_tracks_iKala, 'r') as fhandle:
            track_list = json.load(fhandle)
    
    tracks = track_list['tracks']
