'''
Created on Sep 3, 2016
visualize feature distribution
@author: georgid



'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from contour_classification.experiment_utils import get_data_files,\
    olap_stats
from main_contour_extraction import label_contours_and_store
import json
from contour_classification.experiment_utils import compute_all_overlaps
from contour_classification.contour_utils import plot_contours_interactive
from Parameters import Parameters
import pandas as pd
import matplotlib.pyplot as plt

def printLabels(track_list):
    '''
        - how many examples have zero for Vvariance or/and MFCC
    - how many examples have labels 0 and 1
    '''
#     dset_contour_dict_labeled, dset_annot_dict = label_contours_and_store(Parameters.contour_URI, track_list, normalize=False)
    
    feature1 = 'timbre1'
    
    len_label = [0,0]
    len_feature_diff_zero = [0,0]
    for track in track_list:
#         contour_data_= dset_contour_dict_labeled[track]
        contour_data_,_ = get_data_files(track, normalize=False, meltype=1)
        for label in range(2):
             contour_data_label = contour_data_[contour_data_['labels'] == label]
             curr_len = len(contour_data_label)
             len_label[label] += curr_len
         
             if feature1 in contour_data_label.columns:
                 a = contour_data_label[feature1] == 0.0
                 tmpLen = len(contour_data_label[a])
             else:
                 tmpLen = 0
             len_feature_diff_zero[label] += tmpLen
    
    for label in range(2):         
        print 'len of label {} is  {}'.format( label, len_label[label] )
        print 'feature {} =0  {} times'.format(feature1, len_feature_diff_zero[label] )




def plot_2features_spaces(track_list):
    '''
    contour_data gets very big, so better run with 
    run with 
    - what is 2-feature distribution plot 
    '''    ##### plot only melody contours
    
    import contour_classification.experiment_utils as eu
#   make sure this is run in advance:
#     track_list = [test_track]
#    
    from sys import getsizeof
    contour_data, adat = get_data_files(track_list[0],normalize=False, meltype=1)
    for i, track in enumerate(track_list):
            curr_contour_data, adat = get_data_files(track, normalize=False, meltype=1)
            contour_data = contour_data.append(curr_contour_data, ignore_index=True)
#             print getsizeof(contour_data)
#             if i==21:   break

   
#     for track in track_list:
#         contour_data = dset_contour_dict_labeled[track]
#     for column_name in  contour_data.columns.values:
#         print column_name
    nonvocal_contour_data = contour_data[contour_data['labels'] == 0]
    
#     plt.scatter(contour_data['salience mean'], contour_data['pitch std'], marker='o', c=contour_data['labels']) #    
#     plt.legend()
#     plt.show()
    
    feature2 = 'timbre4' 
    feature1= 'timbre3'
    
            
#         plt.scatter(contour_data['timbre1'], contour_data['timbre2'], marker='o', c=contour_data['labels'])
#         plt.show()
#         plt.scatter(contour_data['timbre2'], contour_data['timbre3'], marker='o', c=contour_data['labels'])
#         plt.show()
#     plt.scatter(contour_data['duration'], contour_data['pitch std'], marker='o', c=contour_data['labels'])
#     plt.show()
    
    plt.scatter(nonvocal_contour_data[feature1], nonvocal_contour_data[feature2], marker='o', c=nonvocal_contour_data['labels'])
    plt.show()


def plot_all_contours(tracks):
    '''
    plot all contours in a recording
    '''

    contours_output_path = Parameters.iKala_annotation_URI
    ##### plot only melody contours
    import contour_classification.experiment_utils as eu
#   make sure this is run in advance:
#     track_list = [test_track]
#     dset_contour_dict, dset_annot_dict = eu.compute_all_overlaps(track_list, meltype=1)
#

    dset_contour_dict_labeled, dset_annot_dict = load_labeled_contours(track_list, contours_output_path)
    contour_data = dset_contour_dict_labeled[dset_contour_dict_labeled.keys()[0]]

    for track in tracks:
        contour_data = dset_contour_dict_labeled[track]


        melody_contour_data = contour_data[contour_data['labels'] == 1]
        nonmelody_contour_data = contour_data[contour_data['labels'] == 0]
        plot_contours_interactive(contour_data, dset_annot_dict[track], track)
#         plot_contours_interactive(melody_contour_data, dset_annot_dict[track], track)

        


if __name__ == '__main__':
#     plot_2_dim_chart()
    
    
    track_list = Parameters.tracks
#     track_list = [Parameters.test_track]
#     plot_all_contours(track_list)
   
    
    #####################################
    
#     dir_tracks = os.path.join( os.path.dirname(os.path.realpath(__file__)) , '../src/contour_classification/melody_trackids_iKala_subset.json' )

#     with open(dir_tracks, 'r') as fhandle:
#             track_list = json.load(fhandle)['tracks']

#     plot_2features_spaces(track_list)



############################
#     track_list = ['10161_chorus']
    
    if len(sys.argv) != 2:
        sys.exit('usage: {}  <path-features>'.format(sys.argv[0]))
    path_ = sys.argv[1]
    Parameters.contour_URI = path_
    printLabels(track_list)
