'''
Created on Sep 3, 2016
visualize feature distribution
@author: georgid



'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from contour_classification.contour_utils import plot_contours_interactive
from main_contour_extraction import load_labeled_contours
from Parameters import Parameters
import pandas as pd
import matplotlib.pyplot as plt

def plot_2features_spaces(track_list):
    ##### plot only melody contours
    import contour_classification.experiment_utils as eu
#   make sure this is run in advance:
#     track_list = [test_track]
#     dset_contour_dict, dset_annot_dict = eu.compute_all_overlaps(track_list, meltype=1)
#    

#     Parameters.contour_URI += '/vv_hopS-0.5/'
    dset_contour_dict_labeled, dset_annot_dict = load_labeled_contours(track_list, Parameters.contour_URI )
    from sys import getsizeof
    
    contour_data = dset_contour_dict_labeled[dset_contour_dict_labeled.keys()[0]]
    for key in dset_contour_dict_labeled.keys()[1:]:
        curr_contour_data = dset_contour_dict_labeled[key]
        contour_data = contour_data.append(curr_contour_data, ignore_index=True)
        print getsizeof(contour_data)
#     for track in track_list:
#         contour_data = dset_contour_dict_labeled[track]
#     for column_name in  contour_data.columns.values:
#         print column_name
    nonvocal_contour_data = contour_data[contour_data['labels'] == 0]
    
#     plt.scatter(contour_data['salience mean'], contour_data['pitch std'], marker='o', c=contour_data['labels']) #    
#     plt.legend()
#     plt.show()
    
    feature2 = 'timbre3' 
    feature1= 'timbre2'
    
    for i in range(5): 
        feature1  = 'timbre' + str(i)
        
        for label in range(2):
            contour_data_ = contour_data[contour_data['labels'] == label]

            print 'label' + str(label)
            print 'feature {} =0  {} times'.format(feature1, len(contour_data_[contour_data_[feature1]== 0.0]))
            print 'feature {} !=0 {} times'.format(feature1, len(contour_data_[contour_data_[feature1] != 0.0]) )
         
#         plt.scatter(contour_data['timbre1'], contour_data['timbre2'], marker='o', c=contour_data['labels'])
#         plt.show()
#         plt.scatter(contour_data['timbre2'], contour_data['timbre3'], marker='o', c=contour_data['labels'])
#         plt.show()
    plt.scatter(contour_data['duration'], contour_data[feature2], marker='o', c=contour_data['labels'])
    plt.show()
    
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
    plot_2features_spaces(track_list)
