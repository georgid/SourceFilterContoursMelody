'''
Created on Sep 3, 2016
visualize feature distribution
@author: georgid
'''

from src.contour_classification.contour_utils import plot_contours_interactive
from src.main_contour_extraction import load_labeled_contours
from src.Parameters import Parameters
import pandas as pd

if __name__ == '__main__':
    contours_path = Parameters.iKala_annotation_URI

       ##### plot only melody contours
    import src.contour_classification.experiment_utils as eu

#   make sure this is run in advance:     
#     track_list = [test_track]
#     dset_contour_dict, dset_annot_dict = eu.compute_all_overlaps(track_list, meltype=1)
#     

    track_list = [Parameters.test_track]
#     track_list = Parameters.tracks
    dset_contour_dict_labeled, dset_annot_dict = load_labeled_contours(track_list, contours_path)
    
    contour_data = dset_contour_dict_labeled[dset_contour_dict_labeled.keys()[0]]
    for key in dset_contour_dict_labeled.keys()[1:]:
        curr_contour_data = dset_contour_dict_labeled[key]
        contour_data = contour_data.append(curr_contour_data, ignore_index=True)
    
#     contour_data = dset_contour_dict_labeled[test_track]
    melody_contour_data =  contour_data[contour_data['labels'] == 1]
    nonmelody_contour_data =  contour_data[contour_data['labels'] == 0]
    
    import matplotlib.pyplot as plt
    plot_contours_interactive(contour_data, dset_annot_dict[Parameters.test_track], Parameters.test_track)
    
#     for column_name in  contour_data.columns.values:
#         print column_name
#     plt.scatter(contour_data['salience mean'], contour_data['pitch std'], marker='o', c=contour_data['labels'])
#     print contour_data['pitch std']
    plt.scatter(contour_data['timbre2'], contour_data['timbre3'], marker='o', c=contour_data['labels'])

    plt.show()
