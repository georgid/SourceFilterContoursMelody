
import matplotlib.pyplot as plt
import os
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from Parameters import Parameters

from contour_classification.experiment_utils import get_data_files

def plot_2features_spaces(track_list, feature1, feature2):
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
    
   
            
    plt.scatter(contour_data[feature1], contour_data[feature2], marker='o', c=contour_data['labels'])
    plt.show()
    
#     plt.scatter(nonvocal_contour_data[feature1], nonvocal_contour_data[feature2], marker='o', c=nonvocal_contour_data['labels'])
#     plt.show()


if __name__ == '__main__':
    #####################################
    
    if len(sys.argv) != 4:
        sys.exit('usage: {}  <contours_path> <feature1_name> <feature2_name>'.format(sys.argv[0]))
    Parameters.contour_URI = sys.argv[1]
    feature1 = sys.argv[2]
    feature2 = sys.argv[3]
    
    dir_tracks = os.path.join( os.path.dirname(os.path.realpath(__file__)) , '../src/contour_classification/melody_trackids_iKala_subset.json' )
 
    with open(dir_tracks, 'r') as fhandle:
            track_list = json.load(fhandle)['tracks']
 
    plot_2features_spaces(track_list, feature1, feature2)

    
