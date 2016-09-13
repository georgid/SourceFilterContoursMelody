import contour_utils as cc
import experiment_utils as eu
import mv_gaussian as mv
import clf_utils as cu
import generate_melody as gm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn import metrics
import sklearn
import pandas as pd
import numpy as np
import random
import glob
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from contour_selection_VD import contours_to_vocal
from Parameters import Parameters
from main_contour_extraction import load_labeled_contours
from src.contour_classification.contour_utils import plot_decoded_melody,\
    plot_contours_interactive
from src.main_contour_extraction import label_contours_and_store




sns.set()
from scipy.stats import boxcox

from contour_utils  import getFeatureInfo




def train_and_classify(mdb_files, train, test, dset_contour_dict, dset_annot_dict):
            '''
            labeling should be already done
            '''
            random.shuffle(train)
            n_train = len(train) - (len(test)/2)
            train_tracks = mdb_files[train[:n_train]]
            valid_tracks = mdb_files[train[n_train:]]
            test_tracks = mdb_files[test]
    
            train_contour_dict = {k: dset_contour_dict[k] for k in train_tracks}
            valid_contour_dict = {k: dset_contour_dict[k] for k in valid_tracks}
            test_contour_dict = {k: dset_contour_dict[k] for k in test_tracks}
    
            train_annot_dict = {k: dset_annot_dict[k] for k in train_tracks}
            valid_annot_dict = {k: dset_annot_dict[k] for k in valid_tracks}
            test_annot_dict = {k: dset_annot_dict[k] for k in test_tracks}
    
            reload(eu)
            olap_stats, zero_olap_stats = eu.olap_stats(train_contour_dict)
            print 'overlapped stats on train data...'
            print olap_stats

            len(train_contour_dict)
    
            reload(cc)
    
            anyContourDataFrame = dset_contour_dict[dset_contour_dict.keys()[0]]
    
            #### CONVERT PANDAS DATA to DATA for scikit Learn 
            feats, idxStartFeatures, idxEndFeatures = getFeatureInfo(anyContourDataFrame)
    
            X_train, Y_train = cc.pd_to_sklearn(train_contour_dict,idxStartFeatures,idxEndFeatures)
            X_valid, Y_valid = cc.pd_to_sklearn(valid_contour_dict,idxStartFeatures,idxEndFeatures)
            X_test, Y_test = cc.pd_to_sklearn(test_contour_dict,idxStartFeatures,idxEndFeatures)
            np.max(X_train,0)
    
    
            
            
            #####################  cross-val of best depth of RFC
            reload(cu)
            best_depth, max_cv_accuracy, plot_dat = cu.cross_val_sweep(X_train, Y_train, plot = False)
            print "best depth is {}".format( best_depth)
            print "max_cv_accuracy is {}".format(max_cv_accuracy)
    
            df = pd.DataFrame(np.array(plot_dat).transpose(), columns=['max depth', 'accuracy', 'std'])
    
            ##################### 3.2 TRAIN and CLASSIFY 
            clf = cu.train_clf(X_train, Y_train, best_depth)
    
            reload(cu)
            P_train, P_valid, P_test = cu.clf_predictions(X_train, X_valid, X_test, clf)
            clf_scores = cu.clf_metrics(P_train, P_test, Y_train, Y_test)
            print clf_scores['test']
    
            #### get threshold with best f-measure on validation dataset
            reload(eu)
            best_thresh, max_fscore, plot_data = eu.get_best_threshold(Y_valid, P_valid)
            max_fscore = 0.0
            print "best threshold = %s" % best_thresh
            print "maximum achieved f score = %s" % max_fscore
    
            # classify and add the melody probability for each contour as a field in the dict
            for key in test_contour_dict.keys():
                test_contour_dict[key] = eu.contour_probs(clf, test_contour_dict[key],idxStartFeatures,idxEndFeatures)
    
            ################### 3.3. Melody decoding. 
            #####  viterbi decoding 
            reload(gm)
            mel_output_dict = {}
            for i, key in enumerate(test_contour_dict.keys()):
                print key
                mel_output_dict[key] = gm.melody_from_clf(test_contour_dict[key], prob_thresh=best_thresh)

    #             mel_output_dict[key] = contours_to_vocal(test_contour_dict[key], prob_thresh=best_thresh)
            return mel_output_dict, test_annot_dict, clf, feats     



def eval(mel_output_dict, test_annot_dict, scores):
    ################ EVALUATION
    reload(gm)
    mel_scores = gm.score_melodies(mel_output_dict, test_annot_dict)
    overall_scores = pd.DataFrame(columns=['VR', 'VFA', 'RPA', 'RCA', 'OA'], 
        index=mel_scores.keys())
    overall_scores['VR'] = [mel_scores[key]['Voicing Recall'] for key in mel_scores.keys()]
    overall_scores['VFA'] = [mel_scores[key]['Voicing False Alarm'] for key in mel_scores.keys()]
    overall_scores['RPA'] = [mel_scores[key]['Raw Pitch Accuracy'] for key in mel_scores.keys()]
    overall_scores['RCA'] = [mel_scores[key]['Raw Chroma Accuracy'] for key in mel_scores.keys()]
    overall_scores['OA'] = [mel_scores[key]['Overall Accuracy'] for key in mel_scores.keys()]
    scores.append(overall_scores)
    print "Overall Scores"
    overall_scores.describe()


# plt.ion()


mel_type=1

reload(eu)

scores = []
scores_nm = []

# EDIT: For MedleyDB
#with open('melody_trackids.json', 'r') as fhandle:
#    track_list = json.load(fhandle)
# For Orchset
# with open('melody_trackids_orch.json', 'r') as fhandle:
#     track_list = json.load(fhandle)
# 


    #  for iKala
tracks  = Parameters.tracks

    
dset_contour_dict, dset_annot_dict = load_labeled_contours(tracks, Parameters.contour_URI)

mdb_files, splitter = eu.create_splits(test_size=0.25)



# repeat split into train and test 1 times
for i in range(1):
        for train, test in splitter: # each splitting is repeated 5 times. see ShuffleLabel 
            mel_output_dict, test_annot_dict, clf, feats = train_and_classify(mdb_files, train, test, dset_contour_dict, dset_annot_dict)
#             for track in mel_output_dict.keys():
#                 plot_contours_interactive(dset_contour_dict[track], dset_annot_dict[track], track)
#                 plot_decoded_melody( mel_output_dict[track] )
            eval(mel_output_dict, test_annot_dict, scores )
            
            # GEORGID commented this. as it is not used
            np.argsort(clf.feature_importances_)
            np.sum(clf.feature_importances_)
            feats_sorted = [feats[k] for k in np.argsort(clf.feature_importances_)]

            print feats_sorted


 
print "End"


allscores = scores[0]
for i in range(1,len(scores),1):
    allscores = allscores.append(scores[i])
    print i
    print (len(allscores))
 
 
allscores.to_csv('allscoresNoTonal.csv')
from pickle import dump
picklefile = 'allscores'
with open(picklefile, 'wb') as handle:
    dump(allscores, handle)
print allscores.describe()
 
 


#
# allscores_nm = scores_nm[0]
# for i in range(1,len(scores_nm),1):
#     allscores_nm = allscores_nm.append(scores_nm[i])
#     print i
#     print (len(allscores_nm))
#
# allscores_nm.describe()
#
# from pickle import dump
# picklefile = 'allscores_nm'
# with open(picklefile, 'wb') as handle:
#     dump(allscores_nm, handle)
#
#
#
#
# picklefile = 'allscores'
#
# from pickle import load
# with open(picklefile, 'rb') as handle:
#     b = load(handle)

