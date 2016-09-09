""" Utility functions for processing contours """

import pandas as pd
import numpy as np
import mir_eval
import sys
sys.path.append('..')
from timbreFeatures import get_ts_contour
from Parameters import Parameters
from timbreFeatures import compute_harmonic_magnitudes
from HarmonicSummationSF import calculateSpectrum
import parsing

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
except:
    print "matplotlib or seaborn not available"

def loadpickle(picklefile):
    from pickle import load
    try:
        with open(picklefile, 'rb') as handle:
            b = load(handle)
    except:
        "Pickle file not found: " + picklefile
    return b

def load_contour_data(fpath, normalize=True):
    """ Load contour data from vamp output csv file.
    Initializes DataFrame to have all future columns.

    Parameters
    ----------
    fpath : str
        Path to vamp output csv file.

    Returns
    -------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    """
    try:
        contour_data = pd.read_csv(fpath, header=None, index_col=None,
                                   delimiter=',').astype(float)
        del contour_data[0]  # all zeros
        del contour_data[1]  # just an unnecessary  index
        headers = contour_data.columns.values.astype('str')
        headers[0:12] = ['onset', 'offset', 'duration', 'pitch mean', 'pitch std',
                         'salience mean', 'salience std', 'salience tot',
                         'vibrato', 'vib rate', 'vib extent', 'vib coverage']
        contour_data.columns = headers
    except:
        contour_data = loadpickle(fpath)
        # trying to load with pickle

    # Check if there is any column with all nans... it should not be considered
    df = contour_data.isnull().all()
    if np.where(df)[0]:
        contour_data = contour_data.drop(contour_data.columns[np.where(df)[0][0]], axis=1)

    #   To ensure the contour has a duration > 0
    contour_data['duration'] = np.fmax(contour_data['duration'].values,0.001)

    contour_data.num_end_cols = 0
    contour_data['overlap'] = -1  # overlaps are unset
    contour_data['labels'] = -1  # all labels are unset
    contour_data['melodiness'] = ""
    contour_data['mel prob'] = -1
    contour_data.num_end_cols = 4

    if normalize:
        contour_data = normalize_features(contour_data)

    return contour_data


def normalize_features(contour_data):
    """ Normalizes (trackwise) features in contour_data.
    Adds labels column with all labels unset.

    Parameters
    ----------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    normalize : Bool
        If true, performs trackwise normalization over salience.

    Returns
    -------
    contour_data : DataFrame
        Pandas data frame with normalized contour feature data.
    """

    _, _, contour_sal = contours_from_contour_data(contour_data)

    # maximum salience value across all contours
    sal_max = contour_sal.max().max()

    # normalize salience features by max salience
    contour_data['salience mean'] = contour_data['salience mean']/sal_max
    contour_data['salience std'] = contour_data['salience std']/sal_max

    # normalize saience total by max salience and duration
    contour_data['salience tot'] = \
        contour_data['salience tot']/(sal_max*contour_data['duration'])

    # compute min and max duration
    dur_min = contour_data['duration'].min()
    dur_max = contour_data['duration'].max()

    # normalize duration to be between 0 and 1
    contour_data['duration'] = \
        (contour_data['duration'] - dur_min)/(dur_max - dur_min)

    # give standardized duration back to total salience
    contour_data['salience tot'] = \
        contour_data['salience tot']*contour_data['duration']

    return contour_data


def contours_from_contour_data(contour_data, n_start=12, n_end=4):
    """ Get raw contour information from contour data

    Parameters
    ----------
    contour_data : DataFrame
        Pandas data frame with all contour data.

    Returns
    -------
    contour_times : DataFrame
        Pandas data frame with all raw contour times.
    contour_freqs : DataFrame
        Pandas data frame with all raw contour frequencies (Hz).
    contour_sal : DataFrame
        Pandas data frame with all raw contour salience values.
    """

    if 'first_time' in contour_data.columns:
        n_start = contour_data.columns.get_loc('first_time')

    # Check if there is any column with all nans... it should not be considered
    # df = contour_data.isnull().all()
    # if np.where(df)[0]:
    #     n_end = contour_data.shape[1]-np.where(df)[0][0]
    #


    contour_times = contour_data.iloc[:, n_start:-n_end:3]
    contour_freqs = contour_data.iloc[:, n_start+1:-n_end:3]
    contour_sal = contour_data.iloc[:, n_start+2:-n_end:3]

    return contour_times, contour_freqs, contour_sal


def load_annotation(fpath):
    """ Load an annotation file into a pandas Series.
    Add column with frequency values also converted to cents.

    Parameters
    ----------
    fpath : str
        Path to annotation file.

    Returns
    -------
    annot_data : DataFrame
        Pandas data frame with all annotation data.
    """
    # try:
    #     annot_data = pd.read_csv(fpath, parse_dates=True,
    #                  index_col=False, header=None)
    # except:
    #     annot_data = pd.read_csv(fpath, parse_dates=True,
    #                  index_col=False, header=None,sep='\t')

    # For Orchset
    separator = '\t'

    # For MedleyDB
    separator = ','

    annot_data = pd.read_table(fpath, parse_dates=True,
                         index_col=False,header=None,sep=separator)
    
    #### for iKala edited by georgi
    if Parameters.datasetIKala:
        framesPerSec = 31.25
        timestamps = np.arange(0.5,len(annot_data),1) / framesPerSec
        annot_data.insert(0, 'time', timestamps)
        annot_data.columns = ['time', 'f0']
        # convert to hz
#         hz_f0 =  mir_eval.util.midi_to_hz(annot_data['f0']) 
        annot_data.ix[annot_data.f0==0,'f0'] = -np.inf # zero MIDIs to -inf.  because in midi2freq they will become 0 Hz.

        hz_f0 = 440.0 * (2.0 ** ((annot_data['f0'] - 69.0)/12.0))        
        annot_data['f0'] = hz_f0
    # Add column with annotation values in cents
    annot_data['cents'] = 1200.0*np.log2(annot_data['f0']/55.0)

    return annot_data


def plot_contours(contour_data, annot_data, contour_data2=None):
    """ Plot contours against annotation.

    Parameters
    ----------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    annot_data : DataFrame
        Pandas data frame with all annotation data.
    """
    if contour_data2 is not None:
        c_times2, c_freqs2, _ = contours_from_contour_data(contour_data2)
        for (times, freqs) in zip(c_times2.iterrows(), c_freqs2.iterrows()):
            times = times[1].values
            freqs = freqs[1].values
            times = times[~np.isnan(times)]
            freqs = freqs[~np.isnan(freqs)]
            plt.plot(times, freqs, '.c')

    c_times, c_freqs, _ = contours_from_contour_data(contour_data)
    plt.figure()
    for (times, freqs) in zip(c_times.iterrows(), c_freqs.iterrows()):
        times = times[1].values
        freqs = freqs[1].values
        times = times[~np.isnan(times)]
        freqs = freqs[~np.isnan(freqs)]
        plt.plot(times, freqs, '.r')

    plt.plot(annot_data['time'], annot_data['f0'], '.k')
    plt.show()

def plot_contours_interactive(contour_data, annot_data, track, contour_data2=None):
    """ Plot contours against annotation.

    Parameters
    ----------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    annot_data : DataFrame
        Pandas data frame with all annotation data.
    """
    
    list_melodycontour_indices = [ 3, 5, 6, 12, 15]
    list_nonmelodycontour_indices = [ 7]
    
    if contour_data2 is not None:
        c_times2, c_freqs2, _ = contours_from_contour_data(contour_data2)
        for (times, freqs) in zip(c_times2.iterrows(), c_freqs2.iterrows()):
            times = times[1].values
            freqs = freqs[1].values
            times = times[~np.isnan(times)]
            freqs = freqs[~np.isnan(freqs)]
            plt.plot(times, freqs, '.c')

    c_times, c_freqs, _ = contours_from_contour_data(contour_data)
    plt.figure()
    
    for i, (times, freqs) in enumerate(zip(c_times.iterrows(), c_freqs.iterrows()) ):
        times = times[1].values
        freqs = freqs[1].values
        times = times[~np.isnan(times)]
        freqs = freqs[~np.isnan(freqs)]
        print "contour {} at time {}".format(i, times[0])
        plt.plot(annot_data['time'], annot_data['f0'], '.k')
        plt.plot(times, freqs, '.r')
        plt.show()
        
        if i in list_melodycontour_indices: # show harmonics for only melody 
            compute_and_plot_harmoncis(times, freqs, track)
#         if i in list_nonmelodycontour_indices: # show harmonics for only  non-melody
#             compute_and_plot_harmoncis(times, freqs, track)

def plot_decoded_melody(mel_output):
    '''
    plot decoded melody for a whole recording
    param: mel_output : pandas Series
    '''
    
    times = []
    values = []        
    for time, val in mel_output.iteritems():
        times.append(time)
        values.append(val)
    
    plt.plot(times, values) 
    plt.show()

def compute_and_plot_harmoncis(times, freqs, track ):
        '''
        visualize harm magnitudes
        skip timeframes with any hfreq = 0
        '''
        
        args, options = parsing.parseOptions([])
        _, fftgram = calculateSpectrum(track + '.wav', 128)
        timestamps_recording = np.arange(len(fftgram)) * float(128) / 44100
        times_contour, idx_start = get_ts_contour(freqs, times[0], timestamps_recording, options)
        hfreqs, magns, _ =  compute_harmonic_magnitudes(freqs,  fftgram, idx_start, options )
        # how many harmonics
        until_Harm = 10
        for i in range(len(hfreqs)):
            curr_hFreqs = hfreqs[i]
            indices_zeros = np.where(curr_hFreqs[:until_Harm] == 0)[0] # sskip timeframes with any hfreq = 0
            if len(indices_zeros) != 0:
                continue
            plt.plot(curr_hFreqs[:until_Harm], magns[i][:until_Harm])
            plt.show()
    
    
def compute_overlap(contour_data, annot_data):
    """ Compute percentage of overlap of each contour with annotation.

    Parameters
    ----------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    annot_data : DataFrame
        Pandas data frame with all annotation data.

    Returns
    -------
    feature_data : DataFrame
        Pandas data frame with feature_data and labels.
    """
    c_times, c_freqs, _ = contours_from_contour_data(contour_data)

    for (times, freqs) in zip(c_times.iterrows(), c_freqs.iterrows()):
        row_idx = times[0]
        times = times[1].values
        freqs = freqs[1].values

        # remove trailing NaNs
        times = times[~np.isnan(times)]
        freqs = freqs[~np.isnan(freqs)]

        # get segment of ground truth matching this contour
        gt_segment = annot_data[annot_data['time'] >= times[0]]
        gt_segment = gt_segment[gt_segment['time'] <= times[-1]]

        if len(gt_segment['time']) == 0:
            # To avoid error in mir_eval
            res = mir_eval.melody.evaluate(np.zeros(1),np.zeros(1), times, freqs)
        else:
            # compute metrics
            res = mir_eval.melody.evaluate(gt_segment['time'].values,
                                       gt_segment['f0'].values, times, freqs)

        contour_data.ix[row_idx, 'overlap'] = res['Overall Accuracy']
    return contour_data


def label_contours(contour_data, olap_thresh):
    """ Compute contours based on annotation.
    Contours with at least olap_thresh overlap with annotation
    are labeled as positive examples. Otherwise negative.

    Parameters
    ----------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    annot_data : DataFrame
        Pandas data frame with all annotation data.
    olap_thresh : float
        Overlap threshold for positive examples

    Returns
    -------
    contour_data : DataFrame
        Pandas data frame with contour_data and labels.
    """
    contour_data['labels'] = 1*(contour_data['overlap'] > olap_thresh)
    return contour_data


def contour_glass_ceiling(contour_fpath, annot_fpath):
    """ Get subset of contour data that overlaps with annotation.

    Parameters
    ----------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    annot_data : DataFrame
        Pandas data frame with all annotation data.

    Returns
    -------
    olap_contours : DataFrame
        Subset of contour_data that overlaps with annotation.
    """
    # indices
    onset = 2
    offset = 3
    duration = 4
    pitch_mean = 5
    pitch_std = 6
    salience_mean = 7
    salience_std = 8
    salience_tot = 9
    vibrato = 10
    vibrato_rate = 11
    vibrato_extent = 12
    vibrato_coverage = 13
    first_time = 14

    hopsizeInSamples = 128.0

    def time_to_index(t):
        return int(np.round(t * 44100 / hopsizeInSamples))

    ###########################################################################
    def contours_to_activation(contours, n_times):

        if isinstance(contours,pd.DataFrame):
            # Edit for contours from melodia, should be 14, from contours from ISMIR2016, should be 12 (Orchset eval. with melodia contours)
            featName, startFeat,EndFeat = getFeatureInfo(contours)
            first_time =  EndFeat+1

        c_last = contours.values[-1]
        nanind = np.where(np.isnan(c_last))[0]
        if len(nanind) > 0:
            nanind = nanind[0]
            c_last = c_last[:nanind]
        activation = [[] for x in range(time_to_index(n_times) + 1)]

        for c_num in contours.values:
            nanind = np.where(np.isnan(c_num))[0]
            if len(nanind) > 0:
                nanind = nanind[0]
                c_num = c_num[first_time:nanind]
            else:
                c_num = c_num[first_time:]
            ind = 0
            while ind < len(c_num):
                time_ind = time_to_index(c_num[ind])
                activation[time_ind].append(c_num[ind+1])
                ind += 3

        return activation

    ###########################################################################
    def pitch_accuracy(ref, activation):
        hits = 0
        misses = 0
        for rval in ref.values:
            ind = time_to_index(rval[0])
            if rval[1] > 0:
                match = False
                for v in activation[ind]:
                    if np.abs(1200*np.log2(v/rval[1])) < 50:
                        match = True
                if match:
                    hits += 1
                else:
                    misses += 1
        return hits / float(hits + misses)
    ###########################################################################

    ref = pd.read_csv(annot_fpath,
                      header=None, index_col=False)
    ref = pd.read_csv(annot_fpath,
                      header=None, sep = '\t',index_col=False)
    try:
        contours = loadpickle(contour_fpath)

        contours.drop('mel prob',inplace=True,axis=1)
        contours.drop('overlap',inplace=True,axis=1)
        contours.drop('labels',inplace=True,axis=1)
        contours.drop('melodiness',inplace=True,axis=1)
    except:
        # In case the contours are csv (created with the hacked MELODIA VAMP plugin from J.Salomon)
        try:
            contoursr = pd.read_csv(contour_fpath,header=None, index_col=False)
            # First two columns are irrelevant
            contours = contoursr.iloc[:,2:]
        except:
            print  "No contours could be loaded"


    n_times = len(ref)
    activation = contours_to_activation(contours, n_times)
    rpa = pitch_accuracy(ref, activation)

    return rpa



def join_contours(contours_list):
    """ Merge features for a multiple track into a single DataFrame

    Parameters
    ----------
    contours_list : list of DataFrames
        List of Pandas data frames with labeled features.

    Returns
    -------
    all_contours : DataFrame
        Merged feature data.
    """
    all_contours = pd.concat(contours_list, ignore_index=False)
    return all_contours

def getFeatureInfo(contourDF):
    if Parameters.useTimbre:
        if 'first_time' in contourDF.columns:
            idxEndFeatures = contourDF.columns.get_loc('first_time')-1
        else:
            sys.exit('contour DataFrame does not have field first_time')
    else:
            idxEndFeatures = 11     # From the original implementation, 12 is the last feature
    
    if 'duration' in contourDF.columns:
        idxStartFeatures = contourDF.columns.get_loc('duration')
    else:
        idxStartFeatures=0
    feats = contourDF.columns[idxStartFeatures:idxEndFeatures+1]
    return feats,idxStartFeatures,idxEndFeatures


def pd_to_sklearn(contour_data,idxfirstfeature=0,idxEndFeatures=11):
    """ Convert pandas data frame to sklearn style features and labels

    Parameters
    ----------
    contour_data : DataFrame or dict of DataFrames
        DataFrame containing labeled features.

    Returns
    -------
    features : np.ndarray
        fetures (n_samples x n_features)
    labels : np.1darray
        Labels (n_samples,)
    """
    offset = 0
    #  Reduce before join for speed and memory saving
    if isinstance(contour_data, dict):
        red_list = []
        lab_list = []
        for key in contour_data.keys():
            # Edit ISMIR offset
            #if isinstance(contour_data[key],pd.DataFrame):
                #print "Is dataframe"
            #    offset = - 2
            red_list.append(contour_data[key].iloc[:, idxfirstfeature: idxEndFeatures+1])
            lab_list.append(contour_data[key]['labels'])

        joined_data = join_contours(red_list)
        joined_labels = join_contours(lab_list)

    else:
        #if isinstance(contour_data,pd.DataFrame):
        #    offset = - 2
        joined_data = contour_data.iloc[:, idxfirstfeature:idxEndFeatures+1]
        joined_labels = contour_data['labels']

    features = np.array(joined_data)
    labels = np.array(joined_labels)

    return features, labels

