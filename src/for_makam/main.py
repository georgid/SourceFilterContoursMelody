'''
Created on Oct 7, 2016

@author: georgid
'''
import sys
from intersect_vocal_and_pitch import download_wav,\
    compute_vocal_pitch, store_pitch_anno
import numpy
from get_vocal_recordings import intersect_vocal_sarki_symbTr,\
    get_recIDs_OK

if __name__ == '__main__':
    
    musicbrainzid = 'ba1dc923-9b0e-4b6b-a306-346bd5438d35'
    audioDir = sys.argv[1]
    pitch_dir = sys.argv[2]
    
    sarki_vocal_rec_ids = intersect_vocal_sarki_symbTr()
    recs_OK = get_recIDs_OK(sarki_vocal_rec_ids)
    
    for rec_ID in recs_OK:
        audio_URI = download_wav(musicbrainzid, audioDir)
        print type(rec_ID)
        intersected_pitch_series = compute_vocal_pitch(rec_ID)
        if intersected_pitch_series == None:
            continue
        store_pitch_anno(rec_ID, intersected_pitch_series,  pitch_dir)
    
    

    
    ### plot
    import matplotlib.pyplot as plt
    
    pitch_array = numpy.array(intersected_pitch_series)
    plt.plot(pitch_array[:,0], pitch_array[:,1])
    plt.show()