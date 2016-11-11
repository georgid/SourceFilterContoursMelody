'''
Created on Oct 5, 2016

@author: georgid
'''

list_long_SAZ = {'cab08727-d5c2-4fda-9d96-d107915a85ec'}
list_erroneous_vocal = {'764c2b10-90b8-44b8-8254-2992302f8a2b'}
list_other_problems = {'67e7422b-24d8-420a-961f-652e6cf66ca9'}
list_not_correct_lyric_linking = {'6fdc4617-e491-44b2-998c-3bc00bc2085', 
                                  
                                  }



import json
import compmusic.dunya.makam

DATA_PATH = '/home/georgid/Documents/otmm_corpus_stats/'

def get_symbTr_works():
    symbTr_work_MBIDs = []
    
    symbTRURI = DATA_PATH + '/data/SymbTrWork.json'
    with open(symbTRURI) as f:
        symbTrs_meta = json.load(f)
        for symbTr in symbTrs_meta:
            uuid = symbTr['uuid']
            mbid = uuid.split('/')[-1]
            symbTr_work_MBIDs.append(mbid)
    return symbTr_work_MBIDs

def get_sarki_works():
    token = '69ed3d824c4c41f59f0bc853f696a7dd80707779'
    compmusic.dunya.conn.set_token(token)
    
    work_sarkis_mbids = []
    works_sarkis = compmusic.dunya.makam.get_form('20edad19-60a7-49cf-a349-266b80af7e8d')['works']
    for elem in works_sarkis:
        work_mbid = elem['mbid']
        work_sarkis_mbids.append(work_mbid)
    return work_sarkis_mbids


def get_vocal_recIDs():
    vocals_URI = DATA_PATH + '/data/InstrumentVoiceMetadata/vocal.json'
    vocal_recording_MBIDs = []
    with open(vocals_URI) as f:
            vocal_recIDs = json.load(f)['performances']
            for vocal_recording in vocal_recIDs:
                vocal_recording_MBIDs.append(str(vocal_recording['mbid']))
    return vocal_recording_MBIDs


def intersect_vocal_sarki_symbTr():
    # intersect 3-lists
    symbTr_work_MBIDs = get_symbTr_works()
    work_sarkis_mbids = get_sarki_works()
    vocal_recording_MBIDs = get_vocal_recIDs()
    
    
    count = 0
    sarki_vocal_rec_ids = []
    
    for symbTr_work_MBID in symbTr_work_MBIDs:
        if symbTr_work_MBID not in work_sarkis_mbids: # only sarki works
            continue
    
    #     count += 1
    # print count
        for rec in compmusic.dunya.makam.get_work(symbTr_work_MBID)['recordings']:
            rec_MBID = rec['mbid']
            if rec_MBID not in vocal_recording_MBIDs: # only recordings for sarki with vocal id 
                continue
            count += 1
    
            print count
            sarki_vocal_rec_ids.append(rec_MBID)
        
        if len(sarki_vocal_rec_ids) > 30:
            break    
        
    return sarki_vocal_rec_ids

def get_recIDs_OK(sarki_vocal_rec_ids):
    '''
    get recordings OK for vocal/non-vocal detection
    after listening test and manually created lists
    '''
    recs_OK = []
    count = 0
    for rec_ID in sarki_vocal_rec_ids:
            if rec_ID not in list_long_SAZ and \
            rec_ID not in list_erroneous_vocal and \
            rec_ID not in list_other_problems:
#                 print 'http://dunya.compmusic.upf.edu/makam/recording/' + str(rec_ID)
                recs_OK.append(rec_ID)
                count+= 1
    return recs_OK


if __name__ == '__main__':
        sarki_vocal_rec_ids = intersect_vocal_sarki_symbTr()
        for idx, sarki_rec_id in enumerate(sarki_vocal_rec_ids):
            print idx
            print 'http://dunya.compmusic.upf.edu/makam/recording/' + str(sarki_rec_id)
