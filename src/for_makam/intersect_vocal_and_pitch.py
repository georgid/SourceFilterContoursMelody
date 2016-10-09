'''
Created on Oct 5, 2016

@author: georgid
'''

import json
import numpy as np

from compmusic import dunya
import os
import subprocess
import sys
dunya.set_token("69ed3d824c4c41f59f0bc853f696a7dd80707779")
import logging

def intersect_section_links(sections, pitch_series):    
    i = 0 # start at zero frame
    for sectionLink in sections:
            
            if 'VOCAL' not in sectionLink['name']:
              continue
            startTime = sectionLink['time'][0]
            
            endTime = sectionLink['time'][1]
            
            while pitch_series[i][0] < startTime : # make zero pitch
                pitch_series[i][1] = 0.0 
                i+=1
                
            while pitch_series[i][0] <= endTime and i< len(pitch_series) :
                i+=1
    return pitch_series

def getWork( musicbrainzid):
        rec_data = dunya.makam.get_recording(musicbrainzid)
        if len(rec_data['works']) == 0:
            raise Exception('No work on recording %s' % musicbrainzid)
        if len(rec_data['works']) > 1:
            raise Exception('More than one work for recording %s Not implemented!' % musicbrainzid)
        w = rec_data['works'][0]
        return w

def download_wav(musicbrainzid, outputDir):
        '''
        download wav for MB recording id from makam collection
        '''
        mp3FileURI = dunya.makam.download_mp3(musicbrainzid, outputDir)
        newName = os.path.join(os.path.abspath(os.path.dirname(mp3FileURI)), musicbrainzid + '.mp3')
        os.rename(mp3FileURI, newName )
    ###### mp3 to Wav: way 1
    #         newName = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test.mp3')
    #         os.rename(mp3FileURI, newName )
    #         mp3ToWav = Mp3ToWav()
    #         wavFileURI = mp3ToWav.run('dummyMBID', newName)
        
        ###### mp3 to Wav: way 2
        wavFileURI = os.path.splitext(newName)[0] + '.wav'
        if os.path.isfile(wavFileURI):
            return wavFileURI
            
        pipe = subprocess.Popen(['/usr/local/bin/ffmpeg', '-i', newName, wavFileURI])
        pipe.wait()
    
        return wavFileURI


# parse section links 
# sections_URI = '/home/georgid/Downloads/derivedfiles_ba1dc923-9b0e-4b6b-a306-346bd5438d35/ba1dc923-9b0e-4b6b-a306-346bd5438d35-jointanalysis-0.1-sections-1.json'

def compute_vocal_pitch(musicbrainzid):
    try:
        pitch_data = dunya.docserver.get_document_as_json(musicbrainzid, "jointanalysis", "pitch", 1, version="0.1")
        pitch_series = pitch_data['pitch']
    except:
        logging.error("no initialmakampitch series could be downloaded. for rec  {}".format(musicbrainzid))
        return None
    try:
        # sections = dunya.docserver.get_document_as_json(musicbrainzid, "scorealign", "sectionlinks", 1, version="0.2")
        sections_all_works = dunya.docserver.get_document_as_json(musicbrainzid, "jointanalysis", "sections", 1, version="0.1")
    except dunya.conn.HTTPError:
        logging.error("section link {} missing".format(musicbrainzid))
        return None
    
    work = getWork( musicbrainzid)
    sections = sections_all_works[work['mbid']]

    intersected_pitch_series =  intersect_section_links(sections, pitch_series)
    return intersected_pitch_series


def store_pitch_anno(musicbrainzid, intersected_pitch_series, pitch_dir):
    '''
    store intersected vocal pitch series in a given dir pitch_dir
    '''
    
    pitch_array = np.array(intersected_pitch_series) 
    f = open(os.path.join(pitch_dir, musicbrainzid + '.pv'), 'w')
    for time,pitch in pitch_array[:,0:2]:
        f.write('{},{}\n'.format(time,pitch))


