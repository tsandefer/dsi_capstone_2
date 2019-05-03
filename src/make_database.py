import pandas as pd
import numpy as np
import os, requests
import pymongo
# Used johnwmiller's LyricsGenius for annotation text
import lyricsgenius  # https://github.com/johnwmillr/LyricsGenius

GENIUS_CLIENT_ACCESS_TOKEN = os.environ['GENIUS_CLIENT_ACCESS_TOKEN']
defaults = {'request': {'token': GENIUS_CLIENT_ACCESS_TOKEN,
                        'base_url': 'https://api.genius.com'},
            'message': {'search_fail': 'The lyrics for this song were not found!',
                        'wrong_input': 'Wrong number of arguments.'}}
base_url = defaults['request']['base_url']
headers = {'Authorization': 'Bearer ' + defaults['request']['token']}

def get_artist_names(artist_names_file):
    '''INPUT: filename (str)
        OUTPUT: artist names (list of strings)'''
    with open(artist_names_file) as file:
        artist_names = [artist_name.strip() for artist_name in file]
    return artist_names

def get_artist_id(artist_name):
    '''INPUT: artist name (str)
        OUTPUT: artist id (int)'''
    search_url = base_url + '/search'
    data = {'q': artist_name}
    artist_data = requests.get(search_url, data=data, headers=headers).json()
    artist_id = artist_data['response']['hits'][0]['result']['primary_artist']['id']
    return artist_id

def request_top_songs_data(artist_id, n_songs, n_page):
    '''Obtains data on top 'n' songs (dict) from API
        INPUT: artist id (int)
                n_songs (int) - number of songs per page
                n_page (int) - page number to retrieve info from
        OUTPUT: top_songs (list of json dicts)'''
    search_url = base_url + f'/artists/{str(artist_id)}/songs?sort=popularity&per_page={n_songs}&page={n_page}'
    top_songs_data = requests.get(search_url, headers=headers).json()
    top_songs = top_songs_data['response']['songs'] # A list of the top 'n' songs, each as json dicts of info
    return top_songs

def make_song_dict(song):
    '''INPUT: song info (json dict) - from iterating through top_songs list
        OUTPUT: useful info for song (dict)'''
    song_info_dict = {'song_id': song['id'],
                        'song_title': song['title'],
                        'artist_id': song['primary_artist']['id'],
                        'full_title': song['full_title'],
                        'song_tate_cnt': song['annotation_count'],
                        'hot': song['stats']['hot'],
                        'pageviews': song['stats']['pageviews'],
                        'n_unreviewed_tates': song['stats']['unreviewed_annotations']}
    return song_info_dict

def gather_annotation_data(genius, song_info_dict, artist_name, annotations):
    '''INPUT: genius (instantiated class from LyricsGenius)
                song info dict (dict) - important info about song
                artist name (str)
                annotations (list of dicts for each annotation) - starts as empty list that we append to as we gather data
        OUTPUT: annotations (^) - returned once we append new data'''
    search_url = base_url + '/referents'
    data = {'song_id': song_info_dict['song_id']}
    referents_page = requests.get(search_url, data=data, headers=headers).json()
    for ref in referents_page['response']['referents']:
        rt_id = ref['id']
        tate = genius.get_annotation(rt_id)
        annotation_info = {'rt_id': rt_id,
                            'ref_text': tate['referent']['fragment'],
                            'tate_text': tate['annotation']['body']['plain'],
                            'votes_total': tate['annotation']['votes_total'],
                            'verified': tate['annotation']['verified'],
                            'state': tate['annotation']['state'],
                            'song_id': song_info_dict['song_id'],
                            'song_title': song_info_dict['song_title'],
                            'full_title': song_info_dict['full_title'],
                            'song_tate_cnt': song_info_dict['song_tate_cnt'],
                            'hot_song': song_info_dict['hot'],
                            'pageviews': song_info_dict['pageviews'],
                            'n_unreviewed_tates': song_info_dict['n_unreviewed_tates'],
                            'url': tate['annotation']['url'],
                            'n_tate_contributors': len(tate['annotation']['authors']),
                            'primary_contributor_id': tate['annotation']['authors'][0]['user']['id'],
                            'primary_contributor_IQ': tate['annotation']['authors'][0]['user']['iq'],
                            'has_voters': tate['annotation']['has_voters'],
                            'comment_cnt': tate['annotation']['comment_count'],
                            'artist_name': artist_name,
                            'artist_id': song_info_dict['artist_id']}
        annotations.append(annotation_info)
    return annotations

def populate_mongoDB(data):
    '''Uses data (dict) obtained from API to populate MongoDB for later retrieval/use
        ----> db = genius, collection = lyric_annotations
        INPUT: data (list of dictionaries)
        OUTPUT: NONE - data is stored in MongoDB'''
    print(f"The dictionary has {len(data)} annotations")
    client = pymongo.MongoClient()
    db =  client.genius # create new db
    coll = db.lyric_annotations # create new collection
    try:
        coll.insert_many(data) # insert the data
    except pymongo.errors.DuplicateKeyError:
        pass
    print(f"The database has {coll.estimated_document_count()} documents.")

def main(n_songs=50, n_page=1, sleeptime=1, verbose=True):
    n_songs = n_songs
    n_page = n_page

    # Top 20 Artists on Genius
    artist_names_file = '../data/artists.txt'
    artist_names = get_artist_names(artist_names_file)

    genius = lyricsgenius.Genius(GENIUS_CLIENT_ACCESS_TOKEN, sleep_time=sleeptime,
                                        verbose=False, remove_section_headers=True)
    annotations = []
    for artist_name in artist_names:
        if verbose:
            print(f"Retrieving data for {artist_name}...")
        artist_id = get_artist_id(artist_name)
        top_songs = request_top_songs_data(artist_id, n_songs, n_page)
        for song in top_songs:
            song_id = song['id']
            if song['primary_artist']['id'] != artist_id: # Move on if not primary artist
                continue # Prevents duplication and attribution issues (ex: Beyonce & JAY-Z)
            song_info_dict = make_song_dict(song)
            annotations = gather_annotation_data(genius, song_info_dict,
                                                    artist_name, annotations)
    populate_mongoDB(annotations)

if __name__ == '__main__':
    # test
    main(n_songs=1, n_page=1, sleeptime=0, verbose=True)
