import json, os, requests
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup # to parse html
from operator import itemgetter
from pprint import pprint
import lyricsgenius
import pymongo
import csv


GENIUS_CLIENT_ACCESS_TOKEN = os.environ['GENIUS_CLIENT_ACCESS_TOKEN']

defaults = {
    'request': {
        'token': GENIUS_CLIENT_ACCESS_TOKEN,
        'base_url': 'https://api.genius.com'
    },
    'message': {
        'search_fail': 'The lyrics for this song were not found!',
        'wrong_input': 'Wrong number of arguments.\n' \
                       'Use two parameters to perform a custom search ' \
                       'or none to get the song currently playing on Spotify.'
    }
}

base_url = defaults['request']['base_url']
headers = {'Authorization': 'Bearer ' + defaults['request']['token']}

def get_artist_names(filename):
    # Import data
    with open(filename) as file:
        artist_names = [line.strip() for line in file]
    return artist_names

def get_artist_id(artist_name):
    search_url = base_url + '/search'
    data = {'q': artist_name}
    artist_data = requests.get(search_url, data=data, headers=headers).json()
    artist_id = artist_data['response']['hits'][0]['result']['primary_artist']['id']
    return artist_id

def request_top50_songs_data(artist_id):
    search_url = base_url + '/artists/' + str(artist_id) + '/songs'
    full_url = search_url + '?sort=popularity&per_page=50'
    top50_songs_data = requests.get(full_url, headers=headers).json()
    # top50_songs will be a list of the top 50 songs, each as json dicts of info
    top50_songs = top50_songs_data['response']['songs']
    return top50_songs

def make_song_dict(song, artist_name):
    song_id = song['id']
    song_title = song['title']
    full_lyrics = _get_full_lyrics(genius, song_id, song_title, artist_name)

    song_info_dict = {'song_id': song['id'],
                        'song_title': song['title'],
                        'full_title': song['full_title'],
                        'song_tate_cnt': song['annotation_count'],
                        'hot': song['stats']['hot'],
                        'pageviews': song['stats']['pageviews'],
                        'n_unreviewed_tates': song['stats']['unreviewed_annotations'],
                        'full_lyrics': full_lyrics}

    return song_info_dict, song_id, song_title

def _get_full_lyrics(genius, song_id, song_title, artist_name):
    song = genius.search_song(song_title, artist_name)
    lyrics = song.lyrics
    full_lyrics = lyrics.split('\n')
    for line in full_lyrics:
         if len(line) == 0:
             full_lyrics.remove(line)
    return full_lyrics

def populate_rt_dict(genius, song_id, rt_ids, rt_dict, song_info_dict, artist_name, artist_id):
    search_url = base_url + '/referents'
    data = {'song_id': song_id}
    referents_page = requests.get(search_url, data=data, headers=headers).json()
    # if referents_page.status_code == 404:
    #     return None

    for ref in referents_page['response']['referents']:
        rt_id = ref['id']
        rt_ids.append(rt_id)

        tate = genius.get_annotation(rt_id)

        tate_text = tate['annotation']['body']['plain']
        ref_text = tate['referent']['fragment']
        rt_dict[rt_id] = {'ref_text': ref_text,
                            'tate_text': tate_text,
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
                            'full_lyrics': song_info_dict['full_lyrics'],
                            'url': tate['annotation']['url'],
                            'n_tate_contributors': len(tate['annotation']['authors']),
                            'primary_contributor_id': tate['annotation']['authors'][0]['user']['id'],
                            'primary_contributor_IQ': tate['annotation']['authors'][0]['user']['iq'],
                            'has_voters': tate['annotation']['has_voters'],
                            'comment_cnt': tate['annotation']['comment_count'],
                            'artist_name': artist_name,
                            'artist_id': artist_id}

    return rt_dict

if __name__ == '__main__':
    # artists_filename = 'top20_genius_artists.txt'
    # artist_names = get_artist_names(artists_filename)
    # all-time top 20 artists on Genius

    artist_names = ['Travis Scott', 'Future', 'Frank Ocean', 'Nicki Minaj', 'Childish Gambino', 'Ed Sheeran', 'A$AP Rocky', 'Logic']

    genius = lyricsgenius.Genius(GENIUS_CLIENT_ACCESS_TOKEN, verbose=False, remove_section_headers=True)

    artist_id_dict = dict()
    song_id_dict = dict()
    rt_dict = dict()

    # artist_id to list of song_ids
    artist_song_dict = dict()

    rt_ids = []

    for artist_name in artist_names:
        print("Retrieving data for {0}...".format(artist_name))
        artist_id = get_artist_id(artist_name)

        if artist_id not in artist_id_dict.keys():
            artist_id_dict[artist_id] = artist_name

        top50_songs = request_top50_songs_data(artist_id)

        for song in top50_songs:
            song_id = song['id']
            # If artist isn't the primary artist, move on to next song
            if song['primary_artist']['id'] != artist_id:
                continue

            song_info_dict, song_id, song_title = make_song_dict(song, artist_name)

            if song_id not in song_id_dict.keys():
                song_id_dict[song_id] = song_title

            rt_dict = populate_rt_dict(genius, song_id, rt_ids, rt_dict, song_info_dict, artist_name, artist_id)

    cols = ['ref_text','tate_text', 'votes_total', 'verified', 'state', 'song_id', 'song_title', 'full_title', 'song_tate_cnt', 'hot_song', 'pageviews', 'n_unreviewed_tates', 'full_lyrics', 'url', 'n_tate_contributors', 'primary_contributor_id', 'primary_contributor_IQ', 'has_voters', 'comment_cnt', 'artist_name', 'artist_id']

    rt_df2 = pd.DataFrame.from_dict(rt_dict, orient='index', columns=cols)
    rt_df2.to_csv('rt_data_dump.csv')
