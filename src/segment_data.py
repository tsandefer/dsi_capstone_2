import pandas as pd
import numpy as np

import re
import pickle

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import spacy
from sklearn.model_selection import train_test_split
import gensim
import os
import collections
import smart_open
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine


pd.options.mode.chained_assignment = None  # default='warn'

def make_train_test_split(df, get_holdout=False):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    if get_holdout:
        holdout_df = test_df.copy()
        train_df.to_csv('../data/genius_train_data.csv')
        holdout_df.to_csv('../data/genius_holdout_data.csv')
        return train_df
    else:
        return train_df, test_df

def save_in_pkl(data_obj, name):
    filename = '../data/' + name + '.pkl'
    pickle.dump(data_obj, open(filename, "wb" ))

def get_ref_tate_dfs(df):
    ref_df = df[['ref_text', 'rt_id']]
    tate_df = df[['tate_text', 'rt_id']]
    ref_df.reset_index(drop=True, inplace=True)
    tate_df.reset_index(drop=True, inplace=True)
    # save_in_pkl(data_obj, name)
    # save_in_pkl(data_obj, name)
    return ref_df, tate_df

def make_combined_rt_df(ref_df, tate_df, is_train=True):
    ref_df.columns = ['text', 'rt_id']
    tate_df.columns = ['text', 'rt_id']
    ref_df['is_ref'] = 1
    tate_df['is_ref'] = 0
    rt_df = pd.concat([ref_df, tate_df], axis=0)
    rt_df.reset_index(drop=True, inplace=True)
    return rt_df

def isolate_corpuses(txt_train_df, txt_test_df):
    text_train = txt_train_df['text']
    text_test = txt_test_df['text']
    return text_train, text_test

def make_corpus_dicts(ref_train_df, ref_test_df, tate_train_df, tate_test_df, rt_train_df, rt_test_df):
    txt_dfs = [(ref_train_df, ref_test_df), (tate_train_df, tate_test_df), (rt_train_df, rt_test_df)]
    corpus_cats = ['referent', 'tate', 'ref-tate']
    corpus_dict = dict()
    for idx, txt_type in enumerate(txt_dfs):
        corpus_type = corpus_cats[idx]
        txt_train_df, txt_test_df = txt_type
        text_train, text_test = isolate_corpuses(txt_train_df, txt_test_df)
        corpus_dict[corpus_type] = (text_train, text_test)
    save_in_pkl(corpus_dict, 'corpus_dict')
    return corpus_dict

def make_rt_doc_dicts(doc_train_df, doc_test_df):
    doc_rt_train = doc_train_df['rt_id']
    doc_rt_test = doc_test_df['rt_id']

    doc_rt_train_dict = doc_rt_train.to_dict()
    doc_rt_test_dict = doc_rt_test.to_dict()
    # save_in_pkl(doc_rt_train_dict, 'doc_rt_train_dict')
    # save_in_pkl(doc_rt_test_dict, 'doc_rt_test_dict')
    return doc_rt_train_dict, doc_rt_test_dict

def make_doc_rt_dic(doc_rt_train_dict, doc_rt_test_dict):
    rt_doc_train_dict = dict((v,k) for k,v in doc_rt_train_dict.items())
    rt_doc_test_dict = dict((v,k) for k,v in doc_rt_test_dict.items())
    # save_in_pkl(rt_doc_train_dict, 'rt_doc_train_dict')
    # save_in_pkl(rt_doc_test_dict, 'rt_doc_test_dict')
    return rt_doc_train_dict, rt_doc_test_dict

def rt_in_train_dic(data, train_df):
    # need to make a dict that will tell us if an rt_id is in train or test set
    rt_in_training_dict = dict()
    for rt_id in data['rt_id'].unique():
        if rt_id in list(train_df['rt_id']):
            rt_in_training_dict[rt_id] = True
        else:
            rt_in_training_dict[rt_id] = False
    # save_in_pkl(rt_in_training_dict, 'rt_in_training_dict')
    return rt_in_training_dict

def artist_rt_dic(data):
    gb_artist = data.groupby('artist_name')['rt_id'].unique()
    artist_rt_dict = dict()
    for artist_id in gb_artist.index:
        artist_rt_dict[artist_id] = gb_artist[artist_id]
    # save_in_pkl(artist_rt_dict, 'artist_rt_dict')
    return artist_rt_dict

def rt_artist_dic(artist_rt_dict):
    rt_artist_dict = dict()
    for k,v in artist_rt_dict.items():
        for i in v:
            rt_artist_dict[i] = k
    # save_in_pkl(rt_artist_dict, 'rt_artist_dict')
    return rt_artist_dict

def song_rt_dic(data):
    gb_song_id = data.groupby('song_id')['rt_id'].unique()
    song_rt_dict = dict()
    for song_id in gb_song_id.index:
        song_rt_dict[song_id] = gb_song_id[song_id]
    return song_rt_dict

def rt_song_dic(song_rt_dict):
    rt_song_dict = dict()
    for k,v in song_rt_dict.items():
        for i in v:
            rt_song_dict[i] = k
    # save_in_pkl(rt_artist_dict, 'rt_artist_dict')
    return rt_song_dict

def song_id_title_dic(data):
    by_song_title = data.groupby('song_id')['song_title'].unique()
    song_id_to_title_dict = dict()
    for song_id in by_song_title.index:
        song_id_to_title_dict[song_id] = by_song_title[song_id][0]
    return song_id_to_title_dict

def print_annotations_per_artist(artist_rtid_dict):
    print("ANNOTATIONS PER ARTIST:\n")
    for k, v in artist_rt_dict.items():
        print(k + ': ' + str(len(v)))

def get_token_lst(corpus, tagged_docs=False, include_ref_tag=False):
    # corpus = list(corpus)
    for idx, line in enumerate(corpus):
        # Returns list of strings where contraction words, newlines, and punctuation is preserved
        tokens = re.findall(r"[\w'|\w’]+|[-–()\"\“\”.,!?;]+|[\n]+", line)
        tokens = [x.lower() for x in tokens]
        if tagged_docs:
            is_tate = 0 if idx < 2033 else 1
            tag_lst = [idx, is_tate] if include_ref_tag else [idx]
            yield gensim.models.doc2vec.TaggedDocument(tokens, tag_lst)
        else:
            yield tokens

def get_train_test_corpuses(txt_series_train, txt_series_test, include_ref_tag=False):
    train_corpus = list(get_token_lst(txt_series_train, tagged_docs=True, include_ref_tag=include_ref_tag))
    test_corpus = list(get_token_lst(txt_series_test))
    return train_corpus, test_corpus

if __name__ == '__main__':
    df = pd.read_csv('../data/preprocessed_data.csv')

    data = make_train_test_split(df, get_holdout=True)
    save_in_pkl(data, 'training_data')

    artists = data['artist_name'].unique()

    train_df, test_df = make_train_test_split(data, get_holdout=False)
    save_in_pkl(train_df, 'train_df')
    save_in_pkl(test_df, 'test_df')

    ref_train_df, tate_train_df = get_ref_tate_dfs(train_df)
    ref_test_df, tate_test_df = get_ref_tate_dfs(test_df)

    rt_train_df = make_combined_rt_df(ref_train_df, tate_train_df, is_train=True)
    rt_test_df = make_combined_rt_df(ref_test_df, tate_test_df, is_train=False)

    corpus_dict = make_corpus_dicts(ref_train_df, ref_test_df, tate_train_df, tate_test_df, rt_train_df, rt_test_df)

    doc_rt_train_dict, doc_rt_test_dict = make_rt_doc_dicts(ref_train_df, ref_test_df)
    # full_train_dict, full_test_dict = make_rt_doc_dicts(rt_train_df, rt_test_df)

    # artists = data['artist_name'].unique()

    rt_doc_train_dict, rt_doc_test_dict = make_doc_rt_dic(doc_rt_train_dict, doc_rt_test_dict)
    rt_in_train_dict = rt_in_train_dic(data, train_df)
    artist_rt_dict = artist_rt_dic(data)
    rt_artist_dict = rt_artist_dic(artist_rt_dict)

    song_rt_dict = song_rt_dic(data)
    rt_song_dict = rt_song_dic(song_rt_dict)
    song_id_to_title_dict = song_id_title_dic(data)

    # dict_names = ['corpus_dict', 'doc_rt_train_dict', 'doc_rt_test_dict', 'rt_doc_train_dict', 'rt_doc_test_dict', 'rt_in_train_dict', 'artist_rt_dict', 'rt_artist_dict']
    # save_in_pkl(dict_names, 'dict_names')

    lookup_dicts = [artists, doc_rt_train_dict, doc_rt_test_dict, rt_doc_train_dict, rt_doc_test_dict, rt_in_train_dict, artist_rt_dict, rt_artist_dict, song_rt_dict, rt_song_dict, song_id_to_title_dict]
    save_in_pkl(lookup_dicts, 'lookup_dicts')
    # print_annotations_per_artist(artist_rt_dict)

    ref_train_pcorpus, ref_test_pcorpus = list(get_train_test_corpuses(corpus_dict['referent'][0], corpus_dict['referent'][1]))
    tate_train_pcorpus, tate_test_pcorpus = list(get_train_test_corpuses(corpus_dict['tate'][0], corpus_dict['tate'][1]))
    rt_train_pcorpus, rt_test_pcorpus = list(get_train_test_corpuses(corpus_dict['ref-tate'][0], corpus_dict['ref-tate'][1]))
    rt_tagged_train_pcorpus, rt_tagged_test_pcorpus = list(get_train_test_corpuses(corpus_dict['ref-tate'][0], corpus_dict['ref-tate'][1], include_ref_tag=True))

    # pcorpus_names = ['ref_tr', 'ref_tst', 'tate_tr', 'tate_tst', 'rt_tr', 'rt_tst', 'rt_tagged_tr', 'rt_tagged_tst']
    pcorpuses = [ref_train_pcorpus, ref_test_pcorpus, tate_train_pcorpus, tate_test_pcorpus, rt_train_pcorpus, rt_test_pcorpus, rt_tagged_train_pcorpus, rt_tagged_test_pcorpus]

    save_in_pkl(pcorpuses, 'pcorpuses')

    # save_in_pkl(pcorpus_names, 'pcorpus_names')


    # for idx in range(len(pcorpus_names)):
    #     save_in_pkl(pcorpuses[idx], pcorpus_names[idx])
