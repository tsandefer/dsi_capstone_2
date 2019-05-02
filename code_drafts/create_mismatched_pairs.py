import pandas as pd
import numpy as np

import pickle
import re

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

from segment_data import save_in_pkl


# pd.options.mode.chained_assignment = None  # default='warn'

def read_in_pkl(name):
    filename = '../data/' + name + '.pkl'
    return pickle.load(open(filename, "rb" ))

def get_true_rt_pairs(train_df):
    pair_df = train_df[['rt_id', 'artist_name']]

    pair_df['ref_id'] = pair_df['rt_id'].copy()
    pair_df['tate_id'] = pair_df['rt_id'].copy()

    pair_df['ref_artist'] = pair_df['artist_name'].copy()
    pair_df['tate_artist'] = pair_df['artist_name'].copy()

    pair_df.drop(['rt_id', 'artist_name'], axis=1, inplace=True)

    return pair_df

def get_mismatched_rt_pairs(pair_df, rt_artist_dict):
    mixed_pair_df = pair_df[['ref_id', 'tate_id', 'ref_artist', 'tate_artist']].copy()

    ref_ids = np.array([mixed_pair_df['ref_id']]).reshape(-1,)
    tate_ids = np.array([mixed_pair_df['tate_id']]).reshape(-1,)

    shuffled_ref_rt, shuffled_tate_rt = shuffle_rtids(ref_ids, tate_ids)
    mism_ref_rt = np.array([])
    mism_tate_rt = np.array([])
    cnt = 0
    tp_net = 0
    while shuffled_ref_rt.shape[0] > 0:
        mism_ref_rt, mism_tate_rt, shuffled_ref_rt, shuffled_tate_rt = separate_same_artist_rtids(shuffled_ref_rt, shuffled_tate_rt, rt_artist_dict, mism_tate_rt, mism_ref_rt)
        cnt += 1
        if cnt > 2000:
            is_true_pair = shuffled_ref_rt == shuffled_tate_rt
            n_true_pairs = is_true_pair.sum()

            # get their song ids -- are they from same?
            shuff_ref_song = np.array([rt_song_dict[rtid] for rtid in shuffled_ref_rt])
            shuff_tate_song = np.array([rt_song_dict[rtid] for rtid in shuffled_tate_rt])

            same_song = shuff_tate_song == shuff_ref_song
            n_same_songs = same_song.sum()

            # If no true pairs or has run through
            if n_true_pairs < 1 and tp_net > 5:
                print("Heads up, the last {0} pairs were not well-scrambled.. There are {1} true pairs among them.".format(shuffled_ref_rt.shape[0], n_true_pairs))
                mism_ref_rt = np.concatenate([mism_ref_rt, shuffled_ref_rt])
                mism_tate_rt = np.concatenate([mism_tate_rt, shuffled_tate_rt])
                break

            cnt = 0
            tp_net += 1
    return mism_ref_rt, mism_tate_rt

def shuffle_rtids(ref_ids, tate_ids):
    n = ref_ids.shape[0]
    shuffled_ref_rt = np.random.choice(ref_ids, n, replace=False)
    shuffled_tate_rt = np.random.choice(tate_ids, n, replace=False)
    return shuffled_ref_rt, shuffled_tate_rt

def separate_same_artist_rtids(shuffled_ref_rt, shuffled_tate_rt, rt_artist_dict, mism_tate_rt, mism_ref_rt):
    shuff_ref_artist = np.array([rt_artist_dict[rtid] for rtid in shuffled_ref_rt])
    shuff_tate_artist = np.array([rt_artist_dict[rtid] for rtid in shuffled_tate_rt])

    diff_artist = shuff_tate_artist != shuff_ref_artist
    mism_tate_rt = mism_tate_rt
    mism_ref_rt = mism_ref_rt
    mism_ref_rt = np.concatenate([mism_ref_rt, shuffled_ref_rt[diff_artist]])
    mism_tate_rt = np.concatenate([mism_tate_rt, shuffled_tate_rt[diff_artist]])

    shuffled_ref_rt = shuffled_ref_rt[~diff_artist]
    shuffled_tate_rt = shuffled_tate_rt[~diff_artist]

    return mism_ref_rt, mism_tate_rt, shuffled_ref_rt, shuffled_tate_rt

def calc_number_of_same_song_pairs(shuffled_ref_rt, shuffled_tate_rt, rt_song_dict):
    shuff_ref_song = np.array([rt_song_dict[rtid] for rtid in shuffled_ref_rt])
    shuff_tate_song = np.array([rt_song_dict[rtid] for rtid in shuffled_tate_rt])

    same_song = shuff_tate_song == shuff_ref_song
    n_same_songs = same_song.sum()
    return n_same_songs

def get_rt_artist(shuff_df):
    shuff_ref_artist = np.array([rt_artist_dict[rtid] for rtid in shuff_df['ref_id']])
    shuff_tate_artist = np.array([rt_artist_dict[rtid] for rtid in shuff_df['tate_id']])
    shuff_df['ref_artist'] = shuff_ref_artist
    shuff_df['tate_artist'] = shuff_tate_artist
    return shuff_df

def get_rt_in_train(shuff_df):
    shuff_ref_in_train = np.array([rt_in_train_dict[rtid] for rtid in shuff_df['ref_id']])
    shuff_tate_in_train = np.array([rt_in_train_dict[rtid] for rtid in shuff_df['tate_id']])

    shuff_df['ref_in_train'] = shuff_ref_in_train
    shuff_df['tate_in_train'] = shuff_tate_in_train
    return shuff_df

def get_rt_doc_ids(shuff_df):
    shuff_ref_doc_id = []
    for rtid in shuff_df['ref_id']:
        at_rtid = shuff_df['ref_id'] == rtid
        if shuff_df[at_rtid]['ref_in_train'].bool():
            doc_id = rt_doc_train_dict[rtid]
        else:
            doc_id = rt_doc_test_dict[rtid]
        shuff_ref_doc_id.append(doc_id)
    shuff_tate_doc_id = []
    for rtid in shuff_df['tate_id']:
        at_rtid = shuff_df['tate_id'] == rtid
        if shuff_df[at_rtid]['ref_in_train'].bool():
            doc_id = rt_doc_train_dict[rtid]
        else:
            doc_id = rt_doc_test_dict[rtid]
        shuff_tate_doc_id.append(doc_id)

    shuff_df['ref_doc_id'] = shuff_ref_doc_id
    shuff_df['tate_doc_id'] = shuff_tate_doc_id
    return shuff_df

def get_rt_raw_txt(shuff_df):
    shuff_ref_raw_txt = []
    for doc_id in shuff_df['ref_doc_id']:
        at_doc_id = shuff_df['ref_doc_id'] == doc_id
        if shuff_df[at_doc_id]['ref_in_train'].bool():
            raw_txt = corpus_dict['referent'][0][doc_id]
        else:
            raw_txt = corpus_dict['referent'][1][doc_id]
        shuff_ref_raw_txt.append(raw_txt)

    shuff_tate_raw_txt = []
    for doc_id in shuff_df['tate_doc_id']:
        at_doc_id = shuff_df['tate_doc_id'] == doc_id
        if shuff_df[at_doc_id]['ref_in_train'].bool():
            raw_txt = corpus_dict['tate'][0][doc_id]
        else:
            raw_txt = corpus_dict['tate'][1][doc_id]
        shuff_tate_raw_txt.append(raw_txt)

    shuff_df['ref_raw_text'] = shuff_ref_raw_txt
    shuff_df['tate_raw_text'] = shuff_tate_raw_txt
    return shuff_df

def get_rt_pp_txt(shuff_df):
    shuff_ref_pp_txt = []
    for doc_id in shuff_df['ref_doc_id']:
        at_doc_id = shuff_df['ref_doc_id'] == doc_id
        if shuff_df[at_doc_id]['ref_in_train'].bool():
            pp_txt = ref_train_pcorpus[doc_id].words
        else:
            pp_txt = ref_test_pcorpus[doc_id]
        shuff_ref_pp_txt.append(pp_txt)
    shuff_tate_pp_txt = []
    for doc_id in shuff_df['tate_doc_id']:
        at_doc_id = shuff_df['tate_doc_id'] == doc_id
        if shuff_df[at_doc_id]['ref_in_train'].bool():
            pp_txt = tate_train_pcorpus[doc_id].words
        else:
            pp_txt = tate_test_pcorpus[doc_id]
        shuff_tate_pp_txt.append(pp_txt)

    shuff_df['ref_pp_text'] = shuff_ref_pp_txt
    shuff_df['tate_pp_text'] = shuff_tate_pp_txt
    return shuff_df

# def get_shuffled_id_corpuses(shuff_df):
#     shuff_ref_corpus = list(shuff_df['ref_pp_text'])
#     shuff_tate_corpus = list(shuff_df['tate_pp_text'])
#     return shuff_ref_corpus, shuff_tate_corpus

def add_info_cols_to_df(rt_df):
    rt_df = get_rt_in_train(rt_df)
    rt_df = get_rt_doc_ids(rt_df)
    rt_df = get_rt_raw_txt(rt_df)
    rt_df = get_rt_pp_txt(rt_df)
    return rt_df

def make_true_pairs_df(test_or_train_df):
    pair_df = get_true_rt_pairs(test_or_train_df)
    pair_df = add_info_cols_to_df(pair_df)
    pair_df['is_pair'] = [1 for _ in range(pair_df.shape[0])]
    pair_df['same_artist'] = [1 for _ in range(pair_df.shape[0])]
    pair_df.reset_index(drop=True, inplace=True)
    return pair_df

def make_false_pairs_df(test_or_train_pair_df):
    shuff_ref_rt, shuff_tate_rt = get_mismatched_rt_pairs(test_or_train_pair_df, rt_artist_dict)
    shuff_df = pd.DataFrame({'ref_id': shuff_ref_rt, 'tate_id': shuff_tate_rt})
    shuff_df = get_rt_artist(shuff_df)
    shuff_df = add_info_cols_to_df(shuff_df)
    shuff_df['is_pair'] = (shuff_df['ref_id']==shuff_df['tate_id']).astype(int)
    shuff_df['same_artist'] = (shuff_df['ref_artist']==shuff_df['tate_artist']).astype(int)
    return shuff_df

def make_full_df(pair_df, shuff_df):
    all_pairs_df = pd.concat([pair_df, shuff_df], axis=0, ignore_index=True)
    return all_pairs_df

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None  # default='warn'

    # data = read_in_pkl('training_data')
    corpus_dict = read_in_pkl('corpus_dict')
    train_df = read_in_pkl('train_df')
    test_df = read_in_pkl('test_df')
    lookup_dicts = read_in_pkl('lookup_dicts')
    pcorpuses = read_in_pkl('pcorpuses')

    artists = lookup_dicts[0]
    # doc_rt_train_dict = lookup_dicts[1]
    # doc_rt_test_dict = lookup_dicts[2]
    rt_doc_train_dict = lookup_dicts[3]
    rt_doc_test_dict = lookup_dicts[4]

    rt_in_train_dict = lookup_dicts[5]
    # artist_rt_dict = lookup_dicts[6]

    rt_artist_dict = lookup_dicts[7]

    # song_rt_dict = lookup_dicts[8]
    rt_song_dict = lookup_dicts[9]
    # song_id_to_title_dict = lookup_dicts[10]


    ref_train_pcorpus = pcorpuses[0]
    ref_test_pcorpus = pcorpuses[1]
    tate_train_pcorpus = pcorpuses[2]
    tate_test_pcorpus = pcorpuses[3]
    # rt_train_pcorpus = pcorpuses[4]
    # rt_test_pcorpus = pcorpuses[5]
    # rt_tagged_train_pcorpus = pcorpuses[6]
    # rt_tagged_test_pcorpus = pcorpuses[7]

    train_true_pair_df = make_true_pairs_df(train_df)
    train_false_pair_df = make_false_pairs_df(train_true_pair_df)
    train_all_pairings_df = make_full_df(train_true_pair_df, train_false_pair_df)

    test_true_pair_df = make_true_pairs_df(test_df)
    test_false_pair_df = make_false_pairs_df(test_true_pair_df)
    test_all_pairings_df = make_full_df(test_true_pair_df, test_false_pair_df)

    print("\nTrain, True")
    print(train_true_pair_df.shape)
    print(train_true_pair_df['is_pair'].sum())
    print(train_true_pair_df['same_artist'].sum())

    print("\nTrain, False")
    print(train_false_pair_df.shape)
    print(train_false_pair_df['is_pair'].sum())
    print(train_false_pair_df['same_artist'].sum())

    print("\nTrain, All")
    print(train_all_pairings_df.shape)
    print(train_all_pairings_df['is_pair'].sum())
    print(train_all_pairings_df['same_artist'].sum())

    print("\nTest, True")
    print(test_true_pair_df.shape)
    print(test_true_pair_df['is_pair'].sum())
    print(test_true_pair_df['same_artist'].sum())

    print("\nTest, False")
    print(test_false_pair_df.shape)
    print(test_false_pair_df['is_pair'].sum())
    print(test_false_pair_df['same_artist'].sum())

    print("\nTest, All")
    print(test_all_pairings_df.shape)
    print(test_all_pairings_df['is_pair'].sum())
    print(test_all_pairings_df['same_artist'].sum())


    # pair_df = get_true_rt_pairs(train_df)
    # pair_df = add_info_cols_to_df(pair_df)
    # pair_df['is_pair'] = [1 for _ in range(pair_df.shape[0])]
    # pair_df['same_artist'] = [1 for _ in range(pair_df.shape[0])]
    # pair_df.reset_index(drop=True, inplace=True)
    #
    #
    # shuff_ref_rt, shuff_tate_rt = get_mismatched_rt_pairs(pair_df, rt_artist_dict)
    # shuff_df = pd.DataFrame({'ref_id': shuff_ref_rt, 'tate_id': shuff_tate_rt})
    # shuff_df = get_rt_artist(shuff_df)
    # shuff_df = add_info_cols_to_df(shuff_df)
    # shuff_df['is_pair'] = (shuff_df['ref_id']==shuff_df['tate_id']).astype(int)
    # shuff_df['same_artist'] = (shuff_df['ref_artist']==shuff_df['tate_artist']).astype(int)
    #
    # shuff_df = get_rt_in_train(shuff_df)
    # shuff_df = get_rt_doc_ids(shuff_df)
    # shuff_df = get_rt_raw_txt(shuff_df)
    # shuff_df = get_rt_pp_txt(shuff_df)

    # shuff_ref_corpus, shuff_tate_corpus = get_shuffled_id_corpuses(shuff_df)


    # print(shuff_df['is_pair'].sum())
    # print(shuff_df['same_artist'].sum())



    # all_pairs_df = pd.concat([pair_df, shuff_df], axis=0, ignore_index=True)

    # save_in_pkl(pair_df, 'true_pairs_df')
    # save_in_pkl(shuff_df, 'false_pairs_df')
    # save_in_pkl(all_pairs_df, 'full_pairs_df')
