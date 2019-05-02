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

def read_in_pkl(name):
    filename = '../data/' + name + '.pkl'
    return pickle.load(open(filename, "rb" ))

def get_true_rt_pairs(train_or_test_df):
    pair_df = train_or_test_df[['rt_id', 'artist_name']]

    pair_df['ref_id'] = pair_df['rt_id'].copy()
    pair_df['tate_id'] = pair_df['rt_id'].copy()

    pair_df['ref_artist'] = pair_df['artist_name'].copy()
    pair_df['tate_artist'] = pair_df['artist_name'].copy()

    pair_df.drop(['rt_id', 'artist_name'], axis=1, inplace=True)
    return pair_df

def get_mismatched_rt_pairs(train_or_test_pair_df, pop_artists, hh_artists):
    ref_ids = np.array([train_or_test_pair_df['ref_id'].copy()]).reshape(-1,)
    tate_ids = np.array([train_or_test_pair_df['tate_id'].copy()]).reshape(-1,)

    pop_refs, hh_refs, pop_tates, hh_tates = get_pop_hh_ref_tate_arrays(ref_ids, tate_ids, pop_artists, hh_artists)

    for arr in [pop_refs, hh_refs, pop_tates, hh_tates]:
        np.random.shuffle(arr)

    n_pop = pop_refs.shape[0]
    n_hiphop = hh_refs.shape[0]

    if n_pop != n_hiphop:
        n_spillover = abs(n_pop - n_hiphop)
        more_pop = np.argmax([n_pop, n_hiphop]) == 0

        spill_group_refs = pop_refs if more_pop else hh_refs
        spill_group_tates = pop_tates if more_pop else hh_tates

        full_group_refs = hh_refs if more_pop else pop_refs
        full_group_tates = hh_tates if more_pop else pop_tates

        spill_artist_refs = np.array([rt_artist_dict[rtid] for rtid in spill_group_refs])
        spill_artist_tates = np.array([rt_artist_dict[rtid] for rtid in spill_group_tates])

        biggest_artist = 'Original Broadway Cast of Hamilton' if more_pop else 'Eminem'

        is_big_artist_r = spill_artist_refs == biggest_artist
        is_big_artist_t = spill_artist_tates == biggest_artist

        spilled_refs = spill_group_refs[is_big_artist_r][:n_spillover]
        spilled_tates = spill_group_tates[~is_big_artist_t][:n_spillover]

        is_pulled_ref = np.isin(spill_group_refs, spilled_refs)
        is_pulled_tate = np.isin(spill_group_tates, spilled_tates)

        other_refs = spill_group_refs[~is_pulled_ref]
        other_tates = spill_group_tates[~is_pulled_tate]

        shuffled_refs = np.concatenate([spilled_refs, full_group_refs, other_refs])
        shuffled_tates = np.concatenate([spilled_tates, other_tates, full_group_tates])
        return shuffled_refs, shuffled_tates
    else:
        shuffled_refs = np.concatenate([pop_refs, hh_refs])
        shuffled_tates = np.concatenate([hh_tates, pop_tates])
        return shuffled_refs, shuffled_tates


def get_pop_hh_ref_tate_arrays(ref_ids, tate_ids, pop_artists, hh_artists):
    ref_artist = np.array([rt_artist_dict[rtid] for rtid in ref_ids])
    tate_artist = np.array([rt_artist_dict[rtid] for rtid in tate_ids])

    is_pop_ref = np.isin(ref_artist, pop_artists)
    is_pop_tate = np.isin(tate_artist, pop_artists)

    pop_refs = ref_ids[is_pop_ref]
    hh_refs = ref_ids[~is_pop_ref]

    pop_tates = tate_ids[is_pop_tate]
    hh_tates = tate_ids[~is_pop_tate]
    return pop_refs, hh_refs, pop_tates, hh_tates

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

def add_info_cols_to_df(rt_df):
    rt_df = get_rt_in_train(rt_df)
    rt_df = get_rt_doc_ids(rt_df)
    rt_df = get_rt_raw_txt(rt_df)
    rt_df = get_rt_pp_txt(rt_df)
    return rt_df

def make_true_pairs_df(train_or_test_df):
    pair_df = get_true_rt_pairs(train_or_test_df)
    pair_df = add_info_cols_to_df(pair_df)
    pair_df['is_pair'] = [1 for _ in range(pair_df.shape[0])]
    pair_df['same_artist'] = [1 for _ in range(pair_df.shape[0])]
    pair_df.reset_index(drop=True, inplace=True)
    return pair_df

def make_false_pairs_df(train_or_test_pair_df, pop_artists, hh_artists):
    shuffled_refs, shuffled_tates = get_mismatched_rt_pairs(train_or_test_pair_df, pop_artists, hh_artists)
    shuff_df = pd.DataFrame({'ref_id': shuffled_refs, 'tate_id': shuffled_tates})
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

    corpus_dict = read_in_pkl('corpus_dict')
    train_df = read_in_pkl('train_df')
    test_df = read_in_pkl('test_df')
    lookup_dicts = read_in_pkl('lookup_dicts')
    pcorpuses = read_in_pkl('pcorpuses')

    artists = lookup_dicts[0]
    rt_doc_train_dict = lookup_dicts[3]
    rt_doc_test_dict = lookup_dicts[4]
    rt_in_train_dict = lookup_dicts[5]
    rt_artist_dict = lookup_dicts[7]
    rt_song_dict = lookup_dicts[9]

    ref_train_pcorpus = pcorpuses[0]
    ref_test_pcorpus = pcorpuses[1]
    tate_train_pcorpus = pcorpuses[2]
    tate_test_pcorpus = pcorpuses[3]

    pop_artists = ['Beyoncé', 'XXXTENTACION', 'Ariana Grande', 'The Weeknd', 'Original Broadway Cast of Hamilton', 'Drake']
    hh_artists = ['Lil Wayne', 'JAY-Z', 'Kanye West', 'Kendrick Lamar', 'J. Cole', 'Eminem']

    train_true_pair_df = make_true_pairs_df(train_df)
    train_false_pair_df = make_false_pairs_df(train_true_pair_df, pop_artists, hh_artists)
    train_all_pairings_df = make_full_df(train_true_pair_df, train_false_pair_df)

    test_true_pair_df = make_true_pairs_df(test_df)
    test_false_pair_df = make_false_pairs_df(test_true_pair_df, pop_artists, hh_artists)
    test_all_pairings_df = make_full_df(test_true_pair_df, test_false_pair_df)

    save_in_pkl(train_all_pairings_df, 'train_pairings_df')
    save_in_pkl(test_all_pairings_df, 'test_pairings_df')



    '''
    TESTS
    '''
    # print("\nTrain, True")
    # print(train_true_pair_df.shape)
    # print(train_true_pair_df['is_pair'].sum())
    # print(train_true_pair_df['same_artist'].sum())
    #
    # print("\nTrain, False")
    # print(train_false_pair_df.shape)
    # print(train_false_pair_df['is_pair'].sum())
    # print(train_false_pair_df['same_artist'].sum())
    #
    # print("\nTrain, All")
    # print(train_all_pairings_df.shape)
    # print(train_all_pairings_df['is_pair'].sum())
    # print(train_all_pairings_df['same_artist'].sum())
    #
    # print("\nTest, True")
    # print(test_true_pair_df.shape)
    # print(test_true_pair_df['is_pair'].sum())
    # print(test_true_pair_df['same_artist'].sum())
    #
    # print("\nTest, False")
    # print(test_false_pair_df.shape)
    # print(test_false_pair_df['is_pair'].sum())
    # print(test_false_pair_df['same_artist'].sum())
    #
    # print("\nTest, All")
    # print(test_all_pairings_df.shape)
    # print(test_all_pairings_df['is_pair'].sum())
    # print(test_all_pairings_df['same_artist'].sum())
