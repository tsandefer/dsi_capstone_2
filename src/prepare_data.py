import pandas as pd
import numpy as np
import re
import pickle
import pymongo

from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import TaggedDocument

def read_mongodb_data():
    client = pymongo.MongoClient()
    db =  client.genius # existing db
    coll = db.lyric_annotations # existing collection
    df = pd.DataFrame(list(coll.find()))
    return df

def read_in_csv_data(filepath='../data/rt_data_dump.csv'):
    df = pd.read_csv(filepath)
    return df

def drop_subpar_info(df, subpar_cols):
    df.drop(subpar_cols, axis=1, inplace=True) # duplicate columns
    drop_empty_text(df, in_refs=False) # non-text annotations (only images, gifs, etc)
    df.drop('hot_song', axis=1, inplace=True) # All songs are "False" --> no information gain

def drop_empty_text(df, in_refs=True):
    is_empty = df['ref_text'].str.len() < 1 if in_refs else df['tate_text'].isna()
    empty_idxs = df[is_empty].index
    df.drop(empty_idxs, axis=0, inplace=True)

def standardize_votes_col(df):
    # Create standardized "votes" feature (takes pageviews into account)
    df['votes_per_1000views'] = (100000 * df['votes_total'] / df['pageviews']).round(2)
    return df

def make_txt_length_features(df):
    df['chars_in_tate'] = df['tate_text'].str.len()
    df['chars_in_referent'] = df['ref_text'].str.len()
    df['ref_word_lst'] = df['ref_text'].str.lower().str.split()
    df['tate_word_lst'] = df['tate_text'].str.lower().str.split()
    df['ref_word_cnt'] = df['ref_word_lst'].str.len()
    df['tate_word_cnt'] = df['tate_word_lst'].str.len()
    return df

def remove_verse_tags_from_tate_text(df):
    ref_lines = list(df['ref_text'])
    for idx, line in enumerate(ref_lines): # remove verse tags from referents
        ref_lines[idx] = re.sub(r'\[.*?\]', '', line)
    df['ref_text'] = ref_lines
    drop_empty_text(df, in_refs=True) # drop row if ref_text now empty str (whole ref was a tag)
    _drop_partial_tag_referents(df, ref_lines) # drop rows with partial tags
    return df

def _drop_partial_tag_referents(df, ref_lines):
    p = re.compile('\[|\]')
    partial_tag_idxs = []
    for idx, line in enumerate(ref_lines):
        has_bracket = p.search(line) != None
        if has_bracket:
            partial_tag_idxs.append(idx)
    df.drop(partial_tag_idxs, axis=0, inplace=True)

def drop_select_parenthesis_referents(df):
    subpar_ref_texts = ['(21st-Century schizoid man)', 'Chorus', 'Justin Vernon',
                        'Kóbor János', 'Intro:', 'ENSEMBLE', 'JEFFERSON',
                        'Verse 2: Eminem', '[Chorus: KING GEORGE', '*Space Bar Tap*',
                        'BURR', 'LEE', '(Guitar Solo)']
    for subpar_ref in subpar_ref_texts:
        is_subpar = df['ref_text'] == subpar_ref
        subpar_idxs = list(df[is_subpar].index)
    df.drop(subpar_idxs, axis=0, inplace=True)

def populate_clean_mongoDB(df):
    client = pymongo.MongoClient()
    db =  client.genius # existing db
    clean_coll = db.clean_data # create new collection
    try:
        clean_coll.insert_many(df.to_dict('records')) # insert the cleaned data
    except pymongo.errors.DuplicateKeyError:
        pass

def clean_data():
    df = read_mongodb_data()
    subpar_cols = ['Unnamed: 0', 'rt_id.1', '_id']
    drop_subpar_info(df, subpar_cols)
    drop_empty_text(df, in_tates=True)
    df = standardize_votes_col(df)
    df = make_txt_length_features(df)
    df = remove_verse_tags_from_tate_text(df)
    drop_select_parenthesis_referents(df)
    # Save cleaned data - MongoDB & .csv
    populate_clean_mongoDB(df)
    df.to_csv('../data/cleaned_data.csv')
    return df

def make_train_test_split(df, get_holdout=False):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    if get_holdout:
        holdout_df = test_df.copy()
        train_df.to_csv('../data/train_data.csv')
        holdout_df.to_csv('../data/holdout_data.csv')
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

def make_corpus_dicts(ref_train_df, ref_test_df, tate_train_df,
                        tate_test_df, rt_train_df, rt_test_df):
    txt_dfs = [(ref_train_df, ref_test_df),
                (tate_train_df, tate_test_df),
                (rt_train_df, rt_test_df)]
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
    return doc_rt_train_dict, doc_rt_test_dict

def make_doc_rt_dic(doc_rt_train_dict, doc_rt_test_dict):
    rt_doc_train_dict = dict((v,k) for k,v in doc_rt_train_dict.items())
    rt_doc_test_dict = dict((v,k) for k,v in doc_rt_test_dict.items())
    return rt_doc_train_dict, rt_doc_test_dict

def rt_in_train_dic(data, train_df):
    # need to make a dict that will tell us if an rt_id is in train or test set
    rt_in_training_dict = dict()
    for rt_id in data['rt_id'].unique():
        if rt_id in list(train_df['rt_id']):
            rt_in_training_dict[rt_id] = True
        else:
            rt_in_training_dict[rt_id] = False
    return rt_in_training_dict

def artist_rt_dic(data):
    gb_artist = data.groupby('artist_name')['rt_id'].unique()
    artist_rt_dict = dict()
    for artist_id in gb_artist.index:
        artist_rt_dict[artist_id] = gb_artist[artist_id]
    return artist_rt_dict

def rt_artist_dic(artist_rt_dict):
    rt_artist_dict = dict()
    for k,v in artist_rt_dict.items():
        for i in v:
            rt_artist_dict[i] = k
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
    for idx, line in enumerate(corpus):
        # Returns list of strings where contraction words, newlines, and punctuation is preserved
        tokens = re.findall(r"[\w'|\w’]+|[-–()\"\“\”.,!?;]+|[\n]+", line)
        tokens = [x.lower() for x in tokens]
        if tagged_docs:
            is_tate = 0 if idx < len(corpus) else 1
            tag_lst = [idx, is_tate] if include_ref_tag else [idx]
            yield TaggedDocument(tokens, tag_lst)
        else:
            yield tokens

def get_train_test_corpuses(txt_series_train, txt_series_test, include_ref_tag=False):
    train_corpus = list(get_token_lst(txt_series_train, tagged_docs=True, include_ref_tag=include_ref_tag))
    test_corpus = list(get_token_lst(txt_series_test))
    return train_corpus, test_corpus

def main():
    df = clean_data()

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

    rt_doc_train_dict, rt_doc_test_dict = make_doc_rt_dic(doc_rt_train_dict, doc_rt_test_dict)
    rt_in_train_dict = rt_in_train_dic(data, train_df)
    artist_rt_dict = artist_rt_dic(data)
    rt_artist_dict = rt_artist_dic(artist_rt_dict)

    song_rt_dict = song_rt_dic(data)
    rt_song_dict = rt_song_dic(song_rt_dict)
    song_id_to_title_dict = song_id_title_dic(data)

    lookup_dicts = [artists, doc_rt_train_dict, doc_rt_test_dict,
                    rt_doc_train_dict, rt_doc_test_dict, rt_in_train_dict,
                    artist_rt_dict, rt_artist_dict, song_rt_dict, rt_song_dict,
                    song_id_to_title_dict]

    save_in_pkl(lookup_dicts, 'lookup_dicts')

    ref_train_pcorpus, ref_test_pcorpus = list(get_train_test_corpuses(corpus_dict['referent'][0], corpus_dict['referent'][1]))
    tate_train_pcorpus, tate_test_pcorpus = list(get_train_test_corpuses(corpus_dict['tate'][0], corpus_dict['tate'][1]))
    rt_train_pcorpus, rt_test_pcorpus = list(get_train_test_corpuses(corpus_dict['ref-tate'][0], corpus_dict['ref-tate'][1]))
    rt_tagged_train_pcorpus, rt_tagged_test_pcorpus = list(get_train_test_corpuses(corpus_dict['ref-tate'][0], corpus_dict['ref-tate'][1], include_ref_tag=True))

    pcorpuses = [ref_train_pcorpus, ref_test_pcorpus, tate_train_pcorpus,
                    tate_test_pcorpus, rt_train_pcorpus, rt_test_pcorpus,
                    rt_tagged_train_pcorpus, rt_tagged_test_pcorpus]

    save_in_pkl(pcorpuses, 'pcorpuses')

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None  # default='warn'
    pd.set_option("display.max_columns", 100)

    df = main()
