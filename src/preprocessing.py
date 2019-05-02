import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
import re

pd.set_option("display.max_columns",100)

def read_in_raw_data():
    df = pd.read_csv('../data/rt_data_dump.csv')
    return df

def drop_subpar_info(df):
    # Drop duplicate columns
    df.drop(['Unnamed: 0', 'rt_id.1', '_id'], axis=1, inplace=True)
    # Drop non-text annotations
    img_only_idxs = df[df['tate_text'].isna()].index
    df.drop(img_only_idxs, axis=0, inplace=True)
    # All songs are "False" -- therefore, this doesn't add anything!
    df.drop('hot_song', axis=1, inplace=True)

def drop_empty_text(df, in_tates=False, in_refs=False):
    # Drop rows with non-useful text
    if in_tates:
        is_null = df['tate_text'].isna()
        img_only_idxs = df[is_null].index
        df.drop(img_only_idxs, axis=0, inplace=True)
    if in_refs:
        is_empty_str = df['ref_text'].str.len() < 1
        empty_str_idxs = list(df[is_empty_str].index)
        df.drop(empty_str_idxs, axis=0, inplace=True)

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
    # remove tag from lines that have both brackets
    for idx, line in enumerate(ref_lines):
        ref_lines[idx] = re.sub(r'\[.*?\]', '', line)
    df['ref_text'] = ref_lines
    # if ref_text now empty str, remove the entire row (whole ref was a tag)
    drop_empty_text(df, in_refs=True)
    drop_partial_tag_referents(df, ref_lines)
    return df

def drop_partial_tag_referents(df, ref_lines):
    p = re.compile('\[|\]')
    partial_tag_idxs = []
    for idx, line in enumerate(ref_lines):
        has_bracket = p.search(line) != None
        if has_bracket:
            partial_tag_idxs.append(idx)
    df.drop(partial_tag_idxs, axis=0, inplace=True)

def drop_select_parenthesis_referents(df):
    subpar_ref_texts = ['(21st-Century schizoid man)', 'Chorus', 'Justin Vernon', 'Kóbor János', 'Intro:', 'ENSEMBLE', 'JEFFERSON', 'Verse 2: Eminem', '[Chorus: KING GEORGE', '*Space Bar Tap*', 'BURR', 'LEE', '(Guitar Solo)']
    subpar_ref_idxs = []
    for subpar_ref in subpar_ref_texts:
        is_subpar = df['ref_text'] == subpar_ref
        subpar_idxs = list(df[is_subpar].index)
        for i in subpar_idxs:
            subpar_ref_idxs.append(i)
    df.drop(subpar_ref_idxs, axis=0, inplace=True)

def main():
    df = read_in_raw_data()
    drop_subpar_info(df)
    drop_empty_text(df, in_tates=True)
    df = standardize_votes_col(df)
    df = make_txt_length_features(df)
    df = remove_verse_tags_from_tate_text(df)
    drop_select_parenthesis_referents(df)
    # df.to_csv('../data/preprocessed_data.csv')
    return df

if __name__ == '__main__':
    df = main()
    df.to_csv('../data/preprocessed_data.csv')
    print("Data has been preprocessed and saved!")
