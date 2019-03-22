import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('./data/rt_data_dump.csv')


'''
Initial Data Prep/Cleaning
'''
# Drop duplicate columns
df.drop(['Unnamed: 0', 'rt_id.1', '_id'], axis=1, inplace=True)

# Drop non-text annotations
img_only_idxs = df[df['tate_text'].isna()].index
df.drop(img_only_idxs, axis=0, inplace=True)

# All songs are "False" -- therefore, this doesn't add anything!
df.drop('hot_song', axis=1, inplace=True)

# Create standardized "votes" feature (takes pageviews into account)
df['votes_per_1000views'] = (100000 * df['votes_total'] / df['pageviews']).round(2)
# New features for the number of characters in annotations/referents
df['chars_in_tate'] = df['tate_text'].str.len()
df['chars_in_referent'] = df['ref_text'].str.len()

# Can we do this for:
#   total # of lines in song?
#   word count in referent/annotation?

# https://stackoverflow.com/questions/18936957/count-distinct-words-from-a-pandas-data-frame

# list of words, in order, for referents/annotations
df['ref_word_lst'] = df['ref_text'].str.lower().str.split()
df['tate_word_lst'] = df['tate_text'].str.lower().str.split()

# word count for referents/annotations
df['ref_word_cnt'] = df['ref_word_lst'].str.len()
df['tate_word_cnt'] = df['tate_word_lst'].str.len()

# Removing Verse/Speaking Tags, Etc...
short_refs = df[df['ref_word_cnt'] <= 3]['ref_text'].unique()
tags_to_remove = []
short_refs_to_keep = []

for ref in short_refs:
    if ref[0] == '[' and ref[-1] == ']':
        tags_to_remove.append(ref)
    else:
        short_refs_to_keep.append(ref)

# COMPLETELY REMOVE
add_to_remove = ['Intro:', 'ENSEMBLE', 'JEFFERSON', 'Verse 2: Eminem', '[Chorus: KING GEORGE', '*Space Bar Tap*', 'BURR', 'LEE', '(Guitar Solo)', '(21st-Century schizoid man)']
# CHANGE/EDIT
edit_values = ['[HAMILTON]\n No', '[HAMILTON]\n Sir!', '[HAMILTON]\n Ha', '[HAMILTON]\n What?']
# OK
ok_keep = ['Mr. President', 'Mr. Vice President:', '“President John Adams”', 'Hamilton', 'Maty Noyes']

replace_dict = {'[HAMILTON]\n No':'No', '[HAMILTON]\n Sir!': 'Sir!', '[HAMILTON]\n Ha': 'Ha', '[HAMILTON]\n What?': 'What?'}

edit_idxs = []
for bad_ref in edit_values:
    mask = df['ref_text'] == bad_ref
    bad_idxs = list(df[mask].index)
    for i in bad_idxs:
        edit_idxs.append(i)

df['ref_text'].replace(replace_dict, inplace=True)

for i in add_to_remove:
    tags_to_remove.append(i)
    short_refs_to_keep.remove(i)

rt_idxs_to_drop = []
for bad_ref in tags_to_remove:
    mask = df['ref_text'] == bad_ref
    bad_idxs = list(df[mask].index)
    for i in bad_idxs:
        rt_idxs_to_drop.append(i)

df.drop(rt_idxs_to_drop, axis=0, inplace=True)






'''
Making querying easier with some artist/song/referent/etc ID dictionaries...
'''
artist_names = df['artist_name'].unique()

artist_id_dict = dict()
for artist_name in artist_names:
    artist_mask = df['artist_name'] == artist_name
    artist_id = df[artist_mask]['artist_id'].unique()[0]
    artist_id_dict[artist_id] = artist_name
'''
    {4: 'Lil Wayne',
     45: 'Eminem',
     69: 'J. Cole',
     2: 'JAY-Z',
     72: 'Kanye West',
     2358: 'The Weeknd',
     130: 'Drake',
     1421: 'Kendrick Lamar',
     26507: 'Ariana Grande',
     498: 'Beyoncé',
     572149: 'Original Broadway Cast of Hamilton',
     396565: 'XXXTENTACION'}
'''

song_id_dict = dict()
for song_title in song_titles:
    title_mask = df['song_title'] == song_title
    song_id = df[title_mask]['song_id'].unique()[0]
    song_id_dict[song_id] = song_title

by_artist = df.groupby('artist_name')
songs_by_artist = by_artist['song_id'].unique()
artist_songs_dict = dict()
for artist in artist_names:
    artist_songs_dict[artist] = songs_by_artist[artist]

titles_by_artist = by_artist['song_title'].unique()
artist_titles_dict = dict()
for artist in artist_names:
    artist_titles_dict[artist] = titles_by_artist[artist]




'''
Initial EDA
'''

col_names = df.columns
Index(['ref_text',
        'tate_text',
        'votes_total',
        'verified',
        'state',
        'song_id',
        'song_title',
        'full_title',
        'song_tate_cnt',
        'hot_song',
        'pageviews',
        'n_unreviewed_tates',
        'full_lyrics', 'url',
        'n_tate_contributors',
        'primary_contributor_id',
        'primary_contributor_IQ',
        'has_voters',
        'comment_cnt',
        'artist_name',
        'artist_id',
        'rt_id',
        '_id',
        'votes_per_1000views',
        'chars_in_tate',
        'chars_in_referent'],
      dtype='object')

'''
# TEXT BASED
'ref_text'
'tate_text'
'full_lyrics'
'ref_word_lst'
'tate_word_lst'

NUMERIC/CONTINUOUS
'votes_per_1000views'
    'votes_total'
    'pageviews'

'primary_contributor_IQ'
'n_tate_contributors'
'comment_cnt'
'song_tate_cnt'
'n_unreviewed_tates'
'chars_in_tate'
'chars_in_referent'
'ref_word_cnt'
'tate_word_cnt'

CLASS - BINARY
'verified'
'has_voters'

MULTI CLASS
'state'

ID BASED FEATS
'rt_id'
'song_id'
'song_title'
'full_title'
'artist_name'
'artist_id'
'_id'
'primary_contributor_id'
'url'



'''

print(df.shape)
# 3,573 annotations, 24 features/columns

print(df['state'].value_counts())
# accepted    3145
# pending      227
# verified     209

print(df['votes_total'].describe())
# count    3581
# mean       66.345
# std       158.87
# min       -46.0
# 25%        12.0
# 50%        29.0
# 75%        65.0
# max      4014.0

print(df['n_tate_contributors'].describe())
# count    3581.0
# mean        2.79
# std         2.12
# min         1.0
# 25%         1.0
# 50%         2.0
# 75%         4.0
# max        15.0

print(df['pageviews'].describe())
# count    3.581000e+03
# mean     1.777649e+06
# std      1.609913e+06
# min      202,259
# 25%      7.933820e+05
# 50%      1.276075e+06
# 75%      2.126581e+06
# max      13,873,324


print(df[df['votes_total'] <= 0]['votes_total'].value_counts().sum())
# Count of annotations with 0 or negative votes = 76

print(df[df['votes_total'] <= 0]['n_tate_contributors'].value_counts())
# Annotations w/ 0 or negative votes had 1, 2, or 3 n_tate_contributors

plt.scatter(x=df['n_tate_contributors'], y=df['votes_total'])

plt.scatter(x=df['votes_total'], y=df['n_tate_contributors'], c=df['votes_total'])

plt.scatter(x=df['votes_total'], y=df['n_tate_contributors'], c=df['verified'])

# ONCE PAGEVIEWS IS STANDARDIZED...
plt.scatter(x=df['votes_per_1000views'], y=df['n_tate_contributors'], c=df['verified'])
plt.scatter(x=df['votes_per_1000views'], y=df['n_tate_contributors'], c=df['votes_total'])


print(df['votes_per_1000views'].describe())
# count    3573.000000
# mean        4.351447
# std         8.538376
# min        -2.710000
# 25%         0.960000
# 50%         2.260000
# 75%         4.880000
# max       181.580000
