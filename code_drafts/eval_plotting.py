import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline

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
import re
from sklearn.linear_model import LogisticRegression, LinearRegression

pd.set_option("display.max_columns",100)

def plot_n_good_bad_pairs_avg_cs_across_train_texts(df, n_pairs, group_labels, colors):
    # Thinking: X, y are lists of pd.series... each list is new group to compare... for loop across list to plot...
    x = df['Mean of CS, WORST']
    y = df['Mean of CS, BEST']
    size = df['percent_self_recognized']*1000
    colors = df['text_int']

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)

    ax.set_title("Avg. Pair Similarity - Comparing Training Texts", fontsize=18)

    ax.scatter(x[:3], y[:3], s=size, alpha=0.5, label='lyrics only')
    ax.scatter(x[3:6], y[3:6], s=size, alpha=0.5, label='annotations only')
    ax.scatter(x[6:9], y[6:9], s=size, alpha=0.5, label='lyrics & annotations')

    ax.scatter(x[9], y[9], s=size, alpha=0.5, label='target_similarity')

    plt.axvline(x=0, c='red', alpha=0.3)
    plt.axhline(y=0, c='red', alpha=0.3)

    # plt.plot(data[:,0], m*data[:,0] + b,color='red',label='Our Fitting Line')
    ax.set_xlabel('Worst Pairs', fontsize=16)
    ax.set_ylabel('Best Pairs', fontsize=16)

    ax.legend(loc='upper center', ncol=4, markerscale=0.5, bbox_to_anchor=(0.5, -0.15))

    # plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)

    plt.plot([0, x[:3].mean()], [0, y[:3].mean()], color='blue', linestyle='-.', linewidth=2, alpha=0.5)
    plt.plot([0, x[3:6].mean()], [0, y[3:6].mean()], color='orange', linestyle='-.', linewidth=2, alpha=1)
    plt.plot([0, x[6:9].mean()], [0, y[6:9].mean()], color='green', linestyle='-.', linewidth=2, alpha=0.5)

    plt.show()

    fig.savefig('avg_pair_cs_texts_versionX.jpg')

def plot_dist_of_pair_cs_across_groups():
    tru_pairs = np.array(cosim_trupairs_rt2)
    non_pairs = np.array(cosim_nonpairs_rt2)

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)

    ax.set_title("Distributions of Referent-Tate Pair Similarity", fontsize=18)

    plt.hist(non_pairs, bins=100, color='red', alpha=0.3, label='mismatched pairs')
    plt.hist(tru_pairs, bins=100, color='blue', alpha=0.3, label='true pairs')

    plt.axvline(x=tru_pairs.mean(), c='blue', alpha=0.6, linewidth=3)
    plt.axvline(x=non_pairs.mean(), c='red', alpha=0.6, linewidth=3)
    # plt.axhline(y=0, c='red', alpha=0.3)

    ax.set_xlabel('Cosine Similarity of Pair', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)

    ax.legend(loc='upper center', ncol=2, markerscale=0.5, bbox_to_anchor=(0.5, -0.15))

    plt.show()

    fig.savefig('cs_distribution_rt2.jpg')


def
