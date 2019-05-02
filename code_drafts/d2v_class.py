# Doc2Vec class that will hopefully make training / evaluation a lot easier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
from scipy.spatial.distance import cosine, euclidean
import re
import preprocessing
from sklearn.linear_model import LogisticRegression, LinearRegression

class DocVecModelEvaluator(object):
    '''
    Uses Gensim's Doc2Vec model to keep track of different model variations for comparison
    '''
    def __init__(self, test_corpus):
        self.good_refs = test_corpus[0]
        self.good_tates = test_corpus[1]
        self.bad_refs = test_corpus[2]
        self.bad_tates = test_corpus[3]

        # get title info for each episode from df_docs_by_ep
        # self.titles = df_docs_by_ep.Title.values
        # dataframe of documents (scripts) by episode
        # self.df_docs_by_ep = df_docs_by_ep
        self.eval_areas_dict = {'percent_self_recognized':0,'mean_good_s':0, 'mean_bad_s':0, 'dist':0}
        self.eval_dict = dict()
        # Short, but descriptive strings of trained models
        self.model_names = []
        self.model_training_corps = dict()
        # append actual trained models on here...
        # self.trained_models = []
        # model header leads to trained model?
        self.trained_models = dict()
        self.training_corpus_lst = dict()
        self.default_hyperparams = {'vector_size': 100, 'dm':1, 'min_count':2, 'epochs':80}
        self.sims = dict()
        self.vecs = dict()

    def get_potential_training_corpuses(self, ref_train, tate_train, rt_train, rt_tag_train):
        self.training_corpus_lst['r'] = ref_train
        self.training_corpus_lst['t'] = tate_train
        self.training_corpus_lst['rt'] = rt_train
        self.training_corpus_lst['rt_tagged'] = rt_tag_train
        return len(self.training_corpus_lst)

    def fit_d2v(self, model_name, train_corp_name='rt_tagged', hyperparams=None):
        hyperparams = hyperparams if hyperparams else self.default_hyperparams
        train_corpus = self.training_corpus_lst[train_corp_name]
        # instantiate new model
        model = gensim.models.doc2vec.Doc2Vec(vector_size=200, dm=1, min_count=2, epochs=110)
        model.build_vocab(train_corpus)
        %time model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        self.model_training_corps[model_name] = train_corp_name
        self.model_names.append(model_name)
        # self.trained_models.append(model)
        self.trained_models[model_name] = model
        return model

    def evaluate_model(self, model_name, train_corpus, test_df, dist='cs', get_sims=False):
        # Metric: model's ability to recognize a training example as its own best match
        eval_areas = self.eval_areas_dict.copy()
        if train_corpus not in self.model_training_corps[model_name]:
            model = self.fit_Doc2Vecs(train_corp_name=train_corpus, model_name, hyperparams=None)
        else:
            model = self.trained_models[model_name]
        self_eval_perc = calc_self_recognition_ability(model_name, train_corpus)
        eval_areas['percent_self_recognized'] = self_eval_perc
        eval_areas['dist'] = dist
        # How to index in to separate into ref/tates?

        good_sims = calc_similarity_series(self.good_tates, self.good_refs, dist=dist)
        mean_good_s = good_sims.mean()
        eval_areas['mean_good_s'] = mean_good_s

        bad_sims = calc_similarity_series(self.bad_tates, self.bad_refs, dist=dist)
        mean_bad_s = bad_sims.mean()
        eval_areas['mean_bad_s'] = mean_bad_s

        if get_sims:
            self.sims[model_name] = {'good_sims':good_sims, 'bad_sims':bad_sims}

        self.eval_dict[model_name] = eval_areas

        model_eval_df = view_eval_dict()

    def view_eval_dict(self):
        model_eval_df = pd.DataFrame.from_dict(self.eval_dict).T.round(2)
        # target_row = pd.DataFrame(target_values, columns=cols, index=['target_values'])
        # model_eval_df = model_eval_df.append(target_row)
        return model_eval_df

    def calc_self_recognition_ability(self, model_name, train_corpus):
        # ability to recognize a training doc as its own best match
        model = self.trained_models[model_name]
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        ranks = []
        second_ranks = []
        n_training_docs = len(train_corpus)
        for doc_id in range(n_training_docs):
            inferred_vector = model.infer_vector(train_corpus[doc_id].words)
            sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs)) #gives you all document tags and their cosine similarity
            # Gets its own self-ranking
            rank = [docid for docid, sim in sims].index(doc_id)
            ranks.append(rank)
            second_ranks.append(sims[1])
        # Let's count how each document ranks with respect to the training corpus
        rank_counter = collections.Counter(ranks)  # Results vary between runs due to random seeding and very small corpus
        n_self_recognized = rank_counter[0]
        self_recog_rate = n_self_recognized / n_training_docs
        greater_than_95 = self_recog_rate >= 0.95
        return rank_counter, self_recog_rate, greater_than_95

    def infer_vector(self, doc, model_name, epochs=None):
        if model_name not in self.model_dict.keys():
            print("Sorry, that model hasn't been trained yet!")
            # return?
        model = self.trained_models[model_name]
        # Can specify num of epochs, but generally might want to go w/ model training setting
        epochs = epochs if epochs else model.epochs
        inferred_vector = model.infer_vector(mod_corpus[doc_id].words, epochs=epochs)
        return inferred_vector

    def calc_pair_similarity(self, doc1, doc2, dist='cs'):
        # doc1 and doc2 are each a list of tokenized texts - one tate, one ref
        v1 = infer_vector(doc1).reshape(-1, 1)
        v2 = infer_vector(doc2).reshape(-1, 1)
        pair_sim = 1 - cosine(ref_inf_vec, tate_inf_vec) if dist = 'cs' else euclidean(ref_inf_vec, tate_inf_vec)
        return pair_sim

    def calc_similarity_series(self, group1, group2, dist='cs'):
        # think can do if each array are numpy arrays...
        # do to the arrays, or have to iterate thru?
        n = group1.shape[0]
        cs_series = []
        for idx in range(n):
            pair_sim = calc_pair_similarity(group1[idx], group2[idx], dist)
            cs_series.append(pair_sim)
        return np.array(cs_series)

    # def print_most_similar_examples(model, mod_corpus, doc_id):
    #     inferred_vector = model.infer_vector(mod_corpus[doc_id])
    #     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    #     print('Document ({}): «{}»\n'.format(doc_id, ' '.join(mod_corpus[doc_id].words)))
    #     print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    #     for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    #         print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(mod_corpus[sims[index][0]].words)))


if __name__ == "__main__":

    # load in data and create corpus
    all_pairs_df = pd.concat([pair_df, shuff_df], axis=0, ignore_index=True)

    X_good = all_pairs_df.loc[:2032, ['ref_pp_text', 'tate_pp_text']]
    X_bad = all_pairs_df.loc[2032:, ['ref_pp_text', 'tate_pp_text']]

    X_good_refs = X_good['ref_pp_text']
    X_good_tates = X_good['tate_pp_text']
    X_bad_refs = X_bad['ref_pp_text']
    X_bad_tates = X_bad['tate_pp_text']

    test_corpus = (X_good_refs, X_good_tates, X_bad_refs, X_bad_tates)

    dv = DocVecModels(test_corpus)
    train_corpus = (ref_train_pcorpus,
                    tate_train_pcorpus,
                    rt_train_pcorpus,
                    rt_tagged_train_pcorpus)

    corps = dv.get_potential_training_corpuses(ref_train_pcorpus,
                                                tate_train_pcorpus,
                                                rt_train_pcorpus,
                                                rt_tagged_train_pcorpus)

    print(corps)

    model = dv.fit_d2v('rt-t-def', train_corp_name='rt_tagged', hyperparams=None)


    get_potential_training_corpuses(ref_train, tate_train, rt_train, rt_tag_train)
