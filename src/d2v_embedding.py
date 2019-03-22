import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

pd.set_option("display.max_columns",100)

def get_and_clean_data():
    df = pd.read_csv('../data/rt_data_dump.csv')
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
    return df

def perform_ttsplit(cleaned_df):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train.to_csv('../data/genius_data_train_319.csv')
    df_test.to_csv('../data/genius_data_test_319.csv')
    return df_train, df_test

def get_ref_tate_dfs(df_train, df_test):
    ref_df_train = df_train[['ref_text', 'rt_id']]
    tate_df_train = df_train[['tate_text', 'rt_id']]

    ref_df_test = df_test[['ref_text', 'rt_id']]
    tate_df_test = df_test[['tate_text', 'rt_id']]

    ref_df_train.reset_index(drop=True, inplace=True)
    tate_df_train.reset_index(drop=True, inplace=True)

    ref_df_test.reset_index(drop=True, inplace=True)
    tate_df_test.reset_index(drop=True, inplace=True)
    # (tate_df_train['rt_id'] == ref_df_train['rt_id']).all()
    # (tate_df_test['rt_id'] == ref_df_test['rt_id']).all()
    return ref_df_train, tate_df_train, ref_df_test, tate_df_test

# GONNA USE THIS TUTORIAL FOR REST OF ATTEMPT:
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
def isolate_corpuses(ref_df_train, tate_df_train, ref_df_test, tate_df_test):
    refs_train = ref_df_train['ref_text']
    tates_train = tate_df_train['tate_text']

    refs_test = ref_df_test['ref_text']
    tates_test = tate_df_test['tate_text']
    return refs_train, refs_test, tates_train, tates_test

def make_rt_doc_idx_dicts(ref_df_train, ref_df_test):
    rt_to_doc_idx_train = ref_df_train['rt_id']
    rt_to_doc_idx_test = ref_df_test['rt_id']

    rt_doc_idx_train_dict = rt_to_doc_idx_train.to_dict()
    rt_doc_idx_test_dict = rt_to_doc_idx_test.to_dict()
    return rt_doc_idx_train_dict, rt_doc_idx_test_dict

# Define a Function to Read and Preprocess Text
# Below, we define a function to open the train/test file (with latin encoding), read the file line-by-line, pre-process each line using a simple gensim pre-processing tool (i.e., tokenize text into individual words, remove punctuation, set to lowercase, etc), and return a list of words. Note that, for a given file (aka corpus), each continuous line constitutes a single document and the length of each line (i.e., document) can vary. Also, to train the model, we'll need to associate a tag/number with each document of the training corpus. In our case, the tag is simply the zero-based line number.
def read_corpus(doc_series, tokens_only=False):
    # with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
    for i, line in enumerate(doc_series):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def get_rt_tt_corpuses(refs_train, refs_test, tates_train, tates_test):
    train_tate_corpus = list(read_corpus(tates_train))
    test_tate_corpus = list(read_corpus(tates_test, tokens_only=True))

    train_refs_corpus = list(read_corpus(refs_train))
    test_refs_corpus = list(read_corpus(refs_test, tokens_only=True))
    return train_tate_corpus, test_tate_corpus, train_refs_corpus, test_refs_corpus

# # Inferring a Vector

# One important thing to note is that you can now infer a vector for any piece of text without having to re-train the model by passing a list of words to the model.infer_vector function. This vector can then be compared with other vectors via cosine similarity.
# Note that infer_vector() does not take a string, but rather a list of string tokens, which should have already been tokenized the same way as the words property of original training document objects.
# Also note that because the underlying training/inference algorithms are an iterative approximation problem that makes use of internal randomization, repeated inferences of the same text will return slightly different vectors.

# # Assessing Model
# To assess our new model, we'll first infer new vectors for each document of the training corpus, compare the inferred vectors with the training corpus, and then returning the rank of the document based on self-similarity. Basically, we're pretending as if the training corpus is some new unseen data and then seeing how they compare with the trained model. The expectation is that we've likely overfit our model (i.e., all of the ranks will be less than 2) and so we should be able to find similar documents very easily. Additionally, we'll keep track of the second ranks for a comparison of less similar documents.
def assess_model(model, mod_corpus):
    ranks = []
    second_ranks = []
    n_training_docs = len(mod_corpus)
    for doc_id in range(n_training_docs):
        inferred_vector = model.infer_vector(mod_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])
    # Let's count how each document ranks with respect to the training corpus
    rank_counter = collections.Counter(ranks)  # Results vary between runs due to random seeding and very small corpus
    cnt_correct_self_similarity_docs = rank_counter[0]
    perc_correct_similarity = cnt_correct_self_similarity_docs / n_training_docs
    greater_than_95 = perc_correct_similarity >= 0.95
    return ranks, second_ranks, rank_counter, perc_correct_similarity, greater_than_95

def print_most_similar_examples(model, mod_corpus, doc_id):
    inferred_vector = model.infer_vector(mod_corpus[doc_id])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    print('Document ({}): «{}»\n'.format(doc_id, ' '.join(mod_corpus[doc_id].words)))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(mod_corpus[sims[index][0]].words)))

def compare_second_most_similar_doc_examples(mod_corpus, mod_second_ranks):
    # We can run the next cell repeatedly to see a sampling other target-document comparisons.
    # Pick a random document from the corpus and infer a vector from the model
    doc_id = random.randint(0, len(mod_corpus) - 1)
    # Compare and print the second-most-similar document
    print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(mod_corpus[doc_id].words)))
    sim_id = second_ranks[doc_id]
    print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(modcorpus[sim_id[0]].words)))

def random_model_tests(model, train_corpus, test_corpus):
    # Testing the Model
    # Using the same approach above, we'll infer the vector for a randomly chosen test document, and compare the document to our model by eye.
    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(test_corpus) - 1)
    inferred_vector = model.infer_vector(test_corpus[ref_doc_id])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    # Compare and print the most/median/least similar documents from the train corpus
    print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

def get_cos_sim_for_best_worst_rt_pairs(df, base_model, r_corpus_train, r_corpus_test, t_corpus_train, t_corpus_test):
    # 3 rt_ids for annotations we know to be particularly "bad"
    bottom_3_rtid = [rt_id for rt_id in df.nsmallest(3, 'votes_per_1000views')['rt_id']]
    # 3 rt_ids for annotations we know to be "good", minus the one that's only annotating a '[VERSE]' tag
    top_3_rtid = [rt_id for rt_id in df.nlargest(3, 'votes_per_1000views')['rt_id']]
    top = []
    bottom = []
    for top_or_bottom in [bottom_3_rtid, top_3_rtid]:
        doc_ids = []
        train_or_tests = []
        cs = []
        for rt_id in top_or_bottom:
            if rt_id in list(corpus_train['rt_id']):
                mask = corpus_train['rt_id'] == rt_id
                doc_ids.append(corpus_train[mask].index[0])
                train_or_tests.append('train')
            else:
                mask = corpus_test['rt_id'] == rt_id
                doc_ids.append(corpus_test[mask].index[0])
                train_or_tests.append('test')
        for idx, doc_id in enumerate(doc_ids):
            # let's start with using tate model
            if train_or_tests[idx] == 'train':
                r_inf_vec = base_model.infer_vector(r_train_corpus[doc_id].words, epochs=base_model.epochs).reshape(-1, 1)
                t_inf_vec = base_model.infer_vector(t_train_corpus[doc_id].words, epochs=base_model.epochs).reshape(-1, 1)
            else:
                r_inf_vec = base_model.infer_vector(r_test_corpus[doc_id], epochs=base_model.epochs).reshape(-1, 1)
                t_inf_vec = base_model.infer_vector(t_test_corpus[doc_id], epochs=base_model.epochs).reshape(-1, 1)
            # might need to just do straight up np.cosine similarity calc between vecs
            rt_iv_cs = 1 - cosine(r_inf_vec, t_inf_vec)
            cs.append(rt_iv_cs)
        mean_cs = np.array(rt_iv_cs).mean()
        if top_or_bottom == bottom_3_rtid:
            bottom.append([mean_cs, cs])
        else:
            top.append([mean_cs, cs])
    return top, bottom


# Training the Model
# Instantiate a Doc2Vec Object
# Now, we'll instantiate a Doc2Vec model with a vector size with 50 words and iterating over the training corpus 40 times. We set the minimum word count to 2 in order to discard words with very few occurrences. (Without a variety of representative examples, retaining such infrequent words can often make a model worse!) Typical iteration counts in published 'Paragraph Vectors' results, using 10s-of-thousands to millions of docs, are 10-20. More iterations take more time and eventually reach a point of diminishing returns.
# However, this is a very very small dataset (300 documents) with shortish documents (a few hundred words). Adding training passes can sometimes help with such small datasets.

if __name__ == '__main__':
cleaned_df = get_and_clean_data()
df_train, df_test = perform_ttsplit(cleaned_df)
ref_df_train, tate_df_train, ref_df_test, tate_df_test = get_ref_tate_dfs(df_train, df_test)
refs_train, refs_test, tates_train, tates_test = isolate_corpuses(ref_df_train, tate_df_train, ref_df_test, tate_df_test)
rt_doc_idx_train_dict, rt_doc_idx_test_dict = make_rt_doc_idx_dicts(ref_df_train, ref_df_test)
train_tate_corpus, test_tate_corpus, train_refs_corpus, test_refs_corpus = get_rt_tt_corpuses(refs_train, refs_test, tates_train, tates_test)

ref_model1 = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
ref_model2 = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=80)
ref_model3 = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=1, epochs=80)

tate_model1 = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
tate_model2 = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=80)
tate_model3 = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=80)

rt_model1 = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
rt_model2 = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=80)
rt_model3 = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=1, epochs=80)

ref_models = [ref_model1, ref_model2, ref_model3]
ref_mod_text = ['RM1(vs=50, mc=2, e=40)', 'RM2(vs=50, mc=2, e=80)', 'RM3(vs=100, mc=1, e=80)']
tate_models = [tate_model1, tate_model2, tate_model3]
tate_mod_text = ['TM1(vs=50, mc=2, e=40)', 'TM2(vs=50, mc=2, e=80)', 'TM3(vs=100, mc=1, e=80)']
rt_models = [tate_model1, tate_model2, tate_model3]
rt_mod_text = ['RTM1(vs=50, mc=2, e=40)', 'RTM2(vs=50, mc=2, e=80)', 'RTM3(vs=100, mc=1, e=80)']


for idx, ref_mod in enumerate(ref_models):
    model_header = ref_mod_text[idx]
    print("EVALUATING:", model_header)
    ref_mod.build_vocab(train_refs_corpus)
    %time ref_mod.train(train_refs_corpus, total_examples=ref_mod.corpus_count, epochs=ref_mod.epochs)

    mod_ranks, mod_second_ranks, mod_rank_counter, perc_correct_similarity, greater_than_95 = assess_model(ref_mod, train_refs_corpus)
    print("Model Self-Similarity Test Passed?:", greater_than_95)
    print("Model % Self-Similar:", perc_correct_similarity.round(4))

    doc_id = 4
    print_most_similar_examples(ref_mod, train_refs_corpus, doc_id)
    compare_second_most_similar_doc_examples(train_refs_corpus, mod_second_ranks)

    random_model_tests(model, train_corpus, test_corpus)

    top, bottom = get_cos_sim_for_best_worst_rt_pairs(df, base_model, train_refs_corpus, test_refs_corpus, train_tate_corpus, test_tate_corpus)
    print(top)
    print(bottom)
    print('NOW, ONTO THE NEXT MODEL!')

for idx, tate_mod in enumerate(tate_models):
    model_header = tate_mod_text[idx]
    print("EVALUATING:", model_header)
    tate_mod.build_vocab(train_tate_corpus)
    %time tate_mod.train(train_tate_corpus, total_examples=tate_mod.corpus_count, epochs=tate_mod.epochs)

    mod_ranks, mod_second_ranks, mod_rank_counter, perc_correct_similarity, greater_than_95 = assess_model(ref_mod, train_refs_corpus)
    print("Model Self-Similarity Test Passed?:", greater_than_95)
    print("Model % Self-Similar:", perc_correct_similarity.round(4))

    doc_id = 4
    print_most_similar_examples(ref_mod, train_refs_corpus, doc_id)
    compare_second_most_similar_doc_examples(train_refs_corpus, mod_second_ranks)

    random_model_tests(model, train_corpus, test_corpus)

    top, bottom = get_cos_sim_for_best_worst_rt_pairs(df, base_model, train_refs_corpus, test_refs_corpus, train_tate_corpus, test_tate_corpus)
    print(top)
    print(bottom)
    print('NOW, ONTO THE NEXT MODEL!')

print("Alright, finished for now. Yay!")

    # for idx, rt_mod in enumerate(rt_models):
    #     model_header = rt_mod_text[idx]
    #     print("EVALUATING:", model_header)
    #     tate_mod.build_vocab(train_tate_corpus)
    #     %time tate_mod.train(train_tate_corpus, total_examples=rt_mod.corpus_count, epochs=rt_mod.epochs)
    #
    #     mod_ranks, mod_second_ranks, mod_rank_counter, perc_correct_similarity, greater_than_95 = assess_model(rt_mod, train_refs_corpus)
    #     print("Model Self-Similarity Test Passed?:", greater_than_95)
    #     print("Model % Self-Similar:", perc_correct_similarity.round(4))
    #
    #     doc_id = 4
    #     print_most_similar_examples(rt_mod, train_refs_corpus, doc_id)
    #     compare_second_most_similar_doc_examples(train_refs_corpus, mod_second_ranks)
    #
    #     sims = random_model_tests(rt_mod, train_corpus, test_corpus)
    #
    #     top, bottom = get_cos_sim_for_best_worst_rt_pairs(df, rt_mod, train_refs_corpus, test_refs_corpus, train_tate_corpus, test_tate_corpus)





# ...How about we try to measure cosine similarity across these document pairs....??

# What does this actually mean, since the model itself is supposed to be trained on different corpuses of documents...?!




# Initialize & train a mode
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)


# Persist a model to disk
fname = get_tmpfile("my_doc2vec_model")
model.save(fname)
model = Doc2Vec.load(fname)  # you can continue training with the loaded model!


# If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)



# Infer vector for a new document:
vector = model.infer_vector(["system", "response"])

'''
