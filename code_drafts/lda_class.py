'''
Topic Modeling LDA Class code originally written by Galvanize DSI classmate, Matt Devor (Github: MattD82)

Slightly adjusted to fit the needs of this project
'''

import pandas as pd
import numpy as np

# sklearn and scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import LatentDirichletAllocation
import scipy.sparse as sp

# import from NLTK
import nltk
from nltk.corpus import stopwords

# import re for text cleaning
import re

# import wordcloud
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

# import data loading functions from main file
import capstone_2 as cap

class SklearnTopicModeler(object):
    '''
    Uses sklearn LatentDirichletAllocation to model topics within the
    Seinfeld script corpus.
    This class should be easily generalizable to any corpus of documents.
    Tokenized to tf matrices using sklearn's CountVectorizer.
    '''
    def __init__(self, corpus, df_docs_by_ep):
        self.corpus = corpus

        # get title info for each episode from df_docs_by_ep
        self.titles = df_docs_by_ep.Title.values

        # dataframe of documents (scripts) by episode
        self.df_docs_by_ep = df_docs_by_ep

    def clean_vectorize(self, max_features=3000):
        # remove single quotes and convert to lower case
        self.corpus = [re.sub("\'", "", sent) for sent in self.corpus]
        self.corpus = [sent.lower() for sent in self.corpus]

        # add these stop words to default NLTK stop words
        more_stop_words = ['ya', 'ha', 'mr', 'okay', 'ah', 'alright', 'apartment', 'talk',
                           'happened', 'car', 'phone', 'looks', 'woman', 'getting', 'new',
                           'day', 'talking', 'wanna', 'bad', 'love', 'looking', 'night',
                           'work', 'em', 'cmon', 'kind', 'god', 'coffee', 'friend', 'away',
                           'making']

        self.stop_words = text.ENGLISH_STOP_WORDS.union(more_stop_words)

        # create tf matrix from corpus - note this removes additional punctuation automatically
        self.vectorizer = CountVectorizer(stop_words=self.stop_words,
                                          max_features=max_features,
                                          max_df = 0.85,
                                          min_df=2)
        self.tf = self.vectorizer.fit_transform(self.corpus)
        self.tf = sp.csr_matrix.toarray(self.tf)

         # theses are the words in our bag of words
        self.feature_names = self.vectorizer.get_feature_names()

    def fit_LDA(self, num_topics=10):
        # default to num_topics = 10 unless user changes.
        self.num_topics = num_topics

        # create LDA model using sklearn
        self.lda = LatentDirichletAllocation(n_components=num_topics,
                                    max_iter=5,
                                    learning_method='online',
                                    random_state=32,
                                    n_jobs=-1)
        self.lda.fit(self.tf)

        # phi is topics as rows and features (our tf-matrix in this case) as columns, which is the same as lda.components
        self.phi = self.lda.components_

        # theta relates total episodes (as rows) to topics (as columns)
        self.theta = self.lda.transform(self.tf)

    def calc_perplexity(self):
        return self.lda.perplexity(self.tf)

    def display_topics(self, num_top_words=10):
        # display topic # and key words for each topic
        # also, returns dictionar
        self.topic_dict = {}
        self.idx_dict = {}
        for topic_idx, topic in enumerate(self.phi):
            print("Topic %d:" % (topic_idx))
            print(" ".join([self.feature_names[i]
                            for i in topic.argsort()[:-num_top_words - 1:-1]]))

            self.topic_dict[topic_idx] = [self.feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
            self.idx_dict[topic_idx] = topic.argsort()[:-num_top_words - 1:-1]

        return self.topic_dict, self.idx_dict

    def display_similar_episodes(self, index, num_episodes=5):
        # allows the user to select an episode (by index), and displays
        # num_espisodes similar episodes using cosine distance
        pd.options.display.max_colwidth = 100
        pd.options.display.max_columns = 50

        article_chosen = self.titles[index]
        theta_dif = self.theta[index]
        diff_mat = pairwise_distances(theta_dif.reshape(1,-1), self.theta, metric='cosine')
        diff_mat = diff_mat.reshape(-1)

        indices_of_similar_episodes = diff_mat.argsort()[:num_episodes+1]

        print("Title of Episode is: {}".format(article_chosen))
        print("\n".join([self.titles[i] for i in indices_of_similar_episodes]))

        print(self.df_docs_by_ep.iloc[indices_of_similar_episodes])

    def display_episodes_from_topic(self, topic, num_episodes):

        indices_of_similar_episodes = self.theta[:, topic].argsort()[::-1][:num_episodes]

        print("Topic Chosen is: {}".format(topic))
        print("\n".join(self.titles[indices_of_similar_episodes]))

        print(self.df_docs_by_ep.iloc[indices_of_similar_episodes])

    def plot_wordcloud(self, num_top_words=10):
        # plots a wordcloud for each topic in the LDA model
        # num_top_words chooses number of words to display in each cloud

        # create dictionaries for word cloud
        topics, idxs = self.display_topics(num_top_words)

        tot_dict = {}
        new_dict = {}
        for i, j in idxs.items():
            t_dict = {}
            for v in j:
                word = self.feature_names[v]
                t_dict[word] = np.sum(self.tf[:, v])
                new_dict[i] = t_dict
                tot_dict.update(t_dict)

        # create wordcloud object and plot
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

        cloud = WordCloud(stopwords=self.stop_words,
                        background_color='white',
                        width=2500,
                        height=1800,
                        colormap='tab10',
                        color_func=lambda *args, **kwargs: cols[i],
                        prefer_horizontal=1.0)

        fig = plt.figure(figsize=(6, 6))

        gs = gridspec.GridSpec(4, 3, wspace=0.0, hspace=0.2, top=0.95, bottom=0.01,left=0.05,right=0.95)

        for i in range(num_top_words):
            ax = plt.subplot(gs[i])

            topic_words = new_dict[i]

            cloud.generate_from_frequencies(topic_words, max_font_size=300)

            ax.imshow(cloud)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.axis('off')
            ax.set_title('Topic ' + str(i), fontdict=dict(size=12))

        plt.show()


if __name__ == "__main__":
    # load in data and create corpus
    df_info, df_scripts = cap.load_data()
    df_docs_by_ep = cap.agg_dialogue_by_episode(df_scripts, df_info)
    corpus = cap.create_corpus_of_espisodes(df_docs_by_ep)

    # create LDA object, clean and vectorize text, and create LDA model
    lda = SklearnTopicModeler(corpus, df_docs_by_ep)
    lda.clean_vectorize()
    lda.fit_LDA()

    # calculate perplexity
    perplexity = lda.calc_perplexity()

    # display key words related to topics and
    lda.display_topics()
    lda.display_similar_episodes(0)
    lda.display_episodes_from_topic(0, 5)

    # plot wordcloud
    lda.plot_wordcloud()
