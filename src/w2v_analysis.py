# http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XI7ULRlKglI

# imports needed and logging
import gzip
import gensim
import logging

logging.basicConfig(format=’%(asctime)s : %(levelname)s : %(message)s’, level=logging.INFO)

 '''
Next, is finding a really good dataset. The secret to getting Word2Vec really working for
you is to have lots and lots of text data in the relevant domain. For example, if your
goal is to build a sentiment lexicon, then using a dataset from the medical domain or
even wikipedia may not be effective. So, choose your dataset wisely.

 '''

with gzip.open (input_file, 'rb') as f:
        for i,line in enumerate (f):
            print(line)
            break

'''
To avoid confusion, the Gensim’s Word2Vec tutorial says that you need to pass a
list of tokenized sentences as the input to Word2Vec.

However, you can actually pass in a whole review as a sentence
(i.e. a much larger size of text),
if you have a lot of data and it should not make much of a difference.

In the end, all we are using the dataset for is to get all neighboring words
(the context) for a given target word.

'''


'''
Now that we’ve had a sneak peak of our dataset,
we can read it into a list so that we can pass
this on to the Word2Vec model. Notice in the code below,
that I am directly reading the compressed file. I’m also
doing a mild pre-processing of the reviews using
gensim.utils.simple_preprocess (line).
This does some basic pre-processing such as tokenization,
lowercasing, etc. and returns back a list of tokens (words).
'''

def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)
