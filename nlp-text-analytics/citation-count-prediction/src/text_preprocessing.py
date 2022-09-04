###########################
# Author: S. A. Owerre
# Date modified: 09/06/2021
# Class: Transformations
###########################

# filter warnings
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import nltk
import string
import gensim
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

class TextPreprocessing:
    """A class for preprocessing text data."""

    def __init__(self):
        """Define parameters."""
        pass

    def sen_tokenizer(self, text):
        """Sentence tokenizer removes:

        1. special characters
        2. punctuations
        3. stopwords
        and finally lemmatizes the token

        Parameters
        ----------
        text:  A string of texts or sentences

        Returns
        -------
        Lemmatized token
        """
        stop_words = stopwords.words('english')
        stop_words_ext = ['use', 'two', 'show', 'study', 'result']
        stop_words.extend(stop_words_ext)

        # remove special characters
        symbols = string.punctuation + '0123456789\n'
        nospe_char = [char for char in text if char not in symbols]
        clean_text = ''.join(nospe_char)

        # lower case, tokenize, lemmatizer, and removes top words
        token = clean_text.lower().split()
        token = [self.word_lemmatizer(x) for x in token if len(x) > 3]
        token = [x for x in token if x not in stop_words]
        return token

    def get_wordnet_pos(self, word):
        """POS (part of speech) tag to help lemmatizer to be effective.
        For example: goes and going will be lemmatized as go.

        Parameters
        ----------
        word:  A word

        Returns
        -------
        POS tag
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV,
        }

        return tag_dict.get(tag, wordnet.NOUN)

    def word_lemmatizer(self, word):
        """Word lemmatization function

        Parameters
        ----------
        word:  A word

        Returns
        -------
        Lemmatized word
        """
        lemmatizer = WordNetLemmatizer()
        word = lemmatizer.lemmatize(word, self.get_wordnet_pos(word))
        return word

    def add_bigram(self, token):
        """Add bigrams in the data."""
        bigram = gensim.models.Phrases(token)
        bigram = [bigram[line] for line in token]
        return bigram

    def add_trigram(self, token):
        """Add trigrams in the data."""
        bigram = self.add_bigram(token)
        trigram = gensim.models.Phrases(bigram)
        trigram = [trigram[line] for line in bigram]
        return trigram

    def bow_vector(self, data, text_col):
        """Create bag of words vector (i.e., document-term matrix) 
        using CountVectorizer.

        Parameters
        ----------
        data:  Pandas dataframe with a text column
        text_col:  Text column in data

        Returns
        -------
        bow vectors in Pandas Dataframe
        """
        counter = CountVectorizer(tokenizer=self.sen_tokenizer)
        bow_docs = pd.DataFrame(
            counter.fit_transform(data[text_col]).toarray(),
            columns=counter.get_feature_names(),
        )
        vocab = tuple(bow_docs.columns)
        return bow_docs, vocab

    def compute_coherence_lda(
        self,
        corpus,
        dictionary,
        tokens_list,
        start=None,
        limit=None,
        step=None,
    ):
        """Compute c_v coherence for various number of topics."""
        topic_coherence = []
        model_list = []
        texts = [[token for sub_token in tokens_list for token in sub_token]]
        for num_topics in range(start, limit, step):
            model = LdaMulticore(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                eta='auto',
                workers=4,
                passes=20,
                iterations=100,
                random_state=42,
                eval_every=None,
                alpha='asymmetric',  # shown to be better than symmetric 
                # in most cases
                decay=0.5,
                offset=64,  # best params from Hoffman paper
            )
            model_list.append(model)
            coherencemodel = CoherenceModel(
                model=model,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v',
            )
            topic_coherence.append(coherencemodel.get_coherence())
        return model_list, topic_coherence