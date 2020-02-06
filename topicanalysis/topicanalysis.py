import random
import spacy
from spacy.lang.en import English
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
import gensim
import pickle
import pyLDAvis.gensim


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            continue
        elif (token.text == 'RT'):
            continue
        elif token.text.startswith('[Image]'):
            continue
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


timeline_df = pd.read_csv("./moviereviews/tt0076759_reviews.csv")
tweets_list = timeline_df['Reviews'].values
# tweets_list = timeline_df.loc[timeline_df['isRT'] == False]['text'].values
# tweets = ' '.join(text_list)

spacy.load('en')
parser = English()
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

text_data = []
for line in tweets_list:
    tokens = prepare_text_for_lda(line)
    text_data.append(tokens)

dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

NUM_TOPICS = 3
ldamodel = gensim.models.ldamodel.LdaModel(corpus,
                                           num_topics=NUM_TOPICS,
                                           id2word=dictionary,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=20,
                                           alpha='auto',
                                           per_word_topics=True)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

""" lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display) """
