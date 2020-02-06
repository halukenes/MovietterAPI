import re
import os
import string
import numpy as np
import pandas as pd
from pprint import pprint
import requests

from pywsd.utils import lemmatize

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
# spacy for lemmatization
import spacy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

stop_words = set(stopwords.words('english'))
stop_words.add('didnt')
stop_words.add('movie')
stop_words.add('film')
stop_words.add('could')
stop_words.add('be')
stop_words.add('even')
stop_words.add('would')

def preprocess(text):
    new_text = re.sub('<.*?>', '', text)   # remove HTML tags
    new_text = re.sub(r'[^\w\s]', ' ', new_text) # remove punc.
    new_text = re.sub(r'\d+','',new_text) # remove numbers
    new_text = new_text.lower() # lower case, .upper() for upper
    return new_text

def tokenization_w(texts):
    tokenized_texts = []
    for text in texts:
        w_token = word_tokenize(text)
        filtered_sentence = [w for w in w_token]
        tokenized_texts.append(filtered_sentence)
    return tokenized_texts

def lemmatization(stem_array):
    lemmatized = []
    for stems in stem_array:
        lemmas = [lemmatize(x) for x in stems if not x in stop_words]
        lemmatized.append(lemmas)
    return lemmatized

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def get_topics(dataset, column_name, method):
    dataset[column_name] = dataset[column_name].apply(preprocess)    
    tokens = tokenization_w(dataset[column_name])
    lemmatized_data = lemmatization(tokens)
    processed_data = [' '.join(list) for list in lemmatized_data]
    data_words = list(sent_to_words(processed_data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Form Bigrams
    data_words_trigrams = [trigram_mod[bigram_mod[doc]] for doc in data_words]

    # Create Dictionary
    id2word = corpora.Dictionary(data_words_trigrams)

    # Create Corpus
    texts = data_words_trigrams

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=8,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=False)

    x=lda_model.show_topics(num_topics=8, num_words=10,formatted=False)
    topics_words = [[wd[0] for wd in tp[1]] for tp in x]

    topics = []
    for words in topics_words:
        topics.append(words)

    return topics

folder_path = "./moviereviews/"

df = pd.DataFrame(columns=['imdb_id', 'name', 'genre', 'dataset_genre','topic1','topic2', 'topic3', 'topic4', 'topic5'])

for folder in os.listdir(folder_path):
    print('looking into movies of: ' + folder)
    genre_path = folder_path + folder + '/'
    
    for file_name in os.listdir(genre_path):
        if file_name.startswith('tt'):
            print(' found movie:    ' + file_name)
            dataset_path = genre_path + file_name
            reviews = pd.read_csv(dataset_path, encoding = 'utf-8')
            print('   finding topics of:    ' + file_name)
            topics = get_topics(reviews, 'Reviews', 't')
            print('   topics found for: ' + file_name)

            apikey = '991e2097'
            imdb_id = file_name.split('_')[0]
            omdbapi_url = "http://www.omdbapi.com"

            print('   sending get request for movie details:    ' + imdb_id)
            r = requests.get(omdbapi_url, params={'apikey':apikey,'i':imdb_id})
            print('   collected movie details of:    ' + imdb_id)

            movie_name = ''
            genre = ''
            try:
                movie_details = r.json()
                movie_name = movie_details['Title']
                genre = movie_details['Genre']
                pass
            except:
                movie_name = 'error_handled'
                genre = 'error_handled'
                pass
            
            movie_row = [imdb_id, movie_name, genre, folder, topics[0], topics[1], topics[2], topics[3], topics[4]]
            print(' the movie is added to dataframe:    ' + imdb_id)
            df.loc[len(df)] = movie_row

df.to_csv("./movie_TA.csv", index=False)