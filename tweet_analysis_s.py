import os
import pandas as pd
import re
import json
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import string
import numpy as np
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from pywsd.utils import lemmatize
from nltk.corpus import stopwords
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

stop_words = stopwords.words('english')
new_stopwords = ["didnt", "dont", "youre", "im", "really", "ever", "want", "sure", "just", "movie",
 "still", "when", "too", "make", "well", "good", "get", "watch", "much"]

folder_path = "./twitterdata/"

def make_analysis(username):
    topic_a = topic_analysis(username)
    em_a = emotion_analysis(username)
    user_row = [username,
                em_a[8],
                em_a[9],
                em_a[10],
                em_a[1],
                em_a[2],
                em_a[3],
                em_a[4],
                em_a[5],
                em_a[6],
                em_a[7],
                topic_a[0],
                topic_a[1],
                topic_a[2],
                topic_a[3],
                topic_a[4]]
    tweet_file = pd.read_csv("./tweet_analysis.csv")
    if not (tweet_file['username'] == username).any():
        tweet_file.loc[len(tweet_file)] = user_row
    else:
        index = tweet_file.index[tweet_file['username'] == username].tolist()[0]
        tweet_file.loc[index] = user_row
    tweet_file.to_csv("./tweet_analysis.csv", index=False)

    return tweet_file
    
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

def tokenization_w(texts):
    tokenized_texts = []
    for text in texts:
        w_token = word_tokenize(text)
        filtered_sentence = [w for w in w_token if not w in stop_words]
        tokenized_texts.append(filtered_sentence)

    return tokenized_texts

def lemmatization(texts):
    stop_words.extend(new_stopwords)
    nlp = spacy.load('en', disable=['parser', 'ner'])
    
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if not token.lemma_ in stop_words])
    
    return texts_out

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def topic_analysis(username):

    folder_path = "./twitterdata/"
    user_timeline = pd.read_csv(folder_path + username + '_timeline.csv', encoding='utf-8')
    filtered_timeline = user_timeline[(user_timeline.isRT == False) & (user_timeline.lang == 'en')]

    filtered_timeline.loc[:,'text'] = filtered_timeline['text'].apply(lambda x: x.lower())
    filtered_timeline.loc[:,'text'] = filtered_timeline['text'].apply(lambda x: re.sub(r'(^|[^@\w])@(\w{1,15})\b', '', x)) #user tags
    filtered_timeline.loc[:,'text'] = filtered_timeline['text'].apply(lambda x: re.sub(r'(^|[^@\w])#(\w{1,15})\b', '', x)) #user tags
    filtered_timeline.loc[:,'text'] = filtered_timeline['text'].apply(lambda x: re.sub(r'http\S+',  '', x))   #urls 
    filtered_timeline.loc[:,'text'] = filtered_timeline['text'].apply(lambda x: x.replace('\n','')) 
    filtered_timeline.loc[:,'text'] = filtered_timeline['text'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
    filtered_timeline.loc[:,'text'] = filtered_timeline['text'].apply(deEmojify) #emojis
    filtered_timeline = filtered_timeline.drop(filtered_timeline[filtered_timeline.text == ''].index) #delete empty strings

    tokens = tokenization_w(filtered_timeline['text'])
    lemmatized_data = lemmatization(tokens)
    data = [' '.join(list) for list in lemmatized_data]

    data_words = list(sent_to_words(data))

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    def make_bigrams(texts):
      return [bigram_mod[doc] for doc in texts]

    data_words_bigrams = make_bigrams(data_words)

    data_lemmatized = lemmatization(data_words_bigrams)

    id2word = corpora.Dictionary(data_lemmatized)

    texts = data_lemmatized

    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=5,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)

    x=lda_model.show_topics(num_topics=8, num_words=10,formatted=False)
    topics_words = [[wd[0] for wd in tp[1]] for tp in x]

    topics = []
    for words in topics_words:
        topics.append(words)
    return topics


def emotion_analysis(username):
    timeline = username + '_timeline.csv'
    user_timeline = pd.read_csv(folder_path + timeline, encoding='utf-8')
    tweet_count = user_timeline[user_timeline['isRT'] == False].shape[0]
    rt_count = user_timeline[user_timeline['isRT'] == True].shape[0]
    time_ordered = user_timeline
    time_ordered['time'] = pd.to_datetime(time_ordered.time)
    time_ordered = time_ordered.sort_values(by=['time'])
    last_tweet = time_ordered.tail(1).iloc[0]['time']
    print(last_tweet)

    filtered_timeline = user_timeline[(user_timeline['isRT'] == False) & (user_timeline['lang'] == 'en')]
    filtered_timeline.loc[:,'text'] = filtered_timeline['text'].apply(lambda x: re.sub(r'(^|[^@\w])@(\w{1,15})\b', ' ', x)) #user tags
    filtered_timeline.loc[:,'text'] = filtered_timeline['text'].apply(lambda x: re.sub(r'http\S+',  ' ', x))   #urls 
    filtered_timeline.loc[:,'text'] = filtered_timeline['text'].apply(lambda x: x.replace('\n',' ')) 
    filtered_timeline.loc[:,'text'] = filtered_timeline['text'].apply(lambda x: x.replace('"', ' '))
    filtered_timeline.loc[:,'text'] = filtered_timeline['text'].apply(deEmojify) #emojis
    filtered_timeline = filtered_timeline.drop(filtered_timeline[filtered_timeline.text == ''].index) #delete empty strings
    single_tweet = '. '.join(filtered_timeline.text.values)
    authenticator = IAMAuthenticator('nI-bfS7Ga_2rnVsqfzLcxcIMFn7LJXaIs__9CQdsyYde')
    tone_analyzer = ToneAnalyzerV3(
        version='2017-09-21',
        authenticator=authenticator
    )
    tone_analyzer.set_service_url('https://gateway-lon.watsonplatform.net/tone-analyzer/api')
    tone_analysis = tone_analyzer.tone(
        {'text': single_tweet},
        content_type='application/json'
    ).get_result()
    sentence_count = len(tone_analysis['sentences_tone'])
    tone_dict = {'anger': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'analytical': 0, 'confident': 0, 'tentative': 0}
    for sentence in tone_analysis['sentences_tone']:
        for tone in sentence['tones']:
            tone_dict[tone['tone_id']] += tone['score']
    for tone in tone_dict.keys():
        tone_dict[tone] = tone_dict[tone] / sentence_count
    # sorted(tone_dict.items(), key=lambda x:x[-1], reverse=True)
    user_emotion_row = [timeline.split('_timeline')[0],
                        tone_dict['anger'],
                        tone_dict['fear'],
                        tone_dict['joy'],
                        tone_dict['sadness'],
                        tone_dict['analytical'],
                        tone_dict['confident'],
                        tone_dict['tentative'],
                        tweet_count,
                        rt_count,
                        last_tweet]
    return user_emotion_row
    # twitteruser_emotion.loc[len(twitteruser_emotion)] = user_emotion_row
    # twitteruser_emotion.to_csv(folder_path + "./twitter_EA.csv", index=False)
