import tweepy
import json
import pandas as pd
import os
import datetime

folder_path = "./twitterdata/"

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def update_timeline(username):
    user = username
    print(user)
    profile_image = ''
    user_timeline = api.user_timeline(id=user)
    tweet_df = pd.DataFrame([], columns=['id', 'isRT', 'time', 'lang', 'text'])
    for index in range(0, len(user_timeline)):
        status = user_timeline[index]
        json_str = json.dumps(status._json)
        tweet = json.loads(json_str)
        isRT = 'RT @' in tweet['text']
        tweet_id = tweet['id']
        tweet_time = tweet['created_at']
        d = datetime.datetime.strptime(tweet_time, '%a %b %d %H:%M:%S %z %Y')
        tweet_time = d.strftime('%d.%m.%Y %H:%M:%S')
        lang = tweet['lang']
        tweet_text = tweet['text']
        tweet_info = [tweet_id, isRT, tweet_time, lang, tweet_text]
        profile_image = tweet['user']['profile_image_url']
        if os.path.isfile(folder_path + user + "_timeline.csv"):
            tweet_df = pd.read_csv(folder_path + user + "_timeline.csv")
            if tweet_id not in tweet_df['id'].values:
                tweet_df.loc[len(tweet_df)] = tweet_info
            tweet_df.to_csv(folder_path + user + "_timeline.csv", index=False)
        else:
            tweet_df.loc[len(tweet_df)] = tweet_info
    tweet_df.to_csv(folder_path + user + "_timeline.csv", index=False)

    print(len(tweet_df.index) < 2)
    return profile_image
