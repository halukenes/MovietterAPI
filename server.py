#!flask/bin/python
from flask import Flask, jsonify, abort, request, make_response
from flask_cors import CORS, cross_origin
from twitterAPI_s import update_timeline
from tweet_analysis_s import make_analysis
from movie_finder import recommend_movie
import pandas as pd
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/user/<username>', methods=['GET'])
@cross_origin()
def get_user(username):
    profile_image = update_timeline(username)
    analysis_topic = make_analysis(username)
    tweet_file = pd.read_csv("./tweet_analysis.csv")
    user_data = tweet_file.loc[tweet_file['username'] == username]
    user_row = user_data.iloc[0]
    related_movies = recommend_movie(username)
    return jsonify({'username': user_row['username'],
                    'anger': user_row['anger'],
                    'fear': user_row['fear'],
                    'joy': user_row['joy'],
                    'sadness': user_row['sadness'],
                    'analytical': user_row['analytical'],
                    'confident': user_row['confident'],
                    'tentative': user_row['tentative'],
                    'topic1': user_row['topic1'],
                    'topic2': user_row['topic2'],
                    'topic3': user_row['topic3'],
                    'topic4': user_row['topic4'],
                    'topic5': user_row['topic5'],
                    'topic_movies': related_movies['topic_movies'],
                    'emotion_movies': related_movies['emotion_movies'],
                    'profile_image': profile_image}), 201
    

if __name__ == '__main__':
    app.run(debug=True)
