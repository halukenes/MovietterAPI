import pandas as pd

movies_df = pd.read_csv('./movie_TA.csv')

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def array_of_string(main_str):
    str1 = main_str.replace(']','').replace('[','').replace('"','').replace('\'','')
    arr = str1.replace(' ','').split(",")
    return arr

def unique_array(arrays):
    for el in arrays:
        count_of_variable = arrays.count(el)
        if count_of_variable > 1:
            for i in range(count_of_variable - 1):
                arrays.remove(el)
    return arrays

def recommend_movie(username):
    scores = related_movies(username, 'topic')
    topic_movies = scores.head(5)['imdb_id'].values.tolist()
    
    user_df = pd.read_csv('./tweet_analysis.csv')   
    user_row = user_df[user_df['username'] == username].iloc[0]
    user_row = user_df[user_df['username'] == username].iloc[0]

    emotion_movies = scores.iloc[5:]

    if user_row['joy'] > user_row['sadness']:
        emotion_movies = emotion_movies[emotion_movies['movie_genre'].str.contains('Romance') |
                                        emotion_movies['movie_genre'].str.contains('War') |
                                        emotion_movies['movie_genre'].str.contains('Fantasy')]
    elif user_row['fear'] > user_row['joy']:
        emotion_movies = emotion_movies[emotion_movies['movie_genre'].str.contains('Action')]
    else: 
        emotion_movies = emotion_movies[emotion_movies['movie_genre'].str.contains('Comedy')]
    
    emotion_rcmd_movies = emotion_movies.head(5)['imdb_id'].values.tolist()

    return {'topic_movies': topic_movies,
            'emotion_movies': emotion_rcmd_movies }

def related_movies(username, purpose):
    user_df = pd.read_csv('./tweet_analysis.csv')
    user_row = user_df[user_df['username'] == username].iloc[0]

    mr_scores = pd.DataFrame(columns=['moviename', 'movie_genre', 'imdb_id', 'score', 'intersection'])
    user_topics = []
    user_topics += array_of_string(user_row.topic1)
    user_topics += array_of_string(user_row.topic2)
    user_topics += array_of_string(user_row.topic3)
    user_topics += array_of_string(user_row.topic4)
    user_topics += array_of_string(user_row.topic5)
    user_topics = unique_array(user_topics)

    all_matches = []
    useless_matches = []
    for row in movies_df.iterrows():
        movie_name = row[1]['name']
        movie_genre = row[1]['genre']
        movie_id = row[1]['imdb_id']
        movie_topic = []
        movie_topic += array_of_string(row[1]['topic1'])
        movie_topic += array_of_string(row[1]['topic2'])
        movie_topic += array_of_string(row[1]['topic3'])
        movie_topic += array_of_string(row[1]['topic4'])
        movie_topic += array_of_string(row[1]['topic5'])
        movie_topic = unique_array(movie_topic)
        intersection_list = []
        for i in range(1,6):
            user_topic = array_of_string(user_row['topic'+str(i)])
            intersection_list += intersection(movie_topic, user_topic)
        intersection_list = list(set(intersection_list))
        movie_score = len(intersection_list)
        score_row = [movie_name, movie_genre, movie_id, movie_score, intersection_list]
        mr_scores.loc[len(mr_scores)] = score_row
        all_matches += intersection_list
    for word in all_matches:
        if all_matches.count(word) > 30:
            useless_matches.append(word)
    useless_matches = list(set(useless_matches))
    for row in mr_scores.iterrows():
        word_list = row[1]['intersection']
        new_intersection = [word for word in word_list if word not in useless_matches]
        row[1]['intersection'] = new_intersection
        row[1]['score'] = len(new_intersection)
    mr_scores = mr_scores.sort_values(by=['score'], ascending=False)
    mr_scores.to_csv("./m.csv", index=False)

    return mr_scores