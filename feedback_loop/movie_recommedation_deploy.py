import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import os
import json
import time
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
from flask import Flask
from flask import jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import time
from datetime import datetime
import re
from zoneinfo import ZoneInfo
import csv

EMBEDDING_SIZE = 50

class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.movie_bias = layers.Embedding(num_movies, 1)
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to be between 0 and 11
        return tf.nn.sigmoid(x)
    
def separate_ratings_and_movies(df):
    """
    Function to separate ratings and movies from a given DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame containing the data.

    Returns:
    tuple: A tuple containing two DataFrames - df_ratingsdata and df_moviesdata.
    """
    df_ratingsdata = df[df['request'].str.startswith('GET /rate')]
    extracted_data = df_ratingsdata.iloc[:, 2].str.extract(r'/rate/(.*?)=(\d+)')
    df_ratingsdata.loc[:, 'movieid'] = extracted_data[0]
    df_ratingsdata.loc[:, 'rating'] = extracted_data[1]
    df_moviesdata = df[df['request'].str.endswith('.mpg')]

    # try and except to see if the rating value in df_ratingsdata is an integer within 1 to 5, else delete the row
    try:
        df_ratingsdata['rating'] = df_ratingsdata['rating'].astype(int)
        df_ratingsdata = df_ratingsdata[df_ratingsdata['rating'].between(1, 5)]
    except Exception as e:
        print(f"Rating value in wrong format or range: {e}")
        df_ratingsdata = df_ratingsdata.drop(df_ratingsdata.index)

    print("--- Data Separated to Ratings and Movies ---")

    return df_ratingsdata, df_moviesdata

# Data Loading
filepath="kafka_data.csv"
movies_df = pd.read_csv(filepath)
#filepath="/home/team18/kafka_data.csv"
#movies_df = pd.read_csv(filepath)
df, _ = separate_ratings_and_movies(movies_df)
print("Data Loaded")

df = df[pd.notnull(df['movieid'])]

# Data preprocessing as before
user_ids = df["userid"].unique().tolist()
movie_ids = df["movieid"].unique().tolist()

user2user_encoded = {x: i for i, x in enumerate(user_ids)}
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

df["user"] = df["userid"].map(user2user_encoded)
df["movie"] = df["movieid"].map(movie2movie_encoded)

num_users = len(user_ids)
num_movies = len(movie_ids)

df['rating'] = df['rating'].values.astype(np.float32)
min_rating = min(df["rating"])
max_rating = max(df["rating"])

df = df.sample(frac=1, random_state=42)
x = df[["user", "movie"]].values
y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

    
model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
#current_date = datetime.now(ZoneInfo("America/New_York")).strftime('%Y-%m-%d')
#print(current_date)
weights_file = f"model_weights"
model.load_weights(weights_file)

pseudo_movie_ids = ['10+things+i+hate+about+you+1999', 'spriggan+1998', 'stargate+sg-1+children+of+the+gods+-+final+cut+2009', 'stargate+continuum+2008', 'star+wars+episode+iii+-+revenge+of+the+sith+2005', 'star+wars+episode+i+-+the+phantom+menace+1999', 'star+wars+1977', 'stanley+kubrick+a+life+in+pictures+2001', 'standing+up+2013', 'standing+in+the+shadows+of+motown+2002', 'stand+up+and+fight+1939', 'stalag+17+1953', 'spring+forward+1999', 'spread+2009', 'senna+2010', 'spontaneous+combustion+1990','spinning+plates+2013', 'spin+2007', 'spies+1928','the+matrix+1999', 'life+as+a+house+2001',"the+shawshank+redemption+1994","the+dark+knight+2008","inception+2010","raiders+of+the+lost+ark+1981","my+neighbor+totoro+1988","forrest+gump+1994","harry+potter+and+the+deathly+hallows+part+2+2011","monty+python+and+the+holy+grail+1975","the+lord+of+the+rings+the+return+of+the+king+2003","spirited+away+2001","the+godfather+1972","the+lord+of+the+rings+the+fellowship+of+the+ring+2001","fight+club+1999","nausica+of+the+valley+of+the+wind+1984","the+green+mile+1999","toy+story+1995","goodfellas+1990","the+dark+knight+rises+2012","seven+samurai+1954", 'the+big+clock+1948', 'matthews+days+1968', 'the+day+of+the+crows+2012', 'mansfield+park+1999', 'kes+1969', 'beauty+and+the+beast+1991', 'red+2008', 'the+dirty+dozen+1967', 'smile+1975', 'interstellar+2014', 'never+say+never+again+1983', 'good+night_+and+good+luck.+2005', 'steal+this+film+2006', 'scarface+1983', 'harry+potter+and+the+deathly+hallows+part+1+2010', 'the+little+mermaid+1989', 'you+and+i+2006', 'the+chronicles+of+narnia+prince+caspian+2008','crumb+1994', 'divorce+american+style+1967', 'cool+hand+luke+1967', 'chappie+2015', 'walle+2008', 'where+the+red+fern+grows+1974', '24+exposures+2013', 'rain+man+1988', 'hugo+2011', 'lost+in+la+mancha+2002', 'sergeant+york+1941', 'westward+the+women+1951']

app = Flask(__name__)


from joblib import load
from keras.utils import CustomObjectScope


@app.route('/recommend/<userid>', methods=['GET'])
def recommend(userid):
    userid = int(userid)

    encoded_user_id = user2user_encoded.get(userid)
    if encoded_user_id is None:
        # print(f"userid {userid} not found in user2user_encoded")
        
        # Randomly select 20 items
        selected_items = random.sample(pseudo_movie_ids, 20)
        # return selected_items
        
        data = [userid] + selected_items
        # Write the data to the CSV file
        with open('/home/team18/deploy/record.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)
            
        return ','.join(selected_items)
    # else:
    #     selected_items = random.sample(pseudo_movie_ids, 20)
    #     return ','.join(selected_items)
        

    # Get all movie IDs in the dataset
    all_movie_ids = list(movie2movie_encoded.keys())

    # Find movies that the user has already rated
    movies_rated_by_user = df[df['userid'] == userid]['movieid'].unique()

    # Find movies that the user has NOT rated yet
    movies_not_rated_by_user = [movie_id for movie_id in all_movie_ids if movie_id not in movies_rated_by_user]

    # Prepare the input for the model: a list of [user, movie] pairs for the given user and all movies they haven't rated
    user_movie_pairs = np.array([[encoded_user_id, movie2movie_encoded[movie_id]] for movie_id in movies_not_rated_by_user])

    # Predict ratings for these pairs
    predicted_ratings = model.predict(user_movie_pairs).flatten()

    # Combine the movie IDs and their predicted ratings
    movie_ratings_pairs = list(zip(movies_not_rated_by_user, predicted_ratings))

    # Sort the movies by their predicted ratings in descending order
    movie_ratings_pairs.sort(key=lambda x: x[1], reverse=True)

    # Extract the top 20 movies
    top_20_movies = movie_ratings_pairs[:20]
    
    movies = ','.join([str(x[0]) for x in top_20_movies])
    
    data = [userid] + top_20_movies
    # Write the data to the CSV file
    with open('/home/team18/deploy/record.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    
    return movies


    # recommendations = ','.join(pseudo_movie_ids)
    # return recommendations
    # return jsonify(pseudo_movie_ids)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
    
    
    
