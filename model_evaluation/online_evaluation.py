import sys
# sys.path.append('D:\\Spring 2024\\MLIP\\group-project-s24-network-narrators')
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
from data_collection.kafka_collection import datacollection
from data_collection.data_preprocessing import separate_ratings_and_movies

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

time_running_kafka = 120  # time in seconds for each collection
num_batches = 10

rmses = []

for batch_num in range(num_batches):
    print(f"Collecting data for batch {batch_num+1}/{num_batches}...")
    test_data = datacollection(float(time_running_kafka))
    df, _ = separate_ratings_and_movies(test_data)

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
    model.load_weights('model_weights')

    # Assume model is predefined and loaded as before
    y_pred = model.predict(x).flatten()

    # Calculate and store RMSE for the current batch
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    rmses.append(rmse)

    print(f"Batch {batch_num+1}/{num_batches} RMSE: {rmse}")

# Calculate and print the average RMSE over all batches
average_rmse = np.mean(rmses)
print("Average Root Mean Square Error across all batches:", average_rmse)
