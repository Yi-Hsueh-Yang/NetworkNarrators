import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
from datetime import datetime
import mlflow
from mlflow.models import infer_signature
import os

EMBEDDING_SIZE = 50
pipeline_version = "v1.2"
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
        x = dot_user_movie + user_bias + movie_bias
        return tf.nn.sigmoid(x)

def separate_ratings_and_movies(df):
    df_ratingsdata = df[df['request'].str.startswith('GET /rate')]
    extracted_data = df_ratingsdata.iloc[:, 2].str.extract(r'/rate/(.*?)=(\d+)')
    
    df_ratingsdata['movieid'] = extracted_data[0]
    df_ratingsdata['rating'] = extracted_data[1]
    df_moviesdata = df[df['request'].str.endswith('.mpg')]

    # Drop rows with NaN values in 'movieid' or 'rating' columns
    df_ratingsdata.dropna(subset=['movieid', 'rating'], inplace=True)

    # Convert 'rating' column to integer type
    df_ratingsdata['rating'] = df_ratingsdata['rating'].astype(int)

    # Filter out ratings outside the range [1, 5]
    df_ratingsdata = df_ratingsdata[df_ratingsdata['rating'].between(1, 5)]
    print(df_ratingsdata.head())

    return df_ratingsdata, df_moviesdata

def train_model():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:6001")
    # mlflow.set_tracking_uri("http://128.2.205.119:6001")
    
    # Set experiment name
    mlflow.set_experiment("Experimentmodel")

    
    filepath= "/home/team18/deploy/kafka_data.csv"
    
    movies_df = pd.read_csv(filepath)
    df_ratingsdata, _ = separate_ratings_and_movies(movies_df)

    user_ids = df_ratingsdata["userid"].unique()
    movie_ids = df_ratingsdata["movieid"].unique()

    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

    df_ratingsdata["user"] = df_ratingsdata["userid"].map(user2user_encoded)
    df_ratingsdata["movie"] = df_ratingsdata["movieid"].map(movie2movie_encoded)

    num_users = len(user_ids)
    num_movies = len(movie_ids)

    df_ratingsdata['rating'] = df_ratingsdata['rating'].values.astype(np.float32)
    min_rating = min(df_ratingsdata["rating"])
    max_rating = max(df_ratingsdata["rating"])

    df_ratingsdata = df_ratingsdata.sample(frac=1, random_state=42)
    x = df_ratingsdata[["user", "movie"]].values
    y = df_ratingsdata["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20)

    with mlflow.start_run():
        mlflow.log_param("embedding_size", EMBEDDING_SIZE)
        mlflow.log_param("num_users", num_users)
        mlflow.log_param("num_movies", num_movies)
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

       
    # Save large data to CSV files
        pd.DataFrame(x_train).to_csv(os.path.join(data_dir, "x_train.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(data_dir, "y_train.csv"), index=False)
        pd.DataFrame(x_val).to_csv(os.path.join(data_dir, "x_val.csv"), index=False)
        pd.DataFrame(y_val).to_csv(os.path.join(data_dir, "y_val.csv"), index=False)

        # Log artifacts (file paths)
        mlflow.log_artifacts(data_dir, artifact_path="data")
      
        # mlflow.log_param("X_train", x_train)
        # mlflow.log_param("y_train", y_train)
        # mlflow.log_param("X_test", x_val)
        # mlflow.log_param("y_test", y_val)

        model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.01))

        model.fit(
            x=x_train,
            y=y_train,
            batch_size=128,
            epochs=30,
            verbose=1,
            validation_data=(x_val, y_val)
        )

        #current_date = datetime.now().strftime('%Y-%m-%d')
        
        mlflow.log_param("pipeline_version", pipeline_version)
        weights_file = f"model_weights"
        weights_path=os.path.join("/home/team18/deploy/", weights_file)
        model.save_weights(weights_path)
        y_pred = model.predict(x_val).flatten()

        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse}")

        mlflow.log_metric("rmse", rmse)
        mlflow.keras.log_model(model, "recommender_model")

if __name__ == "__main__":
    train_model()
