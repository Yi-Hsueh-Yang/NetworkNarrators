import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import joblib
import time
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers



# ratings_df = pd.read_json("/Ratings_data.json")
# movies_df = pd.read_json("/Movies_data.json")



np.random.seed(42)
def load_data(ratings_file,movies_file):
    ratings_df = pd.read_json(ratings_file)
    movies_df = pd.read_json(movies_file)
    # Mapping user ID and movie ID to indices
    user_ids = ratings_df["userid"].unique().tolist()
    user2user_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_ids = ratings_df["movieid"].unique().tolist()
    movie2movie_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    
    # Adding user and movie indices to ratings dataframe
    ratings_df["user_idx"] = ratings_df["userid"].map(user2user_idx)
    ratings_df["movie_idx"] = ratings_df["movieid"].map(movie2movie_idx)
    
    return ratings_df, user2user_idx, movie2movie_idx

# Define the recommender model
def create_model(num_users, num_movies, embedding_size):
    user_input = keras.layers.Input(shape=(1,))
    movie_input = keras.layers.Input(shape=(1,))
    
    user_embedding = keras.layers.Embedding(num_users, embedding_size)(user_input)
    movie_embedding = keras.layers.Embedding(num_movies, embedding_size)(movie_input)
    
    # Concatenate user and movie embeddings
    combined = keras.layers.Concatenate()([user_embedding, movie_embedding])
    
    # Add dense layers for prediction
    x = keras.layers.Flatten()(combined)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(1)(x)
    
    model = keras.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(loss='mse', optimizer='adam')
    
    return model

# Train the model
def train_model(model, x_train, y_train, x_val, y_val, epochs=100, batch_size=64):
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history

# Evaluate the model
def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test).flatten()
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    return rmse, mae

def print_recommendations_for_user(model, user_id, user2user_idx,ratings_df,movie2movie_idx, usertop_n=20):
    print("Starting recommendation generation...")
    # Encode the user ID
    encoded_user_id = user2user_idx.get(user_id)
    if encoded_user_id is None:
        print(f"User ID {user_id} not found.")
        return
    
    # Get all movie IDs in the dataset
    all_movie_ids = list(movie2movie_idx.keys())

    # Find movies that the user has already rated
    movies_rated_by_user = ratings_df[ratings_df['userid'] == user_id]['movieid'].unique()

    # Find movies that the user has NOT rated yet
    movies_not_rated_by_user = [movie_id for movie_id in all_movie_ids if movie_id not in movies_rated_by_user]

    # Prepare the input for the model: a list of [user, movie] pairs for the given user and all movies they haven't rated
    user_movie_pairs = np.array([[encoded_user_id, movie2movie_idx[movie_id]] for movie_id in movies_not_rated_by_user])

    # Predict ratings for these pairs
    predicted_ratings = model.predict([user_movie_pairs[:, 0], user_movie_pairs[:, 1]]).flatten()

    # Combine the movie IDs and their predicted ratings
    movie_ratings_pairs = list(zip(movies_not_rated_by_user, predicted_ratings))

    # Sort the movies by their predicted ratings in descending order
    movie_ratings_pairs.sort(key=lambda x: x[1], reverse=True)

    # Extract the top recommended movies
    top_movies = movie_ratings_pairs[:usertop_n]

    # Print the top recommended movies for the user
    print(f"Top 20 recommended movies for user {user_id} are:")
    for movie_id, rating in top_movies:
        print(f"{movie_id}")

# Function to safely load data or return an empty DataFrame if the file is not found
def safe_load_data(file_path, default_data=None):
    if os.path.exists(file_path):
        return file_path
    else:
        print(f"Warning: {file_path} not found. Using default data.")
        

def main():
    # Load data
    ratings_data_path = "./utils/sample_ratings.json"
    movies_data_path =  "./utils/sample_movies.json"
    
    ratings_df = safe_load_data(ratings_data_path)
    movies_df = safe_load_data(movies_data_path)

    if ratings_df and movies_df:
        ratings_df, user2user_idx, movie2movie_idx = load_data(ratings_df, movies_df)

        # Split data into train and test sets
        train_ratio = 0.8
        train_size = int(len(ratings_df) * train_ratio)
        train_df = ratings_df[:train_size]
        test_df = ratings_df[train_size:]


        # Create model
        num_users = len(user2user_idx)
        num_movies = len(movie2movie_idx)
        embedding_size = 50
        model = create_model(num_users, num_movies, embedding_size)

        # Prepare data for training
        x_train = [train_df["user_idx"].values, train_df["movie_idx"].values]
        y_train = train_df["rating"].values
        x_test = [test_df["user_idx"].values, test_df["movie_idx"].values]
        y_test = test_df["rating"].values

        # Train model
        history = train_model(model, x_train, y_train, x_test, y_test)

        # Evaluate model
        rmse, mae = evaluate_model(model, x_test, y_test)
        print("Root Mean Squared Error:", rmse)
        print("Mean Absolute Error:", mae)

        # Plot loss vs epoch curve for train and test
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        # Print recommendations for a user
        user_id_to_recommend = 230411
        print_recommendations_for_user(model, user_id_to_recommend,user2user_idx, ratings_df, movie2movie_idx)

if __name__ == "__main__":
    main()