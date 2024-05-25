import unittest
import numpy as np
import pandas as pd
import sys
from io import StringIO
import os

from models.pipeline import load_data, create_model, train_model, evaluate_model, print_recommendations_for_user

class TestMilestone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define the parameters for creating the model
        cls.num_users = 8132
        cls.num_movies = 5050
        cls.embedding_size = 50
        # Define sample data files
        base_path = os.path.dirname(__file__)
        cls.ratings_file = os.path.join(base_path, 'utils', 'sample_ratings.json')
        cls.movies_file = os.path.join(base_path, 'utils', 'sample_movies.json')
        # Create the model
        cls.model = create_model(cls.num_users, cls.num_movies, cls.embedding_size)

    def test_load_data(self):
        ratings_df, user2user_idx, movie2movie_idx = load_data(self.ratings_file, self.movies_file)
        self.assertIsInstance(ratings_df, pd.DataFrame)
        self.assertIsInstance(user2user_idx, dict)
        self.assertIsInstance(movie2movie_idx, dict)
        print("load function passed")
        

    def test_create_model(self):
        self.assertIsNotNone(self.model)
        print("create function passed")
        

    def test_train_model(self):
        x_train = [np.array([12,34,67,98,1889]), np.array([15, 1968, 200, 2002, 300])]
        y_train = np.array([4, 4, 5, 4, 2])
        x_val = [np.array([19,5,867,918,3990]), np.array([312, 10, 98, 766, 4])]
        y_val = np.array([4, 4, 5, 3, 4])
        history = train_model(self.model, x_train, y_train, x_val, y_val)
        self.assertIsNotNone(history)
        print("train function passed")

    def test_evaluate_model(self):
        x_test = [np.array([13, 14, 15]), np.array([16, 17, 18])]
        y_test = np.array([4, 5, 4])
        rmse, mae = evaluate_model(self.model, x_test, y_test)
        self.assertIsInstance(rmse, float)
        self.assertIsInstance(mae, float)
        print("eval function passed")

            
if __name__ == '__main__':
    unittest.main()
