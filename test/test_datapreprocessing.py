import pandas as pd
from data_collection.data_preprocessing import seperate_ratings_and_movies, create_movie_dataset, movie_columns_manipulation
from unittest.mock import Mock

def test_seperate_ratings_and_movies():
    # Simulate input DataFrame
    data = {
        'timestamp': ["2024-02-12T10:21:39", "2024-02-13T09:11:49"],
        'userid' : ["105728", "32618"],
        'request': ['GET /rate/frankenstein+1931=3', 'GET /data/m/the+fugitive+1993/19.mpg']
    }
    df = pd.DataFrame(data)

    ratings_df, movies_df = seperate_ratings_and_movies(df)

    assert not ratings_df.empty
    assert not movies_df.empty
    assert 'movieid' in ratings_df.columns
    assert 'rating' in ratings_df.columns

def test_movie_columns_manipulation():
    data = {
        'genres': [[{'name': 'Drama'}]],  # Nested structure as expected by the function
        'production_companies': [[{'name': 'SomeCompany'}]],
        'production_countries': [[{'name': 'SomeCountry'}]],
        'belongs_to_collection': [{'name': 'SomeCollection'}],
        'spoken_languages': [[{'name': 'English'}]],
        'tmdb_id': [123],
        'release_date': ['2021-01-01']
    }
    df = pd.DataFrame(data)

    processed_df = movie_columns_manipulation(df)

    assert isinstance(processed_df, pd.DataFrame)
    assert 'genres' in processed_df.columns  # Check for one or more specific columns to ensure manipulation occurred
    assert processed_df['release_date'].dtype == '<M8[ns]'  # Confirm that 'release_date' is datetime format
