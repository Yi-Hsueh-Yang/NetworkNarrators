import pandas as pd
import requests
from data_collection.kafka_collection import datacollection
import numpy as np
import sys

pd.options.mode.chained_assignment = None

# Define the IP address of the server
ip_address = '128.2.204.215'
language_mapping = {
    'English': 'English', 'Español': 'Spanish', 'svenska': 'Swedish', 'Deutsch': 'German', 'Français': 'French',
    'Magyar': 'Hungarian', 'Latin': 'Latin', 'Italiano': 'Italian', '日本語': 'Japanese', '广州话 / 廣州話': 'Cantonese / Yue',
    '普通话': 'Mandarin', '': 'Unknown', 'العربية': 'Arabic', 'ελληνικά': 'Greek', 'Slovenčina': 'Slovak',
    '한국어/조선말': 'Korean', 'Pусский': 'Russian', 'Polski': 'Polish', 'No Language': 'No Language', 'Srpski': 'Serbian',
    'Afrikaans': 'Afrikaans', 'Português': 'Portuguese', 'Română': 'Romanian', 'suomi': 'Finnish', 'हिन्दी': 'Hindi',
    'Tiếng Việt': 'Vietnamese', 'Český': 'Czech', 'Türkçe': 'Turkish', 'اردو': 'Urdu', 'עִבְרִית': 'Hebrew',
    'فارسی': 'Persian', 'Íslenska': 'Icelandic', 'Esperanto': 'Esperanto', 'Nederlands': 'Dutch', 'ภาษาไทย': 'Thai',
    'Dansk': 'Danish', 'Kiswahili': 'Swahili', 'Somali': 'Somali', 'български език': 'Bulgarian', 'Gaeilge': 'Irish',
    'shqip': 'Albanian', 'Norsk': 'Norwegian', 'বাংলা': 'Bengali', 'Bahasa melayu': 'Malay', 'Bahasa indonesia': 'Indonesian',
    'Català': 'Catalan', 'Український': 'Ukrainian', 'Slovenščina': 'Slovenian', 'தமிழ்': 'Tamil', 'Hrvatski': 'Croatian',
    'ქართული': 'Georgian', 'isiZulu': 'Zulu', 'Bosanski': 'Bosnian', 'Wolof': 'Wolof', 'Kinyarwanda': 'Kinyarwanda',
    'Eesti': 'Estonian', 'پښتو': 'Pashto', 'ਪੰਜਾਬੀ': 'Punjabi', '?????': 'Unknown', 'తెలుగు': 'Telugu',
    'Cymraeg': 'Welsh', 'Lietuviakai': 'Lithuanian', 'Malti': 'Maltese', 'Azərbaycan': 'Azerbaijani', 'Bamanankan': 'Bambara'
}

def seperate_ratings_and_movies(df):
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

def extract_movieid_and_minutes_from_movies(df_movie):
    """
    Extracts movieid and minutes from the given dataframe and returns the count of unique movieids.
    """
    df_movie[['movieid', 'minutes']] = df_movie['request'].str.extract(r'/m/(.+?)/(\d+)\.mpg$')
    unqiuemovies = df_movie['movieid'].unique()
    
    print("--- MovieID and Minutes Extracted ---")

    return unqiuemovies

def extract_userid_from_kafka_requests(df):
    """
    Extracts unique user IDs from the input DataFrame and returns them in a new DataFrame.
    Parameters:
        df (DataFrame): The input DataFrame containing the user IDs.
    Returns:
        DataFrame: A new DataFrame containing the unique user IDs.
    """
    unique_user_ids = df['userid'].unique()
    
    print("--- User IDs Extracted ---")
    return unique_user_ids

# Function to query movie information and return as DataFrame
def get_movie_info(movieid):
    """
    Retrieves movie information based on the provided movie ID and returns the data as a pandas DataFrame.
    """
    response = requests.get(f'http://{ip_address}:8080/movie/{movieid}')
    if response.status_code == 200:
        movie_data = response.json()
        return pd.DataFrame([movie_data])
    else:
        print(f"Failed to fetch movie info for movieid {movieid}")
        return pd.DataFrame()

# Function to query user information and return as DataFrame
def get_user_info(userid):
    """
    Retrieve user information from the specified user ID and return it as a pandas DataFrame.
    """
    response = requests.get(f'http://{ip_address}:8080/user/{userid}')
    if response.status_code == 200:
        user_data = response.json()
        return pd.DataFrame([user_data])
    else:
        print(f"Failed to fetch user info for userid {userid}")
        return pd.DataFrame()

# Function to create a movie dataset
def create_movie_dataset(df_movie):
    """
    A function to create a movie dataset by querying movie info for each movieid in the DataFrame and returning the concatenated movie info DataFrame.
    """
    # Query movie info for each movieid in the DataFrame
    movie_info_dfs = []
    unqiuemovieslist = list(extract_movieid_and_minutes_from_movies(df_movie))

    for movieid in unqiuemovieslist:
        movie_info_dfs.append(get_movie_info(movieid))
    # Concatenate all movie info DataFrames into a single DataFrame
    movie_info_df = pd.concat(movie_info_dfs, ignore_index=True)

    print("--- Movie Dataset Ready ---")
    return movie_info_df

# Function to create a user dataset
def create_user_dataset(df_from_kafka):
    """
    Query user info for each userid in the DataFrame
    Concatenate all user info DataFrames into a single DataFrame
    """
    # Query user info for each userid in the DataFrame
    user_info_dfs = []
    unqiueusers = list(extract_userid_from_kafka_requests(df_from_kafka))
    for userid in unqiueusers:
        user_info_dfs.append(get_user_info(userid))
    # Concatenate all user info DataFrames into a single DataFrame
    user_info_df = pd.concat(user_info_dfs, ignore_index=True)

    print("--- User Dataset Ready ---")
    return user_info_df

# movie df preprocessing
def extract_values(row):
    """
    A function to extract the 'name' values from a list of dictionaries.
    Takes a list of dictionaries as input.
    Returns a list of 'name' values if successful, or None if an exception occurs.
    """
    try:
        return [d['name'] for d in row]
    except:
        return None
def extract_values_dict(row):
    """
    Function to extract the value associated with the 'name' key from a dictionary.
    Parameters:
    row (dict): The input dictionary.
    Returns:
    str or None: The value associated with the 'name' key, or None if the key is not present.
    """
    try:
        return row['name']
    except:
        return None

def movie_columns_manipulation(df):
    """
    Manipulates the columns of the provided DataFrame by extracting values from specified columns and performing data type conversions and replacements. Returns the modified DataFrame.
    """

    df['genres'] = df['genres'].apply(extract_values)
    df['production_companies'] = df['production_companies'].apply(extract_values)
    df['production_countries'] = df['production_countries'].apply(extract_values)
    df['belongs_to_collection'] = df['belongs_to_collection'].apply(extract_values_dict)
    df['spoken_languages'] = df['spoken_languages'].apply(extract_values)

    # map all languages into English version 
    for index, row in df.iterrows():
        updated_languages = []
        for language in row['spoken_languages']:
            updated_languages.append(language_mapping.get(language, language))
        df.at[index, 'spoken_languages'] = updated_languages

    mode_release_date = df['release_date'].mode()[0]  # Get the mode (most common release date)
    df['release_date'].replace('null', mode_release_date, inplace=True)

    df['tmdb_id'] = df['tmdb_id'].astype(str)
    df['release_date'] = pd.to_datetime(df['release_date'])

    df.belongs_to_collection.fillna('None', inplace=True)

    print("--- Movie Dataset Preprocessed and All Set ---")
    return df

def user_columns_manipulation(df):
    """
    Manipulates the user_id column of the input DataFrame by converting it to string type.
    Returns the modified DataFrame.
    """
    df.user_id = df.user_id.astype(str)

    print("--- User Dataset Preprocessed and All Set ---")
    return df

def main():
    time_running_kafka = sys.argv[1]
    kafka_data = datacollection(float(time_running_kafka))
    rating, movie = seperate_ratings_and_movies(kafka_data)

    movie_info = create_movie_dataset(movie)
    user_info = create_user_dataset(kafka_data)

    cleaned_movie = movie_columns_manipulation(movie_info)
    cleaned_user = user_columns_manipulation(user_info)

    # Movie Info
    print("The cleaned movie info DataFrame: ")
    print(np.shape(cleaned_movie))
    # print(cleaned_movie.head())
    # convert the data into json and save it with the name involving current time
    # cleaned_movie.to_json(f'{int(time.time())}_movieinfo.json', orient='records')

    # Ratings
    print("The cleaned rating DataFrame: ")
    print(np.shape(rating))
    # print(rating.head())
    # rating.to_json(f'{int(time.time())}_ratings.json', orient='records')

    # Movie Minutes
    print("The cleaned movie minutes DataFrame: ")
    print(np.shape(movie))
    # print(movie.head())
    # movie.to_json(f'{int(time.time())}_minutes.json', orient='records')

    # User
    print("The cleaned user DataFrame: ")
    print(np.shape(cleaned_user))
    # print(cleaned_user.head())
    # cleaned_user.to_json(f'{int(time.time())}_user.json', orient='records')

if __name__ == '__main__':
    main()
