import mlflow
from flask import Flask, request
import pandas as pd
import numpy as np
import tempfile
import time
import random
import os
from datetime import datetime
from tensorflow.keras.models import load_model

mlflow.set_tracking_uri("http://127.0.0.1:6001")

# Define the experiment name and run ID where the model is logged
experiment_name = "Experimentmodel"
run_id = "09f65e554eba4e3aa4f82c22a7c45ec8"  # run ID

# Load model as a PyFuncModel
logged_model = f"runs:/{run_id}/recommender_model"
loaded_model = mlflow.keras.load_model(logged_model)



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

pseudo_movie_ids = ['10+things+i+hate+about+you+1999', 'spriggan+1998', 'stargate+sg-1+children+of+the+gods+-+final+cut+2009', 'stargate+continuum+2008', 'star+wars+episode+iii+-+revenge+of+the+sith+2005', 'star+wars+episode+i+-+the+phantom+menace+1999', 'star+wars+1977', 'stanley+kubrick+a+life+in+pictures+2001', 'standing+up+2013', 'standing+in+the+shadows+of+motown+2002', 'stand+up+and+fight+1939', 'stalag+17+1953', 'spring+forward+1999', 'spread+2009', 'senna+2010', 'spontaneous+combustion+1990','spinning+plates+2013', 'spin+2007', 'spies+1928','the+matrix+1999', 'life+as+a+house+2001',"the+shawshank+redemption+1994","the+dark+knight+2008","inception+2010","raiders+of+the+lost+ark+1981","my+neighbor+totoro+1988","forrest+gump+1994","harry+potter+and+the+deathly+hallows+part+2+2011","monty+python+and+the+holy+grail+1975","the+lord+of+the+rings+the+return+of+the+king+2003","spirited+away+2001","the+godfather+1972","the+lord+of+the+rings+the+fellowship+of+the+ring+2001","fight+club+1999","nausica+of+the+valley+of+the+wind+1984","the+green+mile+1999","toy+story+1995","goodfellas+1990","the+dark+knight+rises+2012","seven+samurai+1954", 'the+big+clock+1948', 'matthews+days+1968', 'the+day+of+the+crows+2012', 'mansfield+park+1999', 'kes+1969', 'beauty+and+the+beast+1991', 'red+2008', 'the+dirty+dozen+1967', 'smile+1975', 'interstellar+2014', 'never+say+never+again+1983', 'good+night_+and+good+luck.+2005', 'steal+this+film+2006', 'scarface+1983', 'harry+potter+and+the+deathly+hallows+part+1+2010', 'the+little+mermaid+1989', 'you+and+i+2006', 'the+chronicles+of+narnia+prince+caspian+2008','crumb+1994', 'divorce+american+style+1967', 'cool+hand+luke+1967', 'chappie+2015', 'walle+2008', 'where+the+red+fern+grows+1974', '24+exposures+2013', 'rain+man+1988', 'hugo+2011', 'lost+in+la+mancha+2002', 'sergeant+york+1941', 'westward+the+women+1951']

app = Flask(__name__)

def log_inference_to_mlflow(userid, recommendations, start_time):
    # This helper function logs an inference to MLflow
    with mlflow.start_run(run_name="inference"):
        # Log the user ID as a parameter
        mlflow.log_param("userid", userid)
        
        # Create a unique filename for the recommendations artifact
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"recommendations_{timestamp}_{userid}_{experiment_name}_{run_id}.txt"
        recommendations_path = os.path.join(tempfile.gettempdir(), filename)

        # Write recommendations to the file
        with open(recommendations_path, "w") as f:
            f.write(recommendations)
            
        # Log the file as an artifact
        mlflow.log_artifact(recommendations_path, "recommendations")
        
        # Log the time taken for the inference as a metric
        mlflow.log_metric("inference_time", time.time() - start_time)
        
        # Clean up the temporary file
        os.remove(recommendations_path)

@app.route('/recommend/<userid>', methods=['GET'])
def recommend(userid):
    start_time = time.time()  # To log the inference time
    userid = int(userid)  # Ensure userid is an integer

    encoded_user_id = user2user_encoded.get(userid)
    if encoded_user_id is None:
        # User ID not found, return default recommendations
        selected_items = random.sample(pseudo_movie_ids, 20)
        recommendations = ','.join(selected_items)
        log_inference_to_mlflow(userid, recommendations, start_time)
        return recommendations

    all_movie_ids = list(movie2movie_encoded.keys())
    movies_rated_by_user = df[df['userid'] == userid]['movieid'].unique()
    movies_not_rated_by_user = [movie_id for movie_id in all_movie_ids if movie_id not in movies_rated_by_user]

    user_movie_pairs = np.array([[encoded_user_id, movie2movie_encoded[movie_id]] for movie_id in movies_not_rated_by_user])
    predicted_ratings = loaded_model.predict(user_movie_pairs).flatten()
    movie_ratings_pairs = list(zip(movies_not_rated_by_user, predicted_ratings))
    movie_ratings_pairs.sort(key=lambda x: x[1], reverse=True)

    top_20_movies = movie_ratings_pairs[:20]
    recommendations = ','.join([str(x[0]) for x in top_20_movies])
    log_inference_to_mlflow(userid, recommendations, start_time)

    return recommendations

if __name__ == '__main__':
    
    # mlflow.set_tracking_uri("host.docker.internal:6001") 
    # mlflow.set_tracking_uri("http://128.2.205.119:6001")  # Set MLFlow 
    mlflow.set_tracking_uri("http://127.0.0.1:6001")
    mlflow.set_experiment(experiment_name)  # Define the MLFlow experiment
    app.run(host='0.0.0.0', port=8088)
