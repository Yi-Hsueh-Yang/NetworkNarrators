import pandas as pd
from kafka import KafkaConsumer
import time
import os
from datetime import datetime


def save_data(df_ratingsdata, date):
    """
    Saves the ratings data to a CSV file with the given date appended to its filename.
    """
    
    #if not df_ratingsdata.empty:
    file_name = f'kafka_data_{date}.csv'
    filepath=os.path.join("/home/team18",file_name)
    df_ratingsdata.to_csv(filepath, index=False)
    print(f"--- Data for {date} saved to {file_name} ---")
    #else:
        #print("--- No data to save ---")

def datacollection(kafka_broker='localhost:9092', topic_name='movielog18'):
    """
    Collects data from a Kafka topic for a specified amount of time and returns a pandas DataFrame containing the collected data.
    If collection time is more than 5 mins then function saves a csv file in the current repository with the timestamp
    Parameters:
    - collection_time: int, the duration in seconds to collect data
    - kafka_broker: str, the address of the Kafka broker (default='localhost:9092')
    - topic_name: str, the name of the Kafka topic to consume data from (default='movielog18')

    Returns:
    - pandas DataFrame: a DataFrame containing the collected data with columns ['timestamp', 'userid', 'request']
    """

    # Creating Kafka consumer
    try:
        consumer = KafkaConsumer(topic_name, bootstrap_servers=kafka_broker)
    except Exception as e:
        print(f"Failed to create Kafka consumer: {e}")
        return pd.DataFrame()
    # Initializing an empty list to store the data
    data_list = []

    # Consume messages for the specified collection time
    start_time = time.time()
    last_date = datetime.now().strftime('%Y-%m-%d')
    
    for message in consumer:
        try:
            # Parse the message
            parts = message.value.decode('utf-8').split(',')
            timestamp = pd.to_datetime(parts[0], errors='coerce')
            userid = int(parts[1])
            request = parts[2]
            if pd.isnull(timestamp) or pd.isnull(userid):
                continue
            data_list.append({
                "timestamp": timestamp,
                "userid": userid,
                "request": request
            })
            # Check if the date has changed
            current_date = datetime.now().strftime('%Y-%m-%d')
            if current_date != last_date:
                df = pd.DataFrame(data_list)
                #rating, movie = separate_ratings_and_movies(df)
                save_data(df, last_date)
                # Reset data collection for the new day
                data_list = []
                last_date = current_date

        except Exception as e:
            print(f"Failed to process message: {e}")

    # After exiting the loop, save any remaining data for the current day
    if data_list:
        df = pd.DataFrame(data_list)
        #rating, movie = separate_ratings_and_movies(df)
        save_data(df, last_date)

    # Close the consumer
    try:
        consumer.close()
    except Exception as e:
        print(f"Failed to close Kafka consumer: {e}")

    print("--- Data Collection From Kafka Complete! ---")

if __name__ == '__main__':
    datacollection()
