import pandas as pd
from kafka import KafkaConsumer
import time
from datetime import datetime
import re

def datacollection(collection_time, kafka_broker='localhost:9092', topic_name='movielog18'):
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
    for message in consumer:
        if time.time() - start_time >= collection_time:
            break
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
        except Exception as e:
            print(f"Failed to process message: {e}")

    # Close the consumer
    try:
        consumer.close()
    except Exception as e:
        print(f"Failed to close Kafka consumer: {e}")

    try:
        # Create a pandas DataFrame from the collected data
        df = pd.DataFrame(data_list)
    except Exception as e:
        print(f"Failed to process data into DataFrame: {e}")
        return pd.DataFrame()

    # Save the DataFrame as a CSV file with current date appended to file name
    if (collection_time>300):
        current_date = datetime.now().strftime('%Y-%m-%d')
        file_name = f'kafka_data_{current_date}.csv'
        df.to_csv(file_name, index=False)

    print("--- Data Collection From Kafka Complete! ---")
    return df
