import pandas as pd
from confluent_kafka import Consumer, TopicPartition
import logging
from datetime import datetime

def load_recommendation_data(start_percentage, end_percentage):
    rec_df = pd.read_csv('../deploy/recommendations.csv')
    rec_df['timestamp'] = pd.to_datetime(rec_df['timestamp'])
    rec_df = rec_df.sort_values(by='timestamp')

    start_index = int(len(rec_df) * float(start_percentage))  
    end_index = int(len(rec_df) * float(end_percentage))  

    filtered_df = rec_df.iloc[start_index:end_index]
    from_time = int(filtered_df['timestamp'].min().timestamp() * 1000)
    to_time = int(filtered_df['timestamp'].max().timestamp() * 1000)
    user_id_list = set(filtered_df['userid'])
    from_time = int(filtered_df['timestamp'].min().timestamp() * 1000)
    to_time = int(filtered_df['timestamp'].max().timestamp() * 1000)
    filtered_df.to_csv('filtered_recommendation.csv', index=False)
    return filtered_df, from_time, to_time, user_id_list

def process_click_df(user_id, movie_id, click_data_frame):
    # Append a new row to the DataFrame
    new_row = {'user_id': user_id, 'movie_id': movie_id}
    return click_data_frame._append(new_row, ignore_index=True)

def get_data(start_percentage, end_percentage):
    rec_df, from_time, to_time, user_id_list = load_recommendation_data(start_percentage, end_percentage)
    click_df = pd.DataFrame(columns=['user_id', 'movie_id'])  # Initialize DataFrame
    print(from_time, to_time)
    topic = 'movielog18'
    partition = 0
    tp = TopicPartition(topic, partition, -1)  

    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'my-group',
        'auto.offset.reset': 'earliest'
    })

    # Fetch the offset corresponding to the start time
    tp.offset = from_time
    offsets = consumer.offsets_for_times([tp], timeout=5000)
    
    if offsets[0].offset is not None:
        tp.offset = offsets[0].offset
    else:
        print(f"No offset found for time {from_time}, using earliest available.")
        tp.offset = consumer.position(tp)
    
    consumer.assign([tp])
    try:
        while True:
            message = consumer.poll(timeout=1.0)
            if message is None:
                continue
            if message.error():
                print(f"Error: {message.error()}")
                continue
            if message.timestamp()[1] >= to_time:
                break

            event = message.value().decode('utf-8')
            parts = event.split(',')

            if '/data/' in event:
                # print(f"Received message at {message.timestamp()}: {message.value().decode('utf-8')}")
                movie_id = parts[2].split('/')[3]
                user_id = parts[1]
                if user_id in user_id_list:  
                    try:              
                        click_df = process_click_df(int(user_id), movie_id, click_df)
                    except:
                        print(f"Wrong Data Type: {user_id}")
    finally:
        consumer.close()
    
    print('click_df:', click_df.shape)
    # click_df.to_csv('click_df.csv', index=False)
    return click_df

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     main()

