from confluent_kafka import Consumer, KafkaError, TopicPartition
import time

# Configuration for your Kafka Consumer
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'YOUR_GROUP_ID',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(**conf)

def consume_logs(topic, start_offset, end_offset):
    # Assign the consumer to the specific topic and offset
    partition = TopicPartition(topic, 0, start_offset)  # Assuming partition 0, adjust as needed
    consumer.assign([partition])

    # Consume until the end offset
    while True:
        msg = consumer.poll(1.0)  # Adjust poll timeout as needed

        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # End of partition event
                break
            else:
                print(msg.error())
                break

        # Check if the current message's offset is within our desired range
        if msg.offset() > end_offset:
            break

        # Process your message here (e.g., print it or store it)
        print(msg.value().decode('utf-8'))

# Example usage
topic = 'movielog18'
today = time.time()
two_weeks_ago = today - (14 * 24 * 60 * 60)

# Assuming you've calculated 'start_offset' and 'end_offset' for the 5-minute window each day
for day in range(14):
    day_start = two_weeks_ago + (day * 24 * 60 * 60)
    # You need a method to calculate the start and end offsets for each 5-minute window
    start_offset, end_offset = calculate_offsets_for_window(day_start)
    consume_logs(topic, start_offset, end_offset)

consumer.close()
