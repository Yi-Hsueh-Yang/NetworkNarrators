from kafka import KafkaConsumer
from prometheus_client import Counter, Histogram, start_http_server, Gauge
import logging
import threading
import time
import requests

# Set up the main logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main_logger')

# Create a separate logger for unhandled messages
unhandled_logger = logging.getLogger('unhandled_messages')
file_handler = logging.FileHandler('unhandled_messages.log')
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
unhandled_logger.addHandler(file_handler)
unhandled_logger.setLevel(logging.ERROR)

recommendations = {}
clicked_items = {}
lock = threading.Lock()

def process_recommendation(user_id, results):
    with lock:
        if user_id not in recommendations:
            recommendations[user_id] = set(results)
        else:
            recommendations[user_id].update(results)  # Update the set of recommended items
        IMPRESSIONS.labels(user_id=user_id).inc(len(results))

def process_click(user_id, item):
    with lock:
        if user_id in recommendations and item in recommendations[user_id]:
            if user_id not in clicked_items:
                clicked_items[user_id] = set()
            if item not in clicked_items[user_id]:
                clicked_items[user_id].add(item)
                CLICKS.labels(user_id=user_id).inc()

def update_ctr():
    while True:
        with lock:
            total_impressions = sum([IMPRESSIONS.labels(user_id=user_id)._value.get() for user_id in recommendations.keys()])
            total_clicks = sum([CLICKS.labels(user_id=user_id)._value.get() for user_id in recommendations.keys()])
        # print('total_impressions', total_impressions)
        # print('total_clicks', total_clicks)
        if total_impressions > 0:
            ctr = (total_clicks / total_impressions) * 100
            CTR_GAUGE.set(ctr)
        time.sleep(10)  # Update CTR every 10 seconds

def monitor_service_health():
    while True:
        try:
            response = requests.get("http://128.2.205.119:8082/recommend/12345")
            SERVICE_HEALTH.set(1 if response.status_code == 200 else 0)
        except requests.exceptions.RequestException:
            SERVICE_HEALTH.set(0)
        time.sleep(60)  # check every minute

topic = 'movielog18'
start_http_server(8765)

REQUEST_COUNT = Counter('request_count', 'Recommendation Request Count', ['http_status'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')
SERVICE_HEALTH = Gauge('service_health', 'Health of Recommendation Service')
IMPRESSIONS = Counter('impressions', 'Total number of recommendations made', ['user_id'])
CLICKS = Counter('clicks', 'Total number of clicks on recommended items', ['user_id'])
CTR_GAUGE = Gauge('click_through_rate', 'Click-through rate')

def main():
    
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',
        group_id=topic,
        enable_auto_commit=True,
        auto_commit_interval_ms=1000
    )

    ctr_thread = threading.Thread(target=update_ctr)
    ctr_thread.daemon = True
    ctr_thread.start()

    health_thread = threading.Thread(target=monitor_service_health)
    health_thread.daemon = True
    health_thread.start()

    for message in consumer:
        try:
            event = message.value.decode('utf-8')
            parts = event.split(',')

            # Process recommendation request messages
            if 'recommendation request' in parts[2]:
                user_id = parts[1]
                result_string = parts[5:-1]
                results = [parts[4].split(' ')[2]]
                [results.append(r.strip()) for r in result_string]

                process_recommendation(user_id, results)
                # print('recommendations:', recommendations)
                status = parts[3].strip().split(" ")[1]
                REQUEST_COUNT.labels(status).inc()

                # Updating request latency histogram
                time_taken = float(parts[-1].strip().split(" ")[0])
                REQUEST_LATENCY.observe(time_taken / 1000)

            elif '/data/' in event:
                item = parts[2].split('/')[3]
                user_id = parts[1]
                # print(user_id, item)
                process_click(user_id, item)

        except Exception as e:
            unhandled_logger.error(f"Error processing message: {str(e)}, Message Content: {message.value}", exc_info=True)
            # logging.error(f"Error occured while processing message: {str(e)}")
            # print(parts)

if __name__ == "__main__":
    main()
