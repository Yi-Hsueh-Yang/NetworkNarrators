import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Constants
DATA_PATH = 'kafka_data.csv'
API_REQUEST_COLUMN = 'request'
TIME_COLUMN = 'timestamp'

# Load telemetry data
def load_data(filepath):
    data = pd.read_csv(filepath)
    data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN])
    return data

# Filter the dataset
def filter_data(data, pattern):
    filtered_data = data[data[API_REQUEST_COLUMN].str.contains(pattern,  na=False)]
    return filtered_data

# Establish thresholds based on telemetry data
def establish_thresholds(data):
    # Calculate the mean and standard deviation for the request values
    mean_requests = data['numeric_request_value'].mean()
    std_requests = data['numeric_request_value'].std()
    # Set thresholds for anomaly detection
    lower_threshold = mean_requests - 2 * std_requests
    upper_threshold = mean_requests + 2 * std_requests
    return lower_threshold, upper_threshold

# Anomaly detection using statistical thresholds
def detect_anomalies_with_thresholds(data, lower_threshold, upper_threshold):
    data['threshold_anomaly'] = ((data['numeric_request_value'] < lower_threshold) | 
                                 (data['numeric_request_value'] > upper_threshold)).astype(int)
    return data

# Anomaly detection using Isolation Forest
def detect_anomalies_ml(data):
    model = IsolationForest(n_estimators=100, contamination=0.01)
    data['ml_anomaly'] = model.fit_predict(data[['numeric_request_value']])
    return data

# Visualize data with anomalies
def plot_anomalies(data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(data[TIME_COLUMN], data['numeric_request_value'], color='blue', label='Normal')
    threshold_anomalies = data[data['threshold_anomaly'] == 1]
    ml_anomalies = data[data['ml_anomaly'] == -1]
    plt.scatter(threshold_anomalies[TIME_COLUMN], threshold_anomalies['numeric_request_value'], color='orange', label='Threshold Anomaly')
    plt.scatter(ml_anomalies[TIME_COLUMN], ml_anomalies['numeric_request_value'], color='red', label='ML Anomaly')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('API Requests')
    plt.legend()
    plt.show()

# Main function
def main():
    raw_data = load_data(DATA_PATH)

    request_data = filter_data(raw_data, 'recommendation request 17645-team18.isri.cmu.edu:8082')
    historical_data = filter_data(raw_data, 'GET /data/m')

    request_data['numeric_request_value'] = request_data[API_REQUEST_COLUMN].str.extract('(\d+)$').astype(float)
    historical_data['numeric_request_value'] = historical_data[API_REQUEST_COLUMN].str.extract(r'/(\d+)\.mpg$').astype(float)
    historical_data = historical_data.dropna()

    lower_threshold_req, upper_threshold_req = establish_thresholds(request_data)
    lower_threshold_hist, upper_threshold_hist = establish_thresholds(historical_data)

    request_data = detect_anomalies_with_thresholds(request_data, lower_threshold_req, upper_threshold_req)
    historical_data = detect_anomalies_with_thresholds(historical_data, lower_threshold_hist, upper_threshold_hist)

    request_data = detect_anomalies_ml(request_data)
    historical_data = detect_anomalies_ml(historical_data)

    plot_anomalies(request_data, title='API Request Anomalies Over Time')
    plot_anomalies(historical_data, title='Historical Data Anomalies Over Time')

    print("Anomalies detected in request data at the following times:")
    print(request_data.loc[request_data['threshold_anomaly'] == 1, TIME_COLUMN])
    print(request_data.loc[request_data['ml_anomaly'] == -1, TIME_COLUMN])

    print("Anomalies detected in historical data at the following times:")
    print(historical_data.loc[historical_data['threshold_anomaly'] == 1, TIME_COLUMN])
    print(historical_data.loc[historical_data['ml_anomaly'] == -1, TIME_COLUMN])

if __name__ == "__main__":
    main()
