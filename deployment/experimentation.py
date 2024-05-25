import datetime
from flask import Flask, Response, request
import itertools
import requests
from splitio import get_factory
from splitio.exceptions import TimeoutException
import sys
import os
import csv
import datetime

# Load Balancer that splits incoming traffic as per configurations in split.io for experimentation purposes

# Initialize Flask app
app = Flask(__name__)

# Define backend servers
servers = {
    'on': "http://localhost:8504",  # Server for treatment 'on'
    'off': "http://localhost:8505"  # Server for treatment 'off'
}

# Initialize Split.io SDK
factory = get_factory('vinu3o116gk74me7so1jr55mrcuraiev0e74')
try:
    factory.block_until_ready(5)
except TimeoutException:
    print("The Split.io SDK failed to initialize in 5 seconds. Exiting.")
    sys.exit(1)

split = factory.client()

def is_server_up(url):
    """Check if the server is up and responding to requests."""
    try:
        response = requests.get(url + '/recommend/12345')
        print(response.status_code)
        print(response.content)
        return response.status_code == 200
    except:
        return False

@app.route('/recommend/<userid>', methods=['GET'])
def recommend(userid):
    # Determine which treatment to use for this user
    treatment = split.get_treatment(userid, 'movie-recommendation')
    
    # Select the server based on the treatment, default to 'off' if treatment is unknown
    server_url = servers.get(treatment, servers['off'])

    print(server_url)
    
    # Check if the selected server is up
    if not is_server_up(server_url):
        return Response("Server is down", status=503)
    
    # Forward the request to the selected backend server
    response = requests.get(server_url + '/recommend/' + userid)

    #Get recommendations data (assuming it's in JSON format)
    recommendations = response.content

    with open('recommendations.csv', 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'server_url', 'userid', 'recommendations']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:  # If file is empty, write header row
            writer.writeheader()

        writer.writerow({
            'timestamp': datetime.datetime.now().isoformat(),
            'server_url': server_url,
            'userid': userid,
            'recommendations': recommendations
        })

    
    return Response(response.content, response.status_code, response.headers.items())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8503)
