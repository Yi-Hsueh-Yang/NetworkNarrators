from flask import Flask, Response
import itertools
import requests

app = Flask(__name__)

servers = ["http://localhost:8081", "http://localhost:8083"]
cycle = itertools.cycle(servers)

def is_server_up(url):
    try:
        response = requests.get(url + '/recommend/12345')
        return response.status_code == 200
    except:
        return False

@app.route('/recommend/<userid>', methods=['GET'])
def recommend(userid):
    server = next(cycle)
    while not is_server_up(server):
        server = next(cycle)
    response = requests.get(server + '/recommend/' + userid)
    return Response(response.content, response.status_code, response.headers.items())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8082)