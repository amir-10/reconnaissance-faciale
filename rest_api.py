from flask import Flask, request, Response# loading in Flask
import json
import os


# creating a Flask application
app = Flask(__name__)


# creating user url and only allowing post requests.
@app.route('/newUser', methods=['POST'])
def new():

    js = [ { "good" : 0.8, "bad" : 0.2 } ]
    os.system('python3 request.py')
    return Response(json.dumps(js),  mimetype='application/json')

if __name__ == '__main__':
    app.run(port=3000, debug=True)