#!/usr/bin/env python

from annoy import AnnoyIndex
from io import StringIO
import pandas as pd
import flask
from flask import Flask, Response
from simhash import Simhash
import re, time, os, json

model_dir = '/opt/ml/model'

dim = 64
dist = 'hamming'
topk = 25
print('hyperparameters parsed')

t = time.time()
# load the model    
model = AnnoyIndex(dim, dist)
model.load(os.path.join(model_dir, "test.ann")) # super fast, will just mmap the file
print(f'Index loaded in {1000*(time.time()-t):.2f} ms')

# serve the model
app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return Response(response="\n", status=200)

@app.route("/invocations", methods=["POST"])
def predict():
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO(data)
        print("input: ", s.getvalue())
        print(s)
        data = pd.read_csv(s,header=None)
        print(data)
        response = model.get_nns_by_vector(data.to_numpy()[0], topk, search_k=-1, include_distances=False)
        response = str(response)
        print("response: ", response)
    else:
        return flask.Response(response='CSV data only', status=415, mimetype='text/plain')

    return Response(response=response, status=200)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
