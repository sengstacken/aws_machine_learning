#!/usr/bin/env python

from simhash import Simhash
import pandas as pd
import re, time, os, json
import numpy as np
from annoy import AnnoyIndex


# character n-grams
def get_features(s):
    width = 3
    s = s.lower()
    s = re.sub(r'[^\w]+', '', s)
    return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]

if __name__ == '__main__':
    
    # dirs
    config_dir = '/opt/ml/input/config'
    training_dir = '/opt/ml/input/data/training'
    model_dir = '/opt/ml/model'
    
    # read hyperparameters
    with open(os.path.join(config_dir,'hyperparameters.json')) as f:
        hp = json.load(f)
        dim = int(hp['dimension'])
        dist = hp['distance']
        topk = int(hp['topk'])
        numtrees = int(hp['numtrees'])


    print('hyperparameters:',dim,dist,topk,numtrees)
    print('hyperparameters parsed')
    
    # read data
    t = time.time()
    data = np.load(os.path.join(training_dir,'data.npz'),allow_pickle=True)['data']
    print(f'Data loaded from .npz in {(time.time()-t):.2f} seconds')
    
    # convert to simhash
    t = time.time()
    text_hash = [Simhash(get_features(str(k))).value for k in data]
    print(f'Data converted to Simhash in {(time.time()-t):.2f} seconds')
    
    # build AnnoyIndex
    t = time.time()
    m = AnnoyIndex(dim, dist)  # Length of item vector that will be indexed  # "angular", "euclidean", "manhattan", "hamming", or "dot".
    for i in range(len(text_hash)):
        v = format(int(text_hash[i]), f'0{dim}b')
        m.add_item(i, [int(char) for char in v])
    m.build(numtrees,n_jobs=-1)  
    print(f'Annoy Index built in {(time.time()-t):.2f} seconds')
        
    # save index ("model")    
    m.save(os.path.join(model_dir,'test.ann'))
    print('Model Saved')