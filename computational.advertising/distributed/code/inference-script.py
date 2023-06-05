import argparse
import os
import glob
import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import tarfile
import time
import json

# Check the version of TensorFlow Decision Forests
print("Found TensorFlow Decision Forests v" + tfdf.__version__)

if __name__ == "__main__":
    
    INPUTFOLDER = '/opt/ml/processing/input/data'
    MODELFOLDER = '/opt/ml/processing/input/model'
    OUTPUTFOLDER = '/opt/ml/processing/output'

    ### ARGUMENTS
    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--multiprocessing", type=bool, default=False)

    args, _ = parser.parse_known_args()
    
    with open('/opt/ml/config/resourceconfig.json','r') as f:
        d = f.read()
        config = json.loads(d)
        
    print(config)
    
    ### DATA
    print("loading data")
    # read all data from data folder
    csv_files = glob.glob(INPUTFOLDER + "/*.csv")
    print(csv_files)
    df_list = (pd.read_csv(file,header=None) for file in csv_files)
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.astype(str)
    ds = tfdf.keras.pd_dataframe_to_tf_dataset(df,label='0')
      
    ### MODEL
    print("loading model")
    # open file
    file = tarfile.open(MODELFOLDER + "/model.tar.gz")
    # extracting file
    file.extractall(MODELFOLDER)
    file.close()
    
    loaded_model = tf.keras.models.load_model(MODELFOLDER + "/00000000/")
    
    ### EVALUATION
    print("inference")
    start = time.time()
    t=loaded_model.predict(
        ds,
        workers=args.n_workers,
        use_multiprocessing=args.multiprocessing,
        verbose=2
    )
    print(f'total time:  {time.time() - start: .4f} seconds')
    
    outdf = pd.DataFrame(list(zip(df.index.values,t.flatten().tolist())))
     
    ### SAVING
    print("saving data")
    outdf.to_csv(OUTPUTFOLDER + '/output_' + config['current_host'] + '.csv',index=False,header=False)