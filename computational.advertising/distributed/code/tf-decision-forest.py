import argparse
import os
import glob
import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import math

# Check the version of TensorFlow Decision Forests
print("Found TensorFlow Decision Forests v" + tfdf.__version__)


if __name__ == "__main__":

    ### ARGUMENTS
    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    args, _ = parser.parse_known_args()
    
    ### DATA
    print("loading data")
    # read all data from training folder
    train_csv_files = glob.glob(args.train + "/*.csv")
    print(train_csv_files)
    df_list = (pd.read_csv(file,header=None) for file in train_csv_files)
    train_df = pd.concat(df_list, ignore_index=True)
    train_df.columns = train_df.columns.astype(str)
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df,label='0')
    
    # read all data from test folder
    test_csv_files = glob.glob(args.test + "/*.csv")
    print(test_csv_files)
    df_list = (pd.read_csv(file,header=None) for file in test_csv_files)
    test_df = pd.concat(df_list, ignore_index=True)
    test_df.columns = test_df.columns.astype(str) 
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label='0')
    
    ### MODEL
    print("building model")
    model_1 = tfdf.keras.RandomForestModel(verbose=2)
    
    ### TRAINING
    print("training model")
    model_1.fit(train_ds)
    
    ### EVALUATION
    print("post-training evaluation")
    model_1.summary()
    model_1.compile(metrics=["accuracy"])
    evaluation = model_1.evaluate(test_ds, return_dict=True)
    print(f"Test Accuracy: {evaluation['accuracy']:.5f}")
    
    ### SAVING
    print("saving model")
    # A version number is needed for the serving container to load the model
    version = "00000000"
    ckpt_dir = os.path.join(args.model_dir, version)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model_1.save(ckpt_dir)
    
    # Load a model: it loads as a generic keras model.
    # loaded_model = tf.keras.models.load_model("/tmp/my_saved_model")