import argparse
import os
import glob
import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import math
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, Input, Concatenate

# Check the version of TensorFlow 
print("TensorFlow v" + tf.__version__)


if __name__ == "__main__":

    ### ARGUMENTS
    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument("--embedding-dim", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)

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
    df_list = (pd.read_csv(file) for file in train_csv_files)
    train_df = pd.concat(df_list, ignore_index=True)
    train_df.columns = train_df.columns.astype(str)
    
    
    
    
    # read all data from test folder
    test_csv_files = glob.glob(args.train + "/*.csv")
    print(test_csv_files)
    df_list = (pd.read_csv(file) for file in test_csv_files)
    test_df = pd.concat(df_list, ignore_index=True)
    test_df.columns = test_df.columns.astype(str) 
    
    ### MODEL
    print("building model")

    # Add the input layers city and region 
    input1 = Input(shape=(2,))
    emb1 = Embedding(input_dim=400, output_dim=args.embedding_dim)(input1)
    flat1 = Flatten()(emb1)

    # for one hot encoded features
    input2 = Input(shape=(43,))
    emb2 = Embedding(input_dim=2, output_dim=args.embedding_dim)(input2)
    flat2 = Flatten()(emb2)

    # for ip addresses
    input3 = Input(shape=(3,))
    emb3 = Embedding(input_dim=256, output_dim=args.embedding_dim)(input3)
    flat3 = Flatten()(emb3)

    # Merge the inputs
    merged_layer = Concatenate()([flat1, flat2, flat3])

    # Add the output layer
    dense1 = Dense(256, activation='relu')(merged_layer)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(1, activation='sigmoid')(dense2)
    model = Model(inputs=[input1, input2, input3], outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()
    

    ### TRAINING
    print("training model")
    model.fit(
        x = [train_df.iloc[:,1:3].values, train_df.iloc[:,3:46].values, train_df.iloc[:,46:].values],
        y = train_df.iloc[:,0].values, 
        batch_size=args.batch_size,
        epochs=args.epochs, 
        validation_split=0.2,
        verbose=2,
        )

    ### EVALUATION
    print("post-training evaluation")
    loss, acc = model.evaluate(
        x = [test_df.iloc[:,1:3].values, test_df.iloc[:,3:46].values, test_df.iloc[:,46:].values], 
        y = test_df.iloc[:,0].values,
        verbose=2,
    )

    print(f"Test Accuracy: {acc:.5f}")

    ### SAVING
    print("saving model")
    # A version number is needed for the serving container to load the model
    version = "00000000"
    ckpt_dir = os.path.join(args.model_dir, version)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model.save(ckpt_dir)