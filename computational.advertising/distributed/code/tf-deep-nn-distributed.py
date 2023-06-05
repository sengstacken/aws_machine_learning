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

import smdistributed.dataparallel.tensorflow as sdp
sdp.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[sdp.local_rank()], 'GPU')

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
    #parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    args, _ = parser.parse_known_args()
    
    ### DATA
    print("loading data")
    # read all data from training folder
    train_csv_files = glob.glob(args.train + "/*.csv")
    print(train_csv_files)
    df_list = (pd.read_csv(file) for file in train_csv_files)  
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.astype(str)
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df,label='0')
    train_ds = train_ds.batch(args.batch_size)
   
     # read all data from test folder
#     test_csv_files = glob.glob(args.train + "/*.csv")
#     print(test_csv_files)
#     df_list = (pd.read_csv(file) for file in test_csv_files)
#     df = pd.concat(df_list, ignore_index=True)
#     df.columns = df.columns.astype(str)
#     test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df,label='0')
#     test_ds = test_ds.batch(args.batch_size)
    
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
    
    #loss = tf.keras.losses.binary_crossentropy()
    loss = tf.keras.losses.BinaryCrossentropy()
    
    # Scale Learning Rate
    # LR for 8 node run : 0.000125
    # LR for single node run : 0.001
    opt = tf.optimizers.Adam(0.000125 * sdp.size())
    
    ### SAVING
    print("saving model")
    # A version number is needed for the serving container to load the model
    version = "00000000"
    ckpt_dir = os.path.join(args.model_dir, version)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    @tf.function
    def training_step(X, y, first_batch):
        with tf.GradientTape() as tape:
            probs = model(X, training=True)
            loss_value = loss(y, probs)

        # Wrap tf.GradientTape with the library's DistributedGradientTape
        tape = sdp.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        if first_batch:
            # Broadcast model and optimizer variables
            sdp.broadcast_variables(model.variables, root_rank=0)
            sdp.broadcast_variables(opt.variables(), root_rank=0)

        return loss_value

    for epoch in range(args.epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        for batch, (Xb, yb) in enumerate(train_ds.take(10000 // sdp.size())):
            loss_value = training_step(Xb, yb, batch == 0)

            if batch % 50 == 0 and sdp.rank() == 0:
                print("Step #%d\tLoss: %.6f" % (batch, loss_value))

            # SMDataParallel: Save checkpoints only from master node.
            if sdp.rank() == 0:
                model.save(os.path.join(checkpoint_dir, "1"))

        # Save checkpoints only from master node.
        if sdp.rank() == 0:
            model.save(ckpt_dir)

        