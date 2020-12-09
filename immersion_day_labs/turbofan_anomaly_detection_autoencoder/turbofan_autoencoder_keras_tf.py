import argparse, os
import numpy as np

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras import regularizers
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

if __name__ == '__main__':
        
    ## INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    
    ## DATA LOADING
    x_train = np.load(os.path.join(training_dir, 'train.npy'))
    x_val  = np.load(os.path.join(validation_dir, 'val.npy'))
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'test samples')
    
    ## MODEL ARCHITECTURE
    input_dim = x_train.shape[1]
    encoding_dim = int(input_dim / 2) - 1
    hidden_dim = int(encoding_dim / 2)

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(1e-6))(input_layer)
    encoder = Dense(hidden_dim, activation="tanh")(encoder)
    decoder = Dense(encoding_dim, activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='tanh')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    print(autoencoder.summary())

    if gpu_count > 1:
        autoencoder = multi_gpu_model(autoencoder, gpus=gpu_count)

    autoencoder.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')
    
    ## MODEL TRAINING
    history = autoencoder.fit(x_train, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_val,x_val),
                    verbose=1).history
    
    score = autoencoder.evaluate(x_val, x_val, verbose=0)
    print('Validation loss    :', score)
    
    ## MODEL SAVING - save Keras model for Tensorflow Serving
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': autoencoder.input},
        outputs={t.name: t for t in autoencoder.outputs})