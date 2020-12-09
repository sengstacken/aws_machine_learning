import argparse, os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from keras.utils import multi_gpu_model

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--embedding',type=str, default=os.environ['SM_CHANNEL_EMBEDDING'])
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    embedding_dir = args.embedding
    
    print('LOADING DATA....')
    
    # load the data
    x_train = np.load(os.path.join(training_dir, 'training.npz'))['text']
    y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
    x_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['text']
    y_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'test samples')
    
    embedding_matrix  = np.load(os.path.join(embedding_dir, 'embedding.npz'))['embedding']
    print('Embedding Matrix Shape:',embedding_matrix.shape)
    num_tokens = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    
    print('DATA LOADING COMPLETE')

    
    # configure Embedding Layer
    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )

    int_sequences_input = keras.Input(shape=(None,), dtype="int64")
    embedded_sequences = embedding_layer(int_sequences_input)
    x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(128, 5, activation="relu")(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(128, 5, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    preds = layers.Dense(20, activation="softmax")(x)
    model = keras.Model(int_sequences_input, preds)
    
    print(model.summary())

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
        
    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=1)
    
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])
    
    # save model to an S3 directory with version number '00000001'
    model.save(os.path.join(model_dir, '000000001'), 'my_model.h5')
        
#     sess = K.get_session()
#     tf.saved_model.simple_save(
#         sess,
#         os.path.join(model_dir, 'model/1'),
#         inputs={'inputs': model.input},
#         outputs={t.name: t for t in model.outputs})
    