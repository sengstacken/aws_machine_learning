
## Library Imports
import os
import pandas as pd
import time
import argparse
import logging
import glob
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import smdebug.tensorflow as smd

## Constants
TRAIN_VERBOSE_LEVEL = 2
EVALUATE_VERBOSE_LEVEL = 2

## Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.INFO)))

## Parse and load the command-line arguments sent to the script
## These will be sent by SageMaker when it launches the training container
def parse_args():
    logger.info('Parsing command-line arguments...')
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--batchsize', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=4e-3, help='learning rate (default: 4e-3)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    # Data directories
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val-dir', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # Model output directory
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    # Checkpoint info
    parser.add_argument('--checkpoint_enabled', type=str, default='False')
    parser.add_argument('--checkpoint_load_previous', type=str, default='False')
    parser.add_argument('--checkpoint_local_dir', type=str, default='/opt/ml/checkpoints/')
    
    logger.info('Completed parsing command-line arguments.')
    
    return parser.parse_known_args()

def set_seeds(seed):
    tf.random.set_seed(seed)

## Load data from local directory to memory and preprocess
def load_data(data_dir):
    
    # data
    headers = ['clicks', 'Region', 'City', 'Ad slot width', 'Ad slot height',
       'TimestampYear', 'TimestampMonth', 'TimestampWeek', 'TimestampDay',
       'TimestampDayofweek', 'TimestampDayofyear', 'TimestampIs_month_end',
       'TimestampIs_month_start', 'TimestampIs_quarter_end',
       'TimestampIs_quarter_start', 'TimestampIs_year_end',
       'TimestampIs_year_start', 'TimestampHour', 'TimestampMinute',
       'TimestampSecond', '10006', '10024', '10031',
       '10048', '10052', '10057', '10059', '10063', '10067', '10074', '10075',
       '10076', '10077', '10079', '10083', '10093', '10102', '10110', '10111',
       '10684', '11092', '11278', '11379', '11423', '11512', '11576', '11632',
       '11680', '11724', '11944', '13042', '13403', '13496', '13678', '13776',
       '13800', '13866', '13874', '14273', '16593', '16617', '16661', '16706',
       'ip1', 'ip2', 'ip3', 'adex_1', 'adex_2', 'adex_3', 'advis_0', 'advis_1',
       'advis_2', 'advis_255', 'adfmt_0', 'adfmt_1', 'adfmt_5']
    
    files = glob.glob(f'{data_dir}/*.csv')   
    logger.info(f'Files in {data_dir}: {files}')
    df = pd.read_parquet(data_dir)

    # load data
    y = df.pop('clicks')
    y = y.to_numpy(dtype='float32')
    x = df.to_numpy(dtype='float32')
    
    logger.info(f'data loaded from {data_dir}, shapes, input: {x.shape}, output: {y.shape}')
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.cache()
    ds = ds.shuffle(len(x))
    ds = ds.batch(args.batchsize)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    logger.info('Completed loading and preprocessing data.')
    # following best practices from https://www.tensorflow.org/datasets/performances
    return ds

## Construct the network
def create_model():
    logger.info('Creating the model...')
    model = Sequential([

        Dense(128, activation='relu', input_shape=(75,)),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')])
    # Print the model summary
    logger.info(model.summary())
    logger.info('Completed creating the model.')
    return model

## Load the weights from the latest checkpoint
def load_weights_from_latest_checkpoint(model):
    file_list = os.listdir(args.checkpoint_local_dir)
    logger.info('Checking for checkpoint files...')
    if len(file_list) > 0:
        logger.info('Checkpoint files found.')
        logger.info('Loading the weights from the latest model checkpoint...')
        model.load_weights(tf.train.latest_checkpoint(args.checkpoint_local_dir))
        logger.info('Completed loading weights from the latest model checkpoint.')
    else:
         logger.info('Checkpoint files not found.')    
            
## Compile the model by setting the optimizer, loss function and metrics
def compile_model(model, learning_rate):
    logger.info('Compiling the model...')
    
    # Instantiate the optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Instantiate the loss function
    loss_fn = tf.losses.BinaryCrossentropy(from_logits=True)
    
    # metrics
    METRICS = [
#       tf.keras.metrics.TruePositives(name='tp'),
#       tf.keras.metrics.FalsePositives(name='fp'),
#       tf.keras.metrics.TrueNegatives(name='tn'),
#       tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
#       tf.keras.metrics.Precision(name='precision'),
#       tf.keras.metrics.Recall(name='recall'),
#       tf.keras.metrics.AUC(name='auc'),
#       tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=METRICS)
    logger.info('Completed compiling the model.')
    return optimizer, loss_fn
    
## Save the model as a checkpoint
def save_checkpoint(checkpoint):
    logger.debug('Saving model checkpoint...')
    checkpoint.save(os.path.join(args.checkpoint_local_dir, 'tf2-checkpoint'))
    logger.info('Checkpoint counter = {}'.format(checkpoint.save_counter.numpy()))
    logger.debug('Completed saving model checkpoint.')    

## Save the model
def save_model(model, model_dir):
    logger.info('Saving the model...')
    tf.saved_model.save(model, model_dir)
    logger.info('Completed saving the model.')
                 
## Train the model
def train_model(model, model_dir, train_ds, valid_ds, batch_size, epochs, learning_rate):

    logger.info('Training the model...')
    if args.checkpoint_enabled.lower() == 'true':
        logger.info('Initializing to perform checkpointing...')
        checkpoint = ModelCheckpoint(filepath=os.path.join(args.checkpoint_local_dir, 'tf2-checkpoint-{epoch}'),
                                     save_best_only=False, save_weights_only=True,
                                     save_frequency='epoch',
                                     verbose=TRAIN_VERBOSE_LEVEL)
        callbacks = [checkpoint]
        logger.info('Completed initializing to perform checkpointing.')
    else:
        logger.info('Checkpointing will not be performed.')
        callbacks = None

    training_start_time = time.time()
    history = model.fit(train_ds, batch_size=batch_size,
                        epochs=epochs, shuffle=True,
                        validation_data=valid_ds, validation_freq=1,
                        callbacks=[smd.KerasHook.create_from_json_file()], verbose=TRAIN_VERBOSE_LEVEL)

    logger.debug('Completed iterating over epochs.')
    training_end_time = time.time()
    logger.info(f'Training duration = {(training_end_time - training_start_time):.2f} seconds')
    logger.info('Completed training the model.')   

## Evaluate the model
def evaluate_model(model, test_ds):
    logger.info('Evaluating the model...')
    test_loss, test_accuracy = model.evaluate(test_ds,
                                              verbose=EVALUATE_VERBOSE_LEVEL)
    logger.info('Test loss = {}'.format(test_loss))
    logger.info('Test accuracy = {}'.format(test_accuracy))
    logger.info('Completed evaluating the model.')
    return test_loss, test_accuracy
    
if __name__ == '__main__':
    
    logger.info('Executing the main() function...')
    logger.info(f'TensorFlow version : {tf.__version__}')
    
    # parse command line arguments
    args, _ = parse_args()
    logger.info(args)
    
    # set seeds
    set_seeds(args.seed)
    
    # load data
    train_ds = load_data(args.train_dir)
    val_ds = load_data(args.val_dir)
    
    # Create an instance of the model
    model = create_model()
    
    if args.checkpoint_load_previous.lower() == 'true':
        load_weights_from_latest_checkpoint(model)
        #checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
    
    compile_model(model, args.lr)

    # train model            
    train_model(model, args.model_dir, train_ds, val_ds, args.batchsize, args.epochs, args.lr)  
        
    # evaluate the model
    evaluate_model(model, val_ds)
    
    # Save the generated model
    model.save(os.path.join(args.model_dir, "000000001"))
    
    logger.info('Completed executing the main() function.')
