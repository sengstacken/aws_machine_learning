# Import SMDataParallel TensorFlow2 Modules
#import smdistributed.dataparallel.tensorflow as dist

## Library Imports
import os
import pandas as pd
import sys
import time
import argparse
import logging
import smdebug
import smdebug.tensorflow as smd
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model, Input
from tensorflow.train import Checkpoint
#from tensorflow.keras.callbacks import TensorBoard

## Constants
TRAIN_VERBOSE_LEVEL = 0
EVALUATE_VERBOSE_LEVEL = 0

## Configure the logger
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.INFO)))
fh = logging.FileHandler('/opt/ml/output/traininglogs.log')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)

## Parse and load the command-line arguments sent to the script
## These will be sent by SageMaker when it launches the training container
def parse_args():
    logger.info('Parsing command-line arguments...')
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N', help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR', help='learning rate (default: 4e-3)')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')

    # Data directories
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val-dir', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # Model output directory
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    # Checkpoint info
    parser.add_argument('--checkpoint_enabled', type=str, default='False')
    parser.add_argument('--checkpoint_load_previous', type=str, default='False')
    parser.add_argument('--checkpoint_local_dir', type=str, default='/opt/ml/checkpoints/')
    
    parser.add_argument('--tf-logs-path', type=str, help="Path used for writing TensorFlow logs. Can be S3 bucket.")
    logger.info('Completed parsing command-line arguments.')
    
    return parser.parse_known_args()

## Initialize the SMDebugger for the Tensorflow framework
def init_smd():
    logger.info('Initializing the SMDebugger for the Tensorflow framework...')
    # Use KerasHook - the configuration file will be copied to /opt/ml/input/config/debughookconfig.json
    # automatically by SageMaker when it launches the training container
    hook = smd.KerasHook.create_from_json_file()
    logger.info('Debugger hook collections :: {}'.format(hook.get_collections()))
    logger.info('Completed initializing the SMDebugger for the Tensorflow framework.')
    return hook

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
    
    # load data
    df = pd.read_csv(f'{data_dir}/test.csv',names=headers)
    y = df.pop('clicks')
    y = y.to_numpy()
    x = df.to_numpy()
    y = y.astype('float32')
    x = x.astype('float32')
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x),seed=args.seed, reshuffle_each_iteration=True).batch(args.batch_size)
    logger.info('Completed loading and preprocessing data.')
    return ds

## Construct the network
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.di = Dense(128, activation='relu', input_shape=(75,))
        self.d = Dense(128, activation='relu')
        self.do1 = Dropout(0.25)
        self.do2 = Dropout(0.5)
        self.o = Dense(1)

    def call(self, x):
        x = self.di(x)
        x = self.d(x)
        x = self.do1(x)
        x = self.d(x)
        x = self.do2(x)
        x = self.o(x)
        return x

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
def compile_model(hook, model, learning_rate):
    logger.info('Compiling the model...')
    
    # Instantiate the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    # SMDebugger: Wrap the optimizer to retrieve gradient tensors
    optimizer = hook.wrap_optimizer(optimizer)
    # Instantiate the loss function
    loss_fn = tf.losses.BinaryCrossentropy(from_logits=True)
    # Prepare the metrics.
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    
#    tb_callback = TensorBoard('/opt/ml/output/tensorboard/')
#    tb_callback.set_model(model) # Writes the graph to tensorboard summaries using an internal file writer

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=[train_acc_metric])
    logger.info('Completed compiling the model.')
    return optimizer, loss_fn, train_acc_metric, val_acc_metric, train_loss, val_loss

## Prepare the batch datasets
# def prepare_batch_datasets(x_train, y_train, batch_size):
#     logger.info('Preparing train and validation datasets for batches...')
#     # Reserve the required samples for validation
#     x_val = x_train[-(len(x_train) * int(VALIDATION_DATA_SPLIT)):]
#     y_val = y_train[-(len(y_train) * int(VALIDATION_DATA_SPLIT)):]
#     # Prepare the training dataset with shuffling
#     train_dataset = Dataset.from_tensor_slices((x_train, y_train))
#     train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
#     # Prepare the validation dataset
#     val_dataset = Dataset.from_tensor_slices((x_val, y_val))
#     val_dataset = val_dataset.batch(batch_size)
#     logger.info('Completed preparing train and validation datasets for batches.')
#     return x_val, y_val, train_dataset, val_dataset
    
def set_seeds(seed):
    tf.random.set_seed(seed)
    
# training function
@tf.function
def training_step(hook, model, x_batch, y_batch, opt, loss_fn, train_loss, train_acc_metric):
    with hook.wrap_tape(tf.GradientTape()) as tape:
        logits = model(x_batch, training=True)
        loss_value = loss_fn(y_batch, logits)

    grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    train_loss.update_state(loss_value)
    train_acc_metric.update_state(y_batch,logits)
    return loss_value
    
# evaluation function
@tf.function
def test_step(model, x_batch, y_batch, loss_fn, val_loss, val_acc_metric):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    logits = model(x_batch, training=False)
    t_loss = loss_fn(y_batch, logits)
    val_loss.update_state(t_loss)
    val_acc_metric.update_state(y_batch,logits)
    return t_loss
    
## Save the model as a checkpoint
def save_checkpoint(checkpoint):
    logger.debug('Saving model checkpoint...')
    checkpoint.save(os.path.join(args.checkpoint_local_dir, 'tf2-checkpoint'))
    logger.info(f'Checkpoint counter = {checkpoint.save_counter.numpy()}')
    logger.debug('Completed saving model checkpoint.')    
    
## Evaluate the model
def evaluate_model(model, ds):
    logger.info('Evaluating the model...')
    hook.set_mode(smd.modes.EVAL)
    test_loss, test_accuracy = model.evaluate(ds,verbose=EVALUATE_VERBOSE_LEVEL)
    logger.info(f'Test loss = {test_loss}')
    logger.info(f'Test accuracy = {test_accuracy}')
    logger.info('Completed evaluating the model.')
    return test_loss, test_accuracy

## Save the model
def save_model(model, model_dir):
    logger.info('Saving the model...')
    tf.saved_model.save(model, model_dir)
    logger.info('Completed saving the model.')

                            
## Train the model
def train_model(hook, model, model_dir, train_ds, valid_ds, batch_size, epochs, learning_rate):

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    val_loss_results = []
    val_accuracy_results = []
    
    # Compile the model
    optimizer, loss_fn, train_acc_metric, val_acc_metric, train_loss, val_loss = compile_model(hook, model, learning_rate)
      
    # Create the checkpoint object
    if args.checkpoint_enabled.lower() == 'true':
        checkpoint = Checkpoint(model)

    # SMDebugger: Save basic details
    hook.save_scalar('batch_size', batch_size, sm_metric=True)
    hook.save_scalar('number_of_epochs', epochs, sm_metric=True)
    hook.save_scalar('train_steps_per_epoch', len(train_ds), sm_metric=True)
    
    # Perform training
    logger.info('Training the model...')
    hook.set_mode(smd.modes.TRAIN)
    training_start_time = time.time()
    logger.debug('Iterating over epochs...')
    
    # iterate over epochs
    for epoch in range(epochs):
        train_loss.reset_states()
        train_acc_metric.reset_states()
        val_loss.reset_states()
        val_acc_metric.reset_states()
        
        logger.info(f'Starting epoch {int(epoch) + 1}/{epochs}')
        # SMDebugger: Save the epoch number
        hook.save_scalar('epoch_number', int(epoch) + 1, sm_metric=True)
        epoch_start_time = time.time()

        # iterate over training batches
        for batch, (x_batch, y_batch) in enumerate(train_ds):
            
            logger.debug(f'Running training step {int(batch) + 1}')
            # SMDebugger: Save the step number
            hook.save_scalar('step_number', int(batch) + 1, sm_metric=True)
            loss_value = training_step(hook, model, x_batch, y_batch, optimizer, loss_fn, train_loss, train_acc_metric)
            logger.debug(f'Training loss in step = {loss_value}')
            logger.debug(f'Completed running training step {int(batch) + 1}') 

        # iterate over val batches
        for x_batch, y_batch in valid_ds:
            test_step(model, x_batch, y_batch, loss_fn, val_loss, val_acc_metric)

        # end of epoch
        train_loss_results.append(train_loss.result())
        train_accuracy_results.append(train_acc_metric.result())
        val_loss_results.append(val_loss.result())
        val_accuracy_results.append(val_acc_metric.result())
        
        # Save the model as a checkpoint
        if args.checkpoint_enabled.lower() == 'true':
            save_checkpoint(checkpoint)
            
        epoch_end_time = time.time()
        logger.info(f"Epoch duration = {(epoch_end_time - epoch_start_time):.2f} seconds")
        logger.info(f'Completed epoch {int(epoch) + 1}')
        

        # print(f"Validation acc: {val_acc:.4f}")
        print(f"Train Loss: {train_loss.result():.6f}")
        print(f"Train Acc: {train_acc_metric.result():.6f}")
        print(f"Val Loss: {val_loss.result():.6f}")
        print(f"Val Acc: {val_acc_metric.result():.6f}")
        print(f"Time taken: {(epoch_end_time - epoch_start_time):.2f} seconds")
    
    logger.info('Completed iterating over epochs.')
    training_end_time = time.time()
    logger.info(f'Training duration = {(training_end_time - training_start_time):.2f} seconds')
    logger.info('Completed training the model.')   
        
if __name__ == '__main__':
    
    logger.info('Executing the main() function...')
    logger.info(f'TensorFlow version : {tf.__version__}')
    logger.info(f'SMDebug version : {smdebug.__version__}')
    
    # parse command line arguments
    args, _ = parse_args()
    
    # set seeds
    set_seeds(args.seed)
    
    # initialize the SMDebugger
    hook = init_smd()
    
    # load data
    ds = load_data(args.val_dir)
    
    # Create an instance of the model
    model = MyModel()
    
    if args.checkpoint_load_previous.lower() == 'true':
        load_weights_from_latest_checkpoint(model)
        #checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
      
    # train model            
    train_model(hook, model, args.model_dir, ds, ds, args.batch_size, args.epochs, args.lr)  
        
    # evaluate the model
    #evaluate_model(model, ds)
    
    # Save the generated model
    model.save(os.path.join(args.model_dir, "000000001"))
    # save model to an S3 directory with version number '00000001' in Tensorflow SavedModel Format. To export the model as h5 format use model.save('my_model.h5')
    
    
    # Close the SMDebugger hook
    hook.close()
    logger.info('Completed executing the main() function')