# Import modules
import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import gc
import json
import random
import efficientnet.tfkeras as efn
import numpy as np
import pandas as pd
import segmentation_models as sm
import tensorflow as tf
from datetime import datetime
#from kaggle_datasets import KaggleDatasets
from sklearn.model_selection import StratifiedKFold

# Import Custom Modules
from utils import *

# Set Strategy. Assume TPU...if not set default for GPU/CPU
tpu = None
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy()

# Set Constants
EPOCHS = 40
BACKBONE = 'efficientnetb0' # B4 
SEGMENTATION_FRAMEWORK = 'unet' # Choice: 'fpn' OR 'linknet' OR 'unet'
FOLDS = 5
SIZE = 1024
RESIZE = 256 
LR = 0.0001 
FOLD_EARLY_STOP = 5
Q = 500 # Number of Quantiles for Histogram Bins based stratification
SEED = 5363
METRIC_KEYS = ['loss', 'dice', 'accuracy']

#### Constants that can be used on Kaggle Notebooks TPU v3 or Google Colab Pro TPU v2 ##################
#BACKBONE = 'efficientnetb4' 
#SIZE = 1024
#RESIZE = 512 
#LR = 0.0002 
#FOLD_EARLY_STOP = 2 # Because of time/session limits for the 'free' versions of the TPU's
############################################################################################

# Set Batch Size
BASE_BATCH_SIZE = 24        # Max batch size for an 8GB Nvidia GPU card with RESIZE 256 and Backbone EffNet B0
if tpu is not None:         
    BASE_BATCH_SIZE = 8     # TPU v2 or up...
BATCH_SIZE = BASE_BATCH_SIZE * strategy.num_replicas_in_sync
print(f'Replica Count: {strategy.num_replicas_in_sync}')
print(f'Batch Size: {BATCH_SIZE}')

# Seeds
set_seeds(SEED)

# Get Regular Images and Path
#DATA_PATH = KaggleDatasets().get_gcs_path(f'khmtrainphase1')
DATA_PATH = 'C:/KaggleHuBMAP/tfrecords'
MODEL_PATH = '/kaggle/working/'
MODEL_NAME = f'{SEGMENTATION_FRAMEWORK}_{BACKBONE}_{SIZE}_{RESIZE}'

# Get all Kaggle official training files
ALL_TRAINING_FILENAMES = tf.io.gfile.glob(DATA_PATH + '/train/*/*.tfrec')
print(ALL_TRAINING_FILENAMES[:5])
print(len(ALL_TRAINING_FILENAMES))

# Get external training files
ALL_TRAINING_FILENAMES_EXT1 = tf.io.gfile.glob(DATA_PATH + '/ext1/*.tfrec')
print(ALL_TRAINING_FILENAMES_EXT1[:5])
print(len(ALL_TRAINING_FILENAMES_EXT1))

# Read CSV files for mask density info
df = pd.concat([pd.read_csv(f) for f in tf.io.gfile.glob(DATA_PATH + '/train/*.csv')], ignore_index = True)
df['file_name'] = DATA_PATH + '/train/' + df['relative_path']
df_ext = pd.concat([pd.read_csv(f) for f in tf.io.gfile.glob(DATA_PATH + '/ext1/*.csv')], ignore_index = True)
df_ext['file_name'] = DATA_PATH + '/ext1/' + df_ext['relative_path']
print(f'Training files CSV shape: {df.shape}')
print(f'External files CSV shape: {df_ext.shape}')

# Sets for Random Sampling
ALL_TRAINING_FILENAMES_EXT1_MASK = df_ext[df_ext['mask_density'] > 0].file_name.to_list()
ALL_TRAINING_FILENAMES_EXT1_NOMASK = df_ext[df_ext['mask_density'] == 0].file_name.to_list()
print(f'Extra2 with Mask: {len(ALL_TRAINING_FILENAMES_EXT1_MASK)}')
print(f'Extra2 with No Mask: {len(ALL_TRAINING_FILENAMES_EXT1_NOMASK)}')

# Assign Labels based on Mask or No Mask
df['mask_density_label'] = 0
def assign_label(i):
    if i == 0:
        return 0
    if i > 0:
        return 1
df['mask_density_label'] = df['mask_density'].apply(assign_label)
df.mask_density_label.value_counts(sort=True)

# Create Histograms bins based on quantiles...use for stratification
df['bin_label'] = pd.qcut(df[df['mask_density'] > 0]['mask_density'], q = Q, labels = False)
df = df.fillna(-1.0)
df['bin_label'] = df['bin_label'] + 1.0
print(df['bin_label'].value_counts(dropna = False))

# Initialize Metrics Store
M = {}
for metric in METRIC_KEYS: M[f'val_{metric}'] = []

# Create Folds
fold = StratifiedKFold(n_splits = FOLDS, shuffle = True, random_state = SEED)

# Loop Folds
for fold, (tr_idx, val_idx) in enumerate(fold.split(df, df.bin_label.values)):
    # Fold Early Stopping...To limit time required for training
    if fold > FOLD_EARLY_STOP:
        break
    
    # START        
    print(f'############ FOLD {fold + 1} #############')
    
    # CREATE TRAIN AND VALIDATION SUBSETS
    df_train = df.loc[tr_idx]
    VALIDATION_FILENAMES = df.file_name[val_idx]

    # Subsets for Random Sampling
    ALL_TRAINING_FILENAMES_MASK = df_train[df_train['mask_density'] > 0].file_name.to_list()
    ALL_TRAINING_FILENAMES_NOMASK = df_train[df_train['mask_density'] == 0].file_name.to_list()

    # Finalize Validation
    VALIDATION_STEPS_PER_EPOCH = len(VALIDATION_FILENAMES) // BATCH_SIZE
    print(f'Valid Filenames: {VALIDATION_FILENAMES[:5]}')
    print(f'Valid Steps: {VALIDATION_STEPS_PER_EPOCH}')
    validation_dataset = get_validation_dataset(VALIDATION_FILENAMES, BATCH_SIZE, SIZE, RESIZE, SEED, ordered = True, augment = False)
    
    # Cleanup
    tf.keras.backend.clear_session()    
    if tpu is not None:
        tf.tpu.experimental.initialize_tpu_system(tpu)
    gc.collect()
    
    # Create Model
    with strategy.scope():   
        model = create_model(BACKBONE, SEGMENTATION_FRAMEWORK)

    ########### CUSTOM LOOP WITH SAMPLING FOR EACH EPOCH #####################################################################
    fold_history = initialize_fold_history(METRIC_KEYS)
    val_dice = 0.0

    for epoch in range(EPOCHS): 
        print(f'\n======= Training Model Fold {fold} - Epoch: {epoch}')  
         
        # Get Ramdom Samples
        EXTMask = random_sampler(ALL_TRAINING_FILENAMES_EXT1_MASK, 0.60)
        EXTNoMask = random_sampler(ALL_TRAINING_FILENAMES_EXT1_NOMASK, 0.25)
        TrainMask = random_sampler(ALL_TRAINING_FILENAMES_MASK, 0.60)
        TrainNoMask = random_sampler(ALL_TRAINING_FILENAMES_NOMASK, 0.25)

        # Add External Data to Training Filenames....Leave out of Validation Part...Only official training set images are used for that.
        TOTAL_TRAINING_FILENAMES = TrainMask + TrainNoMask + EXTMask + EXTNoMask 
        random.shuffle(TOTAL_TRAINING_FILENAMES)
        print(TOTAL_TRAINING_FILENAMES[:10])

        # Set Steps
        STEPS_PER_EPOCH = len(TOTAL_TRAINING_FILENAMES) // BATCH_SIZE
        
        # Create Datasets
        training_dataset = get_training_dataset(TOTAL_TRAINING_FILENAMES, BATCH_SIZE, SIZE, RESIZE, SEED, ordered = False, augment = True)

        # Set Learning Rate
        tf.keras.backend.set_value(model.optimizer.learning_rate, lr_scheduler(epoch, LR))
        
        # Fit Model
        print(f'Train Steps: {STEPS_PER_EPOCH}')
        print(f'Validation Steps: {VALIDATION_STEPS_PER_EPOCH}')
        print(f'Setting Learning rate: {model.optimizer.learning_rate.numpy():.6f}')
        history = model.fit(training_dataset,
                            epochs = 1,
                            steps_per_epoch = STEPS_PER_EPOCH,
                            validation_steps = VALIDATION_STEPS_PER_EPOCH,
                            validation_data = validation_dataset,
                            verbose = 1)

        # Add Per Epoch History to the full fold history
        for key in fold_history.keys():
            fold_history[key].append(history.history[key][0])

        # Custom Model Checkpointing
        current_val_dice = history.history['val_dice'][0]
        if current_val_dice > val_dice:
            print(f'Val Dice improved from {val_dice} to {current_val_dice} ... Saving Model to: {MODEL_PATH}{MODEL_NAME}_{fold}.h5')
            model.save(f'{MODEL_PATH}{MODEL_NAME}_{fold}.h5')
            # Update Val Dice
            val_dice = current_val_dice
        else:
            print(f'Val Dice did not improve from {val_dice} to {current_val_dice} ... ')
    
    # Evaluate Model
    print(f'\nEvaluate Model Fold {fold}...')
    model.load_weights(f'{MODEL_PATH}{MODEL_NAME}_{fold}.h5')
    eval = model.evaluate(validation_dataset, steps = VALIDATION_STEPS_PER_EPOCH, return_dict = True)
    for metric in METRIC_KEYS: M[f'val_{metric}'].append(eval[metric])
    
    # Cleanup
    del model, training_dataset, validation_dataset
    gc.collect()
    
    # Plot Training and Validation Results for Fold
    plot_training(fold_history, f'{MODEL_PATH}plot_{MODEL_NAME}_{fold}.png')

# Write Final Metrics to file
M['datetime'] = f'{datetime.now()}'
for metric in METRIC_KEYS:
    M['oof_' + metric] = np.mean(M[f'val_{metric}'])
    print(f'OOF {metric}: {M[f"oof_{metric}"]}')

    with open(f'{MODEL_PATH}metrics.json', 'w') as metric_file:
        json.dump(M, metric_file)