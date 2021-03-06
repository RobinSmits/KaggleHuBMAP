### Note. Certain functions were re-used / modified from Public Kaggle Kernels ###

# Import Modules
import cv2
import gc
import os
import random
import shutil
import tifffile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models as sm
import tensorflow as tf

from typing import Tuple
from tqdm import tqdm

# DenseCRF
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_softmax

def clean_and_prepare_dir(dir: str):
    # Remove existing stuff..
    if os.path.exists(dir):
        shutil.rmtree(dir)
    # Create
    os.mkdir(dir)

def rle2mask(mask_rle: str, shape: Tuple)->np.array:
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    mask = img.reshape(shape).T   
    return mask

def get_cols_rows(img_cols: int, img_rows: int, patch_size: int)->Tuple[int, int]:  
    cols = img_cols // patch_size
    rows = img_rows // patch_size    

    return cols, rows

def get_image_mask_patches(image: np.array, mask: np.array, col_number: int, row_number: int, patch_size: int)->Tuple[np.array, np.array]:
    lower_col_range = (col_number * patch_size)
    higher_col_range = lower_col_range + patch_size
    lower_row_range = (row_number * patch_size)
    higher_row_range = lower_row_range + patch_size

    image_patch = image[lower_col_range:higher_col_range, lower_row_range:higher_row_range, :]
    mask_patch = mask[lower_col_range:higher_col_range, lower_row_range:higher_row_range]

    return image_patch, mask_patch

def get_tiff_image(tiff_file_name: str)->np.array:
    image = tifffile.imread(tiff_file_name)
    image = np.squeeze(image)
    
    # Correct Shape if necessary
    if(image.shape[0] == 3):
        image = image.swapaxes(0,1)
        image = image.swapaxes(1,2)
    
    return image

def get_saturation_mean(image_patch: np.array)->float:
    im_hsv = cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR)
    im_hsv = cv2.cvtColor(im_hsv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(im_hsv)
    
    return s.mean()
    
def bytes_feature(value: bytes)->tf.train.Feature:
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def int64_feature(value: int)->tf.train.Feature:
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def image_example(image: np.array, mask: np.array)->tf.train.Example:
    features = {
        'image_bytes': bytes_feature(image.tostring()),
        'mask_bytes' : bytes_feature(mask.tostring()),
        'height': int64_feature(image.shape[0]),
        'width': int64_feature(image.shape[1]),
        'channels': int64_feature(image.shape[2])                
    }
    
    return tf.train.Example(features = tf.train.Features(feature = features))

def create_tfrecord(image: np.array, mask: np.array, filename: str):
    with tf.io.TFRecordWriter(filename, tf.io.TFRecordOptions(compression_type = 'GZIP')) as tfwriter:
        tf_example = image_example(image, mask)
        tfwriter.write(tf_example.SerializeToString())
    
    tfwriter.close()

def write_tfrecord_tiles_v1(image_id: str, image: np.array, mask: np.array, patch_size: int, tfrecords_dataset_dir: str)->pd.DataFrame:
    # Set Output Dir for Image TF Records
    image_tfrecords_dir = f'{tfrecords_dataset_dir}{image_id}/'
    
    # Prepare Image TFRecords Output Dir
    clean_and_prepare_dir(image_tfrecords_dir)

    # Get Cols and Rows based on Image Shape and Patch Size
    cols, rows = get_cols_rows(image.shape[0], image.shape[1], patch_size)   
    
    # Create Pandas Dataframe
    patches_df = pd.DataFrame(columns = ['img_id', 'relative_path', 'saturation_mean', 'mask_density'])
    
    # Loop through cols and rows
    for col_number in range(cols):
        for row_number in range(rows):
            dataset_file_path = f'{image_tfrecords_dir}{image_id}_{col_number}_{row_number}.tfrec'
            relative_path = f'{image_id}/{image_id}_{col_number}_{row_number}.tfrec'

            # Get Image/Mask Patches
            image_patch, mask_patch = get_image_mask_patches(image, mask, col_number, row_number, patch_size)
            
            # Get Saturation Mean
            saturation_mean = get_saturation_mean(image_patch)
                        
            # Create TFRecord
            create_tfrecord(image_patch, mask_patch, dataset_file_path)

            # Add Dataframe Row
            mask_density = np.count_nonzero(mask_patch)
            patches_df = patches_df.append({'img_id': image_id, 
                                            'relative_path': relative_path,
                                            'saturation_mean': saturation_mean,
                                            'mask_density': mask_density}, ignore_index = True)

    return patches_df

def write_tfrecord_tiles_v2(external_images: list, data_dir: str, tfrecords_dataset_dir: str)->pd.DataFrame:
    # Create Pandas Dataframe
    patches_df = pd.DataFrame(columns = ['img_id', 'relative_path', 'saturation_mean', 'mask_density'])
    
    # Loop through images
    for ext_image in tqdm(external_images, total = len(external_images)):
        # Get Image Patch
        image_patch = tf.io.decode_png(tf.io.read_file(f'{data_dir}images_1024/{ext_image}'))
        image_patch = image_patch.numpy()

        # Get Mask Patch
        mask_patch = tf.io.decode_png(tf.io.read_file(f'{data_dir}masks_1024/{ext_image}'))
        mask_patch = (mask_patch.numpy() > 0).astype(np.uint8)
        mask_patch = np.squeeze(mask_patch)
        
        # Get Image Id and filenames
        image_id = ext_image.split('.')[0]
        file_name = f'{image_id}.tfrec'
        dataset_file_path = f'{tfrecords_dataset_dir}{file_name}'

        # Get Saturation Mean
        saturation_mean = get_saturation_mean(image_patch)
        
        # Create TFRecord
        create_tfrecord(image_patch, mask_patch, dataset_file_path)

        # Add Dataframe Row
        mask_density = np.count_nonzero(mask_patch)
        patches_df = patches_df.append({'img_id': image_id, 
                                        'relative_path': file_name,
                                        'saturation_mean': saturation_mean,
                                        'mask_density': mask_density}, ignore_index = True)

    return patches_df

################# TRAINING UTILS #################################################################################################

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def initialize_fold_history(metric_keys: list)->dict:
    history = {}
    for metric in metric_keys: history[metric] = []
    for metric in metric_keys: history[f'val_{metric}'] = []

    return history

def parse_image(tf_example, size, resize, seed, augment = True):
    feature_description = {'height': tf.io.FixedLenFeature([], tf.int64),
                            'width': tf.io.FixedLenFeature([], tf.int64),
                            'channels': tf.io.FixedLenFeature([], tf.int64),
                            'image_bytes': tf.io.FixedLenFeature([], tf.string),
                            'mask_bytes': tf.io.FixedLenFeature([], tf.string)}
    
    single_example = tf.io.parse_single_example(tf_example, feature_description)
    
    image_bytes =  tf.io.decode_raw(single_example['image_bytes'], out_type = 'uint8')   
    image = tf.reshape(image_bytes, (size, size, 3))   
    mask_bytes =  tf.io.decode_raw(single_example['mask_bytes'], out_type = 'bool')    
    mask = tf.reshape(mask_bytes, (size, size, 1))
    
    # Normalize
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    
    # Resize
    image = tf.image.resize(image, [resize, resize])
    mask = tf.cast(mask, tf.float32)
    mask = tf.image.resize(mask, [resize, resize])
    
    if augment:            
        # Horizontal Flip
        flip_lr_number = random.uniform(0, 1)
        if flip_lr_number < 0.50:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        # Vertical Flip
        flip_ud_number = random.uniform(0, 1)
        if flip_ud_number < 0.50:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        
        # Get Random Number
        number = random.uniform(0, 1) 
        
        # Rotate 90 degrees
        if 0.00 <= number < 0.17:
            image = tf.image.rot90(image, k = 1)
            mask = tf.image.rot90(mask, k = 1)

        # Rotate 180 degrees
        if 0.17 <= number < 0.34:
            image = tf.image.rot90(image, k = 2)
            mask = tf.image.rot90(mask, k = 2)

        # Rotate 270 degrees
        if 0.34 <= number < 0.50:
            image = tf.image.rot90(image, k = 3)
            mask = tf.image.rot90(mask, k = 3)

        # OneOf
        one_of_number = random.uniform(0, 1)
        if 0.0 <= one_of_number < 0.50:    
            # Other Augmentations...perform always but with small intervals
            image = tf.image.random_saturation(image, 0.94, 1.06, seed = seed)
            #image = tf.image.random_brightness(image, 0.05, seed = seed)
            image = tf.image.random_contrast(image, 0.94, 1.06, seed = seed)
            image = tf.image.random_hue(image, 0.075, seed = seed)
        else:
            one_of_one_number = random.uniform(0, 1)
            if 0.0 <= one_of_one_number < 1/4.:
                image = tf.image.random_saturation(image, 0.94, 1.06, seed = seed)
            if 1/4. <= one_of_one_number < 2/4.:
                image = tf.image.random_brightness(image, 0.05, seed = seed)
            if 2/4. <= one_of_one_number < 3/4.:
                image = tf.image.random_contrast(image, 0.94, 1.06, seed = seed)
            if 3/4. <= one_of_one_number <= 4/4.:
                image = tf.image.random_hue(image, 0.075, seed = seed)
                
    return tf.cast(image, tf.float32), tf.cast(mask, tf.float32)

def load_dataset(filenames, size, resize, seed, ordered = False, augment = False):
    ignore_order = tf.data.Options() 
    ignore_order.experimental_deterministic = ordered
    
    dataset = tf.data.TFRecordDataset(filenames, 
                                      num_parallel_reads = tf.data.experimental.AUTOTUNE, 
                                      compression_type = "GZIP")
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(lambda tf_example: parse_image(tf_example, size, resize, seed, augment = augment), num_parallel_calls = tf.data.experimental.AUTOTUNE)
    
    return dataset

def get_training_dataset(training_filenames: list, batch_size: int, size: int, resize: int, seed: int, ordered = False, augment = False)->tf.data.Dataset:
    dataset = (load_dataset(training_filenames, size, resize, seed, ordered = ordered, augment = augment)
                            .repeat()
                            .shuffle(1024, seed = seed, reshuffle_each_iteration = True)
                            .batch(batch_size, drop_remainder = True)
                            .prefetch(tf.data.experimental.AUTOTUNE))
    
    return dataset

def get_validation_dataset(validation_filenames: list, batch_size: int, size: int, resize: int, seed: int, ordered = True, augment = False)->tf.data.Dataset:
    dataset = (load_dataset(validation_filenames, size, resize, seed, ordered = ordered, augment = augment)
                            .batch(batch_size, drop_remainder = True)
                            .prefetch(tf.data.experimental.AUTOTUNE))
    
    return dataset

def lr_scheduler(epoch: int, LR: float)->float:
    # Set Custom LR Schedule
    if epoch == 0:
        lr = 0.15 * LR
    if epoch == 1:
        lr = 0.30 * LR
    if epoch == 2:
        lr = 0.45 * LR
    if epoch == 3:
        lr = 0.60 * LR
    if epoch == 4:
        lr = 0.75 * LR
    if epoch == 5:
        lr = 0.90 * LR
    if epoch > 5:
        lr = LR   
    if epoch >= 25:
        lr = 0.8 * LR
    if epoch >= 35:
        lr = 0.60 * LR
        
    return lr

def dice(output, target, axis = None, smooth = 1e-10):
    output = tf.dtypes.cast( tf.math.greater(output, 0.5), tf.float32)
    target = tf.dtypes.cast( tf.math.greater(target, 0.5), tf.float32)
    
    inse = tf.reduce_sum(output * target, axis = axis)
    
    l = tf.reduce_sum(output, axis = axis)
    r = tf.reduce_sum(target, axis = axis)

    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice, name = 'dice')
    
    return dice

def random_sampler(item_list: list, sample_percentage: float, debug = False)->list:
    item_list_length = len(item_list)
    k_samples = int(item_list_length * sample_percentage)
    sampled_list = random.sample(item_list, k_samples)

    # Shuffle Sample Selection
    random.shuffle(sampled_list)

    # Summary
    if debug:
        print(f'Random Sampler Input List length: {item_list_length}   Percentage: {sample_percentage}   Sampled Items: {len(sampled_list)}')
    
    return sampled_list

def plot_training(history: dict, plot_file_name: str):
    plt.figure(figsize = (16, 6))
    n_e = np.arange(len(history['dice']))

    plt.plot(n_e, history['dice'], '-o', label = 'Train dice', color = '#ff7f0e')
    plt.plot(n_e, history['val_dice'], '-o', label = 'Val dice', color = '#1f77b4')
    x = np.argmax(history['val_dice']); y = np.max(history['val_dice'])
    xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color='#1f77b4'); plt.text(x-0.03*xdist, y-0.13*ydist, 'max dice\n%.4f'%y, size = 12)
    plt.ylabel('dice',size=14); plt.xlabel('Epoch', size=14)
    plt.legend(loc=2)
    plt2 = plt.gca().twinx()

    plt2.plot(n_e, history['loss'], '-o', label = 'Train Loss', color = '#2ca02c')
    plt2.plot(n_e, history['val_loss'], '-o', label = 'Val Loss', color = '#d62728')
    x = np.argmin(history['val_loss']); y = np.min(history['val_loss'])
    ydist = plt.ylim()[1] - plt.ylim()[0]

    plt.scatter(x,y,s=200,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss', size = 12)
    plt.ylabel('Loss', size = 14)
    plt.legend(loc = 3)
    plt.savefig(plot_file_name)

def create_model(backbone: str, segmentation_framework: str)->tf.keras.Model:
    if segmentation_framework == 'fpn':
        model = sm.FPN(backbone, 
                        encoder_weights = 'imagenet', 
                        activation = 'sigmoid', 
                        encoder_freeze = False, 
                        classes = 1)

    if segmentation_framework == 'linknet':
        model = sm.Linknet(backbone, 
                           encoder_weights = 'imagenet', 
                           activation = 'sigmoid', 
                           encoder_freeze = False, 
                           classes = 1)

    if segmentation_framework == 'unet':
        model = sm.Unet(backbone, 
                        encoder_weights = 'imagenet', 
                        activation = 'sigmoid', 
                        encoder_freeze = False, 
                        classes = 1)

    # Compile Model
    model.compile(optimizer = tf.keras.optimizers.Adam(),
                    loss = (0.50 * sm.losses.BinaryCELoss()) + (0.25 * sm.losses.DiceLoss()) + (0.25 * sm.losses.BinaryFocalLoss()),
                    metrics = [dice, 'accuracy'])
    
    return model

################# INFERENCE UTILS #################################################################################################

def rle_encode_less_memory(img: np.ndarray)->str:
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

def make_grid(shape: tuple, window: int, min_overlap: int)->np.ndarray:
    x, y = shape
    
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num = nx, endpoint = False, dtype = np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num = ny, endpoint = False, dtype = np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    
    slices = np.zeros((nx,ny, 4), dtype = np.int64)
    
    for i in range(nx):
        for j in range(ny):
            slices[i,j] = x1[i], x2[i], y1[j], y2[j]

    return slices.reshape(nx*ny, 4)

def image_normal(image: np.ndarray, resize: int)->np.ndarray:
    image = np.expand_dims(image, 0)
    
    # Normalize
    image = image / 255.0
    image = tf.image.resize(image, [resize, resize])
            
    return image

def image_tta(image: np.ndarray, resize: int)->tuple:
    # Normalize
    image = image / 255.0

    # Resize
    image = tf.image.resize(image, [resize, resize])

    # TTA
    image_lr = np.fliplr(image)
    image_ud = np.flipud(image)
    image_tr = np.transpose(image, (1, 0, 2))
    
    return np.expand_dims(image, 0), np.expand_dims(image_lr, 0), np.expand_dims(image_ud, 0), np.expand_dims(image_tr, 0) 

def image_tta_batch(image: np.ndarray, resize: int)->np.ndarray:
    # Normalize
    image = image / 255.0

    # Resize
    image = tf.image.resize(image, [resize, resize])

    # TTA
    image_lr = np.fliplr(image)
    image_ud = np.flipud(image)
    image_tr = np.transpose(image, (1, 0, 2))    
    image_stack = np.stack((image, image_lr, image_ud, image_tr))

    return image_stack

def get_prediction(image: np.ndarray, resize: int, fold_models: list, use_batch: bool, use_tta: bool)->np.ndarray:
    # Placeholder
    pred = None
        
    # Process Image and create Predictions .. works for single model or list of models
    if not use_tta: # Just use plain input image
        # Process Image
        image = image_normal(image, resize)
        
        # Create Predictions
        for fold_model in fold_models:
            if pred is None:
                pred = np.squeeze(fold_model.predict(image))
            else:
                pred += np.squeeze(fold_model.predict(image))
                    
    elif use_batch and use_tta:
        # Process Image
        image_batch = image_tta_batch(image, resize)

        # Create Predictions
        for fold_model in fold_models:
            if pred is None:
                pred_tta = np.squeeze(fold_model.predict(image_batch))
                pred = (pred_tta[0,] + np.fliplr(pred_tta[1,]) + np.flipud(pred_tta[2,]) + np.transpose(pred_tta[3,])) / 4 
            else:
                pred_tta = np.squeeze(fold_model.predict(image_batch))
                pred += (pred_tta[0,] + np.fliplr(pred_tta[1,]) + np.flipud(pred_tta[2,]) + np.transpose(pred_tta[3,])) / 4 
    
    elif not use_batch and use_tta:
        # Process Image
        image, image_lr, image_ud, image_tr = image_tta(image, resize)
        
        # Create Predictions ... because of larger backbone models and patch sizes 2048 * 2048 and up 
        # .. it was needed to predict on single images instead of a batch of images to resolve memory issues.  
        for fold_model in fold_models:
            if pred is None:
                pred1 = np.squeeze(fold_model.predict(image))
                pred2 = np.squeeze(fold_model.predict(image_lr))
                pred3 = np.squeeze(fold_model.predict(image_ud))
                pred4 = np.squeeze(fold_model.predict(image_tr))
                pred = (pred1 + np.fliplr(pred2) + np.flipud(pred3) + np.transpose(pred4)) / 4 
            else:
                pred1 = np.squeeze(fold_model.predict(image))
                pred2 = np.squeeze(fold_model.predict(image_lr))
                pred3 = np.squeeze(fold_model.predict(image_ud))
                pred4 = np.squeeze(fold_model.predict(image_tr))
                pred += (pred1 + np.fliplr(pred2) + np.flipud(pred3) + np.transpose(pred4)) / 4
            
    pred = pred / len(fold_models)
    
    return pred

def get_models(model_weights_list: list)->list:
    fold_models = []
    
    # Memory Cleanup
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Load Models
    for fold_model_path in model_weights_list:
        fold_models.append(tf.keras.models.load_model(fold_model_path, compile = False))

    return fold_models

def crf_softmax(probs: np.ndarray, size: int, window: int)->np.ndarray:    
    # The first dimension needs to be equal to the number of classes.
    # Let's have one "foreground" and one "background" class.
    probs = np.tile(probs[np.newaxis,:,:],(2,1,1))
    probs[1,:,:] = 1 - probs[0,:,:]
    
    # Inference 
    U = unary_from_softmax(probs)
    d = dcrf.DenseCRF2D(size, size, 2)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy = (3, 3), compat = 3, kernel = dcrf.DIAG_KERNEL, normalization = dcrf.NORMALIZE_SYMMETRIC)
    
    # Run inference for X iterations
    Q_unary = d.inference(5)

    # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
    map = np.argmin(Q_unary, axis = 0)

    # DenseCRF flattens everything ... so reshape it back
    map = map.reshape((size, size))

    # Resize back to original patch
    crf_pred = cv2.resize(map.astype(np.uint8), (window, window), interpolation = cv2.INTER_AREA)
    pred = (crf_pred > 0.5).astype(np.uint8)
    
    return pred 