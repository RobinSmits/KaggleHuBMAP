# Import Modules
import cv2
import os
import shutil
import tifffile
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple
from tqdm import tqdm

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
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
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