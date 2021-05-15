# Import Modules
import gc
import numpy as np
import pandas as pd
import shutil
import tensorflow as tf
from utils import * 
from tqdm import tqdm

# Constants
patch_size = 1024

# Required Folders - Modify these if required to fit your local folder structure
root_dir = 'C:/KaggleHuBMAP/'
train_data_dir = f'{root_dir}train/'        # Folder for official Kaggle HuBMAP Train Data
ext1_data_dir = f'{root_dir}ext1/'          # Folder with External data from: https://www.kaggle.com/baesiann/glomeruli-hubmap-external-1024x1024
tfrecords_dir = f'{root_dir}tfrecords/'     # Output directory for created TFRecords
tfrecords_train_dir = f'{tfrecords_dir}train/'
tfrecords_ext1_dir = f'{tfrecords_dir}ext1/'

# Prepare TFRecords and Dataset Output Dir
clean_and_prepare_dir(tfrecords_dir)
clean_and_prepare_dir(tfrecords_train_dir)
clean_and_prepare_dir(tfrecords_ext1_dir)


########################################################################################################################################
#### Prepare and Process official Kaggle Train Images ##################################################################################

# Read Train Info
train_df = pd.read_csv(f'{root_dir}train.csv')
print(train_df.shape)

# Loop through all Train Images
train_image_count = train_df.shape[0]
for image_index in tqdm(range(train_image_count), total = train_image_count):
    # Get Image ID
    image_id = train_df['id'][image_index]    

    # Get TIFF Image
    image = get_tiff_image(f'{train_data_dir}{image_id}.tiff')
    
    # Get Mask
    mask = rle2mask(train_df['encoding'][image_index], (image.shape[1], image.shape[0]))
    
    # Create Patches and TFRecords for TIFF Image
    patch_df = write_tfrecord_tiles_v1(image_id, image, mask, patch_size, tfrecords_train_dir)
    
    # Create Dataframe
    patch_df.to_csv(f'{tfrecords_train_dir}{image_id}_patches.csv', index = False)

    # Clean Memory
    gc.collect()


########################################################################################################################################
#### Prepare and Process First External Dataset Images #################################################################################

# List Images
ext1_images = os.listdir(f'{ext1_data_dir}images_1024/')

# Create
patch_df = write_tfrecord_tiles_v2(ext1_images, ext1_data_dir, tfrecords_ext1_dir)

# Create Dataframe
patch_df.to_csv(f'{tfrecords_ext1_dir}ext1_patches.csv', index = False)

# Clean Memory
gc.collect()


########################################################################################################################################
#### Create Final Zip File for upload to Kaggle Datasets ###############################################################################
shutil.make_archive(f'{root_dir}train_files_fase1', 'zip', root_dir = tfrecords_dir, base_dir = './')

# Final
print('=== Finished')