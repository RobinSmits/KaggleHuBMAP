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
generate_stage2 = False                         # Generate Stage 1 ... OR ... Generate Stage 2 based on Pseudo Labelling

# Required Folders - Modify these if required to fit your local folder structure
root_dir = 'C:/KaggleHuBMAP/'
train_data_dir = f'{root_dir}train/'            # Folder for official Kaggle HuBMAP Train Data
test_data_dir = f'{root_dir}test/'              # Folder for official Kaggle HuBMAP Test Data. Used for Pseudo Labelling Stage2
ext1_data_dir = f'{root_dir}ext1/'              # Folder with External data from: https://www.kaggle.com/baesiann/glomeruli-hubmap-external-1024x1024
ext2_data_dir = f'{root_dir}ext2/'              # Folder with External data from: https://portal.hubmapconsortium.org/search?entity_type%5B0%5D=Dataset
tfrecords_dir = f'{root_dir}tfrecords/'         # Output directory for created TFRecords
tfrecords_train_dir = f'{tfrecords_dir}train/'
tfrecords_test_dir = f'{tfrecords_dir}test/'
tfrecords_ext1_dir = f'{tfrecords_dir}ext1/'
tfrecords_ext2_dir = f'{tfrecords_dir}ext2/'

# Prepare TFRecords and Dataset Output Dir
clean_and_prepare_dir(tfrecords_dir)
# Only generate stage 1 train files
if not generate_stage2:  
    clean_and_prepare_dir(tfrecords_train_dir)
    clean_and_prepare_dir(tfrecords_ext1_dir)
# Only generate stage 2 train files
else:  
    clean_and_prepare_dir(tfrecords_test_dir)
    clean_and_prepare_dir(tfrecords_ext2_dir)

#### STAGE 1 ###########################################################################################################################

# Only generate stage 1 train files
if not generate_stage2:
    
    #### Prepare and Process official Kaggle Train Images ##############################################################################
    
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

    #### Prepare and Process First External Dataset Images #############################################################################
    
    # List Images
    ext1_images = os.listdir(f'{ext1_data_dir}images_1024/')

    # Create
    patch_df = write_tfrecord_tiles_v2(ext1_images, ext1_data_dir, tfrecords_ext1_dir)

    # Create Dataframe
    patch_df.to_csv(f'{tfrecords_ext1_dir}ext1_patches.csv', index = False)

    # Clean Memory
    gc.collect()


    #### Create Final Zip File for upload to Kaggle Datasets ###############################################################################
    shutil.make_archive(f'{root_dir}train_files_stage1', 'zip', root_dir = tfrecords_dir, base_dir = './')

#### STAGE 2 ###########################################################################################################################

# Only generate stage 2 train files
if generate_stage2:
    
    #### Prepare and Process official Kaggle Test Images ##############################################################################
    
    # Read Pseudo Label Info for Public Test. 
    # Pseudo Label public test data by using a selected ensemble of models and perform inference based on the public test data
    # The generated predictions .csv file contains the masks for public test data. We can re-use these as additional training data
    test_df = pd.read_csv(f'{root_dir}pseudolabel_test.csv')
    print(test_df.shape)

    # Loop through all public Test Images
    test_image_count = test_df.shape[0]
    for image_index in tqdm(range(test_image_count), total = test_image_count):
        # Get Image ID
        image_id = test_df['id'][image_index]    

        # Get TIFF Image
        image = get_tiff_image(f'{test_data_dir}{image_id}.tiff')
        
        # Get Mask
        mask = rle2mask(test_df['predicted'][image_index], (image.shape[1], image.shape[0]))
        
        # Create Patches and TFRecords for TIFF Image
        patch_df = write_tfrecord_tiles_v1(image_id, image, mask, patch_size, tfrecords_test_dir)
        
        # Create Dataframe
        patch_df.to_csv(f'{tfrecords_test_dir}{image_id}_patches.csv', index = False)

        # Clean Memory
        gc.collect()

    #### Prepare and Process Second External Dataset Images #############################################################################
    
    # Pseudo Label second external dataset by using a selected ensemble of models and perform inference on the data
    # The generated predictions .csv file contains the predicted masks for the second external data set. We can re-use these as additional training data
    ext2_df = pd.read_csv(f'{root_dir}pseudolabel_ext2.csv')
    print(ext2_df.shape)

    # Loop through all second external data set images
    ext2_image_count = ext2_df.shape[0]
    for image_index in tqdm(range(ext2_image_count), total = ext2_image_count):
        # Get Image ID
        image_id = ext2_df['id'][image_index]    

        # Get TIFF Image
        image = get_tiff_image(f'{ext2_data_dir}{image_id}.tiff')
        
        # Get Mask
        mask = rle2mask(ext2_df['predicted'][image_index], (image.shape[1], image.shape[0]))
        
        # Create Patches and TFRecords for TIFF Image
        patch_df = write_tfrecord_tiles_v1(image_id, image, mask, patch_size, tfrecords_ext2_dir)
        
        # Create Dataframe
        patch_df.to_csv(f'{tfrecords_ext2_dir}{image_id}_patches.csv', index = False)

        # Clean Memory
        gc.collect()

    #### Create Final Zip File for upload to Kaggle Datasets ###############################################################################
    shutil.make_archive(f'{root_dir}train_files_stage2', 'zip', root_dir = tfrecords_dir, base_dir = './')

# Final
print('=== Finished Training Files Processing')