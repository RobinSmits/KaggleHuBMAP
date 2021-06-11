# Import Modules
import cv2
import gc
import glob
import os
import pathlib
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import efficientnet.tfkeras
import numpy as np
import pandas as pd
import tensorflow as tf

# Import Custom Modules
from utils import *

# Constants and Settings
WINDOW = 1024
MIN_OVERLAP = 256
RESIZE = 256
THRESHOLD = 0.40
submission = {} # Placeholder Dictionary

# Inference Settings
USE_BATCH = True        # When USE_TTA = True ... AND ... large Backbone B4 or higher and WINDOW > 2048 ... set this to False to prevent memory issues.
USE_TTA = True          # Apply Test Time Augmentation ... OR ... only use plain input patch.
USE_THRESHOLD = True    # Use a fixed THRESHOLD ... OR process probabilities using DenseCRF

# Get Test Files Path
p = pathlib.Path('C:/KaggleHuBMAP/test/')

# Specify your model(s) files to load
model_weights_list = ['C:/KaggleHuBMAP/models/model.h5']

# Get Models
fold_models = get_models(model_weights_list)

# Perform Inference
for id, filename in tqdm(enumerate(p.glob('*.tiff')), total = len(list(p.glob('*.tiff')))):

    print(f'\n=================== {id+1} Predicting {filename.stem}')    
    
    with rasterio.open(filename.as_posix()) as dataset:
        if dataset.count == 3:
            slices = make_grid(dataset.shape, window = WINDOW, min_overlap = MIN_OVERLAP)
            preds = np.zeros(dataset.shape, dtype = np.uint8)
            
            for (x1, x2, y1, y2) in tqdm(slices, total = len(slices)):
                image = dataset.read([1,2,3], window = Window.from_slices((x1, x2), (y1, y2))).transpose(1,2,0).copy()
                
                # Get Prediction for image
                probs = get_prediction(image, RESIZE, fold_models, USE_BATCH, USE_TTA)

                # Use Threshold or DenseCRF for Final Patch Mask
                if USE_THRESHOLD:
                    probs = cv2.resize(probs, (WINDOW, WINDOW), interpolation = cv2.INTER_AREA)
                    pred = (probs > THRESHOLD).astype(np.uint8)
                else: # USE DenseCRF
                    pred = crf_softmax(probs, RESIZE, WINDOW)                    
                
                # Add To Final
                preds[x1:x2, y1:y2] += pred
                
            # Finalize Full Mask
            preds = (preds > 0.5).astype(np.uint8)
            
            # Encode Final RLE for submission
            submission[id] = {'id': filename.stem, 'predicted': rle_encode_less_memory(preds)}
            
            # Cleanup
            del slices, preds
            gc.collect()

        else:
            h, w = (dataset.height, dataset.width)

            slices = make_grid(dataset.shape, window = WINDOW, min_overlap = MIN_OVERLAP)
            preds = np.zeros(dataset.shape, dtype = np.uint8)

            for (x1, x2, y1, y2) in tqdm(slices, total = len(slices)):
                subdatasets = dataset.subdatasets
                if len(subdatasets) > 0:
                    image = np.zeros((WINDOW, WINDOW, len(subdatasets)), dtype=np.uint8)
                    for i, subdataset in enumerate(subdatasets, 0):
                        with rasterio.open(subdataset) as layer:
                            image[:,:,i] = layer.read(1, window = Window.from_slices((x1, x2), (y1, y2)))

                # Get Prediction for image
                probs = get_prediction(image, RESIZE, fold_models, USE_BATCH, USE_TTA)

                # Use Threshold or DenseCRF for Final Patch Mask
                if USE_THRESHOLD:
                    probs = cv2.resize(probs, (WINDOW, WINDOW), interpolation = cv2.INTER_AREA)
                    pred = (probs > THRESHOLD).astype(np.uint8)
                else:
                    pred = crf_softmax(probs, RESIZE, WINDOW)
                
                # Add To Final
                preds[x1:x2, y1:y2] += pred
                
            # Finalize Mask
            preds = (preds > 0.5).astype(np.uint8)
            
            # Encode Final RLE for submission
            submission[id] = {'id': filename.stem, 'predicted': rle_encode_less_memory(preds)}
            
            # Cleanup
            del slices, preds
            gc.collect()
            
    # Cleanup
    gc.collect()

# Create Submission
submission = pd.DataFrame.from_dict(submission, orient = 'index')
submission.to_csv('submission.csv', index = False)
print(submission.head())