# Kaggle HuBMAP - 'Hacking the Kidney'

## Introduction

Recently I participated in the Kaggle Data Science competition 'Hacking the Kidney'. It required applying semantic segmentation to predict the functional tissue units within kidney images.

With a best score of 0.9437 and a final submission score of 0.9419 for an ensemble of Unet, FPN and Linknet segmentation models. With multiple ensembles scoring within 1% from the leaderboard winning score I was really happy with the results achieved.

## Datasets

Training was performed in 2 stages. In the first stage the following data was used:
- Original Kaggle HuBMAP Training data (To obtain the data you need to have an official Kaggle account).
- External data set [https://www.kaggle.com/baesiann/glomeruli-hubmap-external-1024x1024](https://www.kaggle.com/baesiann/glomeruli-hubmap-external-1024x1024)

In the second stage pseudo-labelling was used for the following data:
- The original Kaggle Public Test data.
- 2 additional external images from [https://portal.hubmapconsortium.org/](https://portal.hubmapconsortium.org/search?entity_type[0]=Dataset)

After pseudo-labelling the second stage data it could be used as additional training-data

## Training

The training was performed in 2 stages. For each stage the following actions were performed. Note that multiple iterations of this process were done doing various additional experiments with hyperparameters, different models, different inference combinations etcetera.

Stage 1:
- Preprocess official Kaggle Train data and first External Data Set.
- Upload data to Kaggle and create public data set (assuming you have some good hardware...you can use the data on your PC at home)
- Perform 3 - 5 folds StratifiedKFold training of the model. Depending on the run-time for training a complete fold I usually trained only a few folds. Kaggle has a limit of 9 hours for a Notebook and session limit on Google Colab Pro is 12 hours.
- Depending on the validation score for each fold some would be submitted to determine the Public LB score.
- Based on the best scoring folds (based on local CV score and Public LB score) a few folds would be combined into an ensemble. 

Stage 2:
In stage 2 pseudo-labelling would be used to create additional training data and again perform training of the different models.
- Based on the best scoring (Local CV and Public LB) ensemble from stage 1 the predictions would be made for the stage 2 datasets.
- With the predictions for the stage 2 datasets the additional training data would be preprocessed.
- Upload the additional training data and create public data set.
- Perform 5 folds StratifiedKFold training of the model. In this stage all 4 datasources would be used. Depending on the run-time for training a complete fold I usually trained only a few folds. Kaggle has a limit of 9 hours for a GPU/TPU Notebook and the session limit on Google Colab Pro is 12 hours.
- Depending on the validation score for each fold some would be submitted to determine the Public LB score.
- Based on the best scoring folds (Local CV and Public LB score) a few folds would be combined into an ensemble and selected for the final submission. 

## Inference

I started the competition with a very simple inference pipeline containing only the following. The main advantage being that it takes little time for inference:
- Single segmentation model
- No TTA
- Use of a fixed threshold to generate the final mask

Eventually towards the end of the competition the inference pipeline contained the following components:
- Ensemble of 3 EffNet B4 models as backbone and FPN, Unet or Linknet as segmentation framework.
- Patches 2560 pixels scaled to 1280 pixels. 512 pixels used as overlap.
- Original image with 3 fold TTA (Horizontal flip, vertical flip and transpose)
- DenseCRF was used to generate the final mask for each patch based on the probabilities.

## Hardware

For hardware I used a combination of my own Data Science desktop or an Azure VM mostly for preprocessing the datasets as required in Kaggle.

For training the models I could only limited try this out on my desktop with a 1070Ti. The majority of training the models was done on Google Colab Pro TPUv2 notebooks. The remainder of the training was done on Kaggle Kernel TPUv3 notebooks.

For inference the Kaggle GPU Notebooks were used.

## Things tried that didn't work

- Use gaussian weighthing to compensate for any segmentation prediction border effects. In combination with or without overlap this only gave slightly worse results. After I started using DenseCRF this gave a very nice improvement (also on the borders..based on visual inspections) and I basically skipped trying to improve with gaussian weighting.
- Using finetuned backbones. Backbones finetuned as classifier (mask / no mask) for the kidney patches did lead to a quicker convergence occassionally..however the achieved Dice score was almost always less than when using a default backbone that was only pretrained on 'imagenet'.
- Trying more advanced segmentation models like Unet 3+ and Attention U-Net. The scores achieved were less than those achieved with FPN, Unet and Linknet.
- Pseudo labelling seemed to give only a very limited effect.

## Results

After the competition ended I was able to perform some isolated training and inference runs to see the difference between the 2 stages and the various inference options. Note that because of the randomness within the deep learning frameworks and the random sampling these results can vary between different training rounds. They are only ment to give an indication of the performance. When showing the performance of the single fold scores I use the mean of all 5 folds.

Below an overview of the inference results for a 5-fold cross validation training run with the following main information:
- Segmentation Framework: Unet
- Backbone: EfficientNet B0
- Epochs: 40
- Batchsize: 24
- Learning Rate (with custom scheduler): 0.0001
- Patches 1024 resized to 256

For inference a fixed threshold of 0.4 or DenseCRF was used. Inference was on patches of 1024 pixs resized to 256 pixs.

| Inference Setup | Stage 1 Score | Stage 2 Score |
|:---------------|----------------:|----------------:|
| Average of single models - No TTA - Fixed Threshold | 0.9193 | 0.9275 |
| Average of single models - TTA - Fixed Threshold | 0.9254 | 0.9308 |
| Average of single models - TTA - DenseCRF | 0.9300 | 0.9341 |
| Ensemble (5 folds) - No TTA - Fixed Threshold | 0.9288 | 0.9332 |
| Ensemble (5 folds) - TTA - Fixed Threshold | 0.9316 | 0.9352 |
| Ensemble (5 folds) - No TTA - DenseCRF | 0.9331 | 0.9360 |
| Ensemble (5 folds) - TTA - DenseCRF | 0.9351 | 0.9384 |

Below an overview of a second experiment with the inference results for a 5-fold cross validation training run with the following main information:
- Segmentation Framework: FPN
- Backbone: EfficientNet B4
- Epochs: 35
- Batchsize: 48
- Learning Rate (with custom scheduler): 0.0002
- Patches 1024 resized to 512

For inference a fixed threshold of 0.4 or DenseCRF was used. Inference was on patches of 1024 pixs resized to 512 pixs. For the ensemble with TTA and DenseCRF only 3 folds could be used because of the Kaggle Notebook time limits.

| Inference Setup | Stage 1 Score | Stage 2 Score |
|:---------------|----------------:|----------------:|
| Average of single models - No TTA - Fixed Threshold | 0.9335 | 0.9365 |
| Average of single models - TTA - Fixed Threshold | 0.9361 | 0.9380 |
| Average of single models - TTA - DenseCRF | 0.9391 | 0.9408 |
| Ensemble (5 folds) - No TTA - Fixed Threshold | 0.9393 | 0.9400 |
| Ensemble (5 folds) - TTA - Fixed Threshold | 0.9401 | 0.9408 |
| Ensemble (5 folds) - No TTA - DenseCRF | 0.9424 | 0.9427 |
| Ensemble (**3 folds) - TTA - DenseCRF | 0.9434 | 0.9433 |

Looking at the results it is interresting to note that for the smaller EfficientNet B0 backbone pseudo-labelling and retraining with the additional labelled data has some positive effect but when training with an EfficientNet B4 backbone there is virtually no difference.