# Kaggle HuBMAP - 'Hacking the Kidney'

## Introduction

Recently I participated in the Kaggle Data Science competition 'Hacking the Kidney'. It required semantic segmentation to predict the functional tissue units within kidney images.

With a best score of 0.9437 and a final submission score of 0.9419 for an ensemble of Unet, FPN and Linknet segmentation models. With multiple ensembles scoring within 1% from the leaderboard winning score I was really happy with the results achieved.

## Datasets

Training was performed in 2 stages. In the first stage the following data was used:
- Original Kaggle HuBMAP Training data.
- External data set [https://www.kaggle.com/baesiann/glomeruli-hubmap-external-1024x1024](https://www.kaggle.com/baesiann/glomeruli-hubmap-external-1024x1024)

In the second stage pseudo-labelling was used for the following data:
- The original Kaggle Public Test data.
- 2 additional external images from [https://portal.hubmapconsortium.org/](https://portal.hubmapconsortium.org/search?entity_type[0]=Dataset)

After pseudo-labelling the second stage data it could be used as additional training-data

## Training

The training was performed in 2 stages. For each stage the following actions were performed. Note that multiple iterations of this process were done doing various additional experiments with hyperparameters, different models, different inference etc.

Stage 1:
- Preprocess official Kaggle Train data and first External Data Set.
- Upload data to Kaggle and create public data set.
- Perform 3 - 5 folds StratifiedKFold training of the model. Depending on the run-time for training a complete fold I usually trained only a few folds. Kaggle has a limit of 9 hours for a Notebook and session limit on Google Colab Pro is 12 hours.
- Depending on the validation score for each fold some would be submitted to determine the Public LB score.
- Based on the best scoring folds (based on validation score and Public LB score) a few folds would be combined into an ensemble. 

Stage 2:
In stage 2 pseudo-labelling would be used to create additional training data and again perform training of the different models.
- Based on the ensemble from stage 1 the predictions would be made for the stage 2 datasets.
- With the predictions for the stage 2 datasets the additional training data would be preprocessed.
- Upload the additional training data and create public data set.
- Perform 3 - 5 folds StratifiedKFold training of the model. In this stage all 4 datasources would be used. Depending on the run-time for training a complete fold I usually trained only a few folds. Kaggle has a limit of 9 hours for a Notebook and session limit on Google Colab Pro is 12 hours.
- Depending on the validation score for each fold some would be submitted to determine the Public LB score.
- Based on the best scoring folds (based on validation score and Public LB score) a few folds would be combined into an ensemble and selected for the final submission. 


## Inference

I started the competition with a very simple inference pipeline containing only the following:
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

<< TODO >>

## Results

<< TODO >>