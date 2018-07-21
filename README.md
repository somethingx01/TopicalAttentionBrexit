# TopicalAttentionBrexit
An opinion dynamic tracing model on dataset I
For code on dataset II please redirect to https://github.com/somethingx01/TopicalAttentionElection

=============
Downloading large datafiles
-------------
Download dataset

#.7z 
This repository should include preprocessed datasets organized in epochs, which are essential for for training and prediction. However, github doesn't allow large files to be upload, so you have to manually download the dataset and place them in the ./datasets/ directory. Please click xxxxxxxxxxxxxx to download the .7z file.

#.7z

-------------
Download saved model

-------------
Calculate dataset statistics

#Dataset Statistics:
Run the main_sentiment_instance_counts_epochs.py. Note that the dataset epochs are organized in an time-descending order. That is, SNSTtrain_08 for epoch 0, SNStrain_07 for epoch 1, etc.



=============
Training and prediction
-------------
Training

#A Quick Start
If your machine is not eligible for a training (CUDA 7.0+ with 8G+GPURAM, 100G RAM), then loading a trained model ( by commenting train() ) and performing prediction would be suggested, whether on CPU or GPU.

-------------
Prediction

#Predict an epoch from trained models:
The defult code is predicting EPOCH_8.

#Understandable Topics:
