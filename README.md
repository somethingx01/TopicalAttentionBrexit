# TopicalAttentionBrexit
An opinion dynamic tracing model on dataset I
For code on dataset II please redirect to https://github.com/somethingx01/TopicalAttentionElection

Downloading large datafiles
-------------
Download dataset

## BrexitDataset.7z

* Epoched dataset \\ 
This repository should include preprocessed datasets organized in epochs, which are essential for training and prediction. However, github doesn't allow large files to be uploaded, so you have to manually download the dataset and place them in the ./datasets/ directory. Please click https://s3-ap-southeast-1.amazonaws.com/datasetandparams/dataset_brexit_and_params/BrexitDataset.7z to download the BrexitDataset.7z file.

* Tweets
If you are looking for the original tweets, i.e., 363961 tweets as mentioned in Table 1, please download the tweet IDs by https://datasetandparams.s3-ap-southeast-1.amazonaws.com/dataset_brexit_and_params/BrexitTweetIDs.txt

* UserID (mosaiced) - TweetIDs
If you are looking for the original user-tweet mapping, i.e., 38335 users and their tweets, please download the userID - tweetIDs by https://datasetandparams.s3-ap-southeast-1.amazonaws.com/dataset_brexit_and_params/Brexit_userid2tweetids_mosaiceduserid2tweetid Be aware that the userIDs are masked. That is to say, you may need to crawl the tweets to get the true auther/user id. 

* UserID (mosaiced) - FriendIDs (mosaiced)
If you are looking for the original user-friend relationships, i.e., 38335 users and their friends, please download the userID - friendIDs by https://datasetandparams.s3-ap-southeast-1.amazonaws.com/dataset_brexit_and_params/Brexit_userid2friendids_relationshipmosaiced

## BrexitParams.7z
The params are essential for setting the settings.py(e.g., Traning_Instance_Count, Testing_Instance_Count) every time you perform training or prediction. Please click https://s3-ap-southeast-1.amazonaws.com/datasetandparams/dataset_brexit_and_params/BrexitParams.7z to download the BrexitParams.7z file.

-------------
Download saved model

## model
'model' is the file name. 'model' is the trained and saved model, you can load it to reproduce the experiment results. Click https://s3-ap-southeast-1.amazonaws.com/datasetandparams/dataset_brexit_and_params/model to download the 'model' file. You should place it in the ./save/0/ directory so that the default 'prediction' function in train.py can detect it. 

-------------
Calculate dataset statistics

## Dataset Statistics:
Run the main_sentiment_instance_counts_epochs.py. Note that the dataset epochs are organized in an time-descending order. That is, SNSTtrain_08 for epoch 0, SNStrain_07 for epoch 1, etc.

## Text Content:
For dataset text content purposes, please redirect to https://github.com/somethingx01/TwitterCrawlerUsingIDs.

Training and prediction
-------------
Training

## A Quick Start
Call train() to perform training, whether on CPU or GPU.
<!--- # "If your machine is not eligible for a training (CUDA 7.0+ with 8G+GPURAM, 100G RAM), then loading a trained model ( by commenting call train() and perform training, whether on CPU or GPU. --->

-------------
Prediction

## Predict an epoch from trained models:
The defult code is predicting EPOCH_8 by by commenting train().

Bib entry
-------------
Please cite with the below bib entry if you reference the code in your research:
```
@article{zhu2019neural,
  title={Neural opinion dynamics model for the prediction of user-level stance dynamics},
  author={Zhu, Lixing and He, Yulan and Zhou, Deyu},
  journal={Information Processing \& Management},
  year={2019},
  publisher={Elsevier}
}
```
