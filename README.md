# Emotion_Image_Classifier
 My Capstone Project for my MS Data Science degree.  Classifying emotion in images.

## Motivation
This project was completed for a Capstone class at The George Washington University.  I was tasked with completing a data related project
of my choosing from start to finish.  I chose to focus on the task of classifying emotion in images using convolutional neural networks.

## Data Source
The data for this project comes from researchers at the University of Denver.  Details about the data source can be found in
this paper: https://arxiv.org/pdf/1708.03985.pdf . The dataset essentially consists of 1 million images labeled for human emotion.
Refer to the paper to see how to request the data from the authors. This repository does not contain the actual data.

## Preparing Data
Much of the time spent on this project involved getting the data into the correct place to be used for modeling.

### get_data.py
Gets the image files from my Google Drive to a Google Cloud VM.  When data is requested from the University of Denver, the images are
given in a OneDrive.  I manually moved the files to my own Google Drive before I started writing code.

### consolidate_data.py
Gets all images in decompressed folders into one directory.

### get_filenames_and_labels.py
Gets all filenames and labels from the provided csv files and saves them as npy files.

### inspect_images.py
Show images and labels in dataset selected by index.

### train_valid_test_clean
Loads the npy files and splits the labels and filenames into train, test, and validation sets.  Also cleans up unwanted
categories.

## Modeling

### custom_generator.py
Instantiates custom generators to load image data in batches for modeling.

### model_4.py
Custom VGG Type CNN.  Refer to report for details.

### model_evaluation_4.py
Evaluates Custom VGG Type CNN on test set.

### model_6.py
Pre-trained ResNet50 model.  Refer to report for details.

### model_evaluation_6.py
Evaluates Pre-trained ResNet50 model on test set.

## Demo

### demo.ipynb
This Jupyter Notebook shows some results from using the model on images from the internet.  Feel free to create an "images"
subdirectory.  The model will loop through it and classify the emotions present.
