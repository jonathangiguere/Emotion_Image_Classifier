import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


all_labels = np.load("/home/ubuntu/capstone/labels.npy")

path = '/home/ubuntu/capstone/all_images/'

# get list of all file names
img_filename_list = []
for img_file in os.listdir(path):
    img_filename_list.append(img_file)

# specify which images to inspect
slice_idx_1 = 300000
slice_idx_2 = 300050
files_to_inspect = img_filename_list[slice_idx_1:slice_idx_2]
labels_to_inspect = all_labels[slice_idx_1:slice_idx_2]

label_defs = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust',
              'Anger', 'Contempt', 'None', 'Uncertain', 'Non-Face']

# show specified files
for i, _ in enumerate(files_to_inspect):
    image = cv2.imread(path + _)[:, :, ::-1] # this gets RGB channels
    plt.imshow(image)
    label_arr = list(labels_to_inspect[i])
    label_idx = label_arr.index(1)
    plt.title(label_defs[label_idx])
    plt.show()