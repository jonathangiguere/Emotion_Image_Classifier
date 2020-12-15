import numpy as np
import os
import pandas as pd
from keras.utils import to_categorical

# Read in training csv
train_df = pd.read_csv('/home/ubuntu/capstone/csv_files/training.csv', usecols=['subDirectory_filePath', 'expression'],
                       dtype={'subDirectory_filePath': 'string', 'expression': 'int16'}) #int32 if breaks

# Read in validation csv
valid_df = pd.read_csv('/home/ubuntu/capstone/csv_files/validation.csv', usecols=['subDirectory_filePath', 'expression'],
                       dtype={'subDirectory_filePath': 'string', 'expression': 'int16'})

# Remove sub folder at beginning of image name
train_df['subDirectory_filePath'] = train_df['subDirectory_filePath'].map(lambda x: x.split('/', 1)[-1])
valid_df['subDirectory_filePath'] = valid_df['subDirectory_filePath'].map(lambda x: x.split('/', 1)[-1])

# Combine dataframes
combined_df = train_df.append(valid_df)

# Get all filenames and labels as seperate lists
filenames = list(combined_df['subDirectory_filePath'])
labels = list(combined_df['expression'])

# One hot vector representation of labels
y_labels_one_hot = to_categorical(labels)

# Save to use later
np.save('filenames.npy', filenames)
np.save('labels.npy', y_labels_one_hot)

print(len(filenames))
print(y_labels_one_hot.shape)
