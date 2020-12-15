import numpy as np
from sklearn.model_selection import train_test_split

# Load file names and labels
x, y = np.load("/home/ubuntu/capstone/filenames.npy"), np.load("/home/ubuntu/capstone/labels.npy")

print(x.shape)
print(y.shape)

# Loop through labels and keep track of indices where the non-faces are
# Also drop None and Uncertain categories
# Also drop Contempt and Disgust categories
drop_indices = []
for _ in range(len(y)):
    if y[_][10] == 1 or y[_][9] == 1 or y[_][8] == 1 or y[_][7] == 1 or y[_][5] == 1: # or y[_][4] ... add back to drop Fear
        drop_indices.append(_)

# Drop label rows where indices match
y = np.delete(y, drop_indices, axis=0)
y = np.delete(y, 10, axis=1) # Drop last column because all vals are 0 after removing non-face rows
y = np.delete(y, 9, axis=1) # Do the same for None and Uncertain categories
y = np.delete(y, 8, axis=1)
y = np.delete(y, 7, axis=1) # Do the same for Contempt and Disgust categories
y = np.delete(y, 5, axis=1)
#y = np.delete(y, 4, axis=1) # Do the same for Fear category

# Drop image names where indices match
x = np.delete(x, drop_indices)

print(len(drop_indices))

# Get validation set 500 per category
def get_indices_valid(label):
    valid_indices = []
    for _ in range(len(y)): # Loop through all labels
        if len(valid_indices) < 500: # Get 500 indices for the label
            if y[_][label] == 1:
                valid_indices.append(_)
    return valid_indices

# Get 500 indices for all categories
valid_indices = []
for _ in range(6):
    valid_indices = valid_indices + get_indices_valid(_)

# Take indices identified as validation data
y_valid = np.take(y, valid_indices, axis=0)

# Take indices from the input data as well
X_valid_filenames = np.take(x, valid_indices)

# Drop the validation data from original data
y = np.delete(y, valid_indices, axis=0)
x = np.delete(x, valid_indices)

# Now get test data with train test split...0.00924 gives us the same test size and validation size of 2500
X_train_filenames, X_test_filenames, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.01073, stratify=y)

print('Train Final:')
print(X_train_filenames.shape)
print(y_train.shape)
print(np.sum(y_train, axis=0))
print()

print('Valid Final:')
print(X_valid_filenames.shape)
print(y_valid.shape)
print(np.sum(y_valid, axis=0))
print()

print('Test Final:')
print(X_test_filenames.shape)
print(y_test.shape)
print(np.sum(y_test, axis=0))
print()

# Save all data to numpy files in new directory
# Train
np.save('/home/ubuntu/capstone/train_test_valid/X_train_filenames.npy', X_train_filenames)
np.save('/home/ubuntu/capstone/train_test_valid/y_train.npy', y_train)

# Valid
np.save('/home/ubuntu/capstone/train_test_valid/X_valid_filenames.npy', X_valid_filenames)
np.save('/home/ubuntu/capstone/train_test_valid/y_valid.npy', y_valid)

# Test
np.save('/home/ubuntu/capstone/train_test_valid/X_test_filenames.npy', X_test_filenames)
np.save('/home/ubuntu/capstone/train_test_valid/y_test.npy', y_test)
