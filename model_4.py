import os
import random
import numpy as np
import tensorflow as tf
from custom_generator import JG_Generator

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization

import pandas as pd
import matplotlib.pyplot as plt

from keras import backend as K

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set matplotlib sizes
plt.rc('font', size=20)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=20)
plt.rc('figure', titlesize=20)

# Set seed for reproducible results
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Data Prep
# Load file names and labels
X_train_filenames = np.load('/home/ubuntu/capstone/train_test_valid/X_train_filenames.npy')
y_train = np.load('/home/ubuntu/capstone/train_test_valid/y_train.npy')

X_valid_filenames = np.load('/home/ubuntu/capstone/train_test_valid/X_valid_filenames.npy')
y_valid = np.load('/home/ubuntu/capstone/train_test_valid/y_valid.npy')

################################################################################################################
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(6))
model.add(Activation('softmax'))

################################################################################################################

# Checkpoint callback
checkpoint_cb = keras.callbacks.ModelCheckpoint('/home/ubuntu/capstone/models/model_4.h5',
                                                save_best_only=True)

# Early stopping callback
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

# Create custom loss function
def weighted_categorical_crossentropy(weights):

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

# Set Flag for weighted or regular loss
loss_flag = 'weights'

if loss_flag == 'no_weights':
    weights = np.array([1, 1, 1, 1, 1, 1])
elif loss_flag == 'weights':
    weights = np.array([0.802518525169482, 0.802092227896231, 0.866721731456969,
                        1.0450040554419, 1.6127733638543, 0.870890096181114])

# Compile model in correct mode
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=weighted_categorical_crossentropy(weights),
              metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

# Instantiate generators for feeding in data
batch_size = 64
my_training_batch_generator = JG_Generator(X_train_filenames, y_train, batch_size)
my_validation_batch_generator = JG_Generator(X_valid_filenames, y_valid, batch_size)

# Call the custom generators and train the model
history = model.fit_generator(generator=my_training_batch_generator,
                   steps_per_epoch=int(len(X_train_filenames) // batch_size),
                   epochs=5,
                   verbose=1,
                   validation_data=my_validation_batch_generator,
                   validation_steps=int(3000 // batch_size),
                    callbacks=[checkpoint_cb, early_stopping_cb])

print('Training Complete')

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))

# Set grid
plt.grid(True)

# Show the figure
plt.tight_layout()
plt.show()
