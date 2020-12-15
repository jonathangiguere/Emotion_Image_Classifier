from sklearn.metrics import classification_report, multilabel_confusion_matrix, confusion_matrix
import numpy as np
from tensorflow import keras
from custom_generator import JG_Generator

from keras import backend as K
import tensorflow as tf

# Load test set
X_test_filenames = np.load('/home/ubuntu/capstone/train_test_valid/X_test_filenames.npy')
y_test = np.load('/home/ubuntu/capstone/train_test_valid/y_test.npy')

X_test_filenames = X_test_filenames[:3000]
y_test = y_test[:3000]

# Load model
model = keras.models.load_model('/home/ubuntu/capstone/models/model_4.h5', compile=False)

# Include custom loss to compile the model again
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

# Compile Model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=weighted_categorical_crossentropy(weights), metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

batch_size = 64

my_test_batch_generator = JG_Generator(X_test_filenames, y_test, batch_size)

y_pred = model.predict_generator(generator=my_test_batch_generator,
                                 steps=int(3082 // batch_size),
                                 verbose=1)

# Get predictions from softmax output
y_pred_bool = np.argmax(y_pred, axis=1)

# Get non one hot encoded form of y_test
y_test = y_test.argmax(axis=1)

# Show confusion matrix
print(multilabel_confusion_matrix(y_test, y_pred_bool))

# Show other confusion matrix
print(confusion_matrix(y_test, y_pred_bool))

# Show classification report
print(classification_report(y_test, y_pred_bool))