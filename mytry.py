# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:19:34 2019

@author: siddh
"""

import os
import librosa
import pandas as pd
import numpy as np

#from scipy.io import wavfile

df = pd.read_csv("S:/ty sem2/13-62-EN50/ESC-50-master/meta/esc50.csv")
print(df.head())

subpath = "S:/ty sem2/13-62-EN50/ESC-50-master/audio/"
features = []

for index, row in df.iterrows():
    audiopath = os.path.join(os.path.abspath(subpath), str(row["filename"]))
    wave, sr = librosa.load(audiopath)
    mfccs = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=40)
    features.append([mfccs, row["category"]])

featuresdf = pd.DataFrame(features, columns=['feature', 'category'])
# print(features)
# pd.get_dummies(featuresdf["category"])
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.category.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

print(x_train.shape[0], x_train.shape[1], x_train.shape[2])
print(x_test.shape[0], x_test.shape[1], x_test.shape[2])

num_rows = 72
num_columns = 120
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2

# Construct model 
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100 * score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

history = model.fit(x_train, y_train, batch_size=64, epochs=1000, validation_data=None, callbacks=None)

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)
# Print test accuracy+
print('\n', 'Test accuracy:', score[1])

# list all data in history
print(history.history.keys())

# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(history)
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.title("Validation accuracy per epoch")
# plt.show()


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

