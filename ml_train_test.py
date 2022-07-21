import os
from pathlib import Path
import pickle
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras import datasets, layers, models
from keras.utils import np_utils
from keras import regularizers
from keras.layers import Dense, Dropout, BatchNormalization

train_data_file = Path(os.environ["ICHOR_INPUT_DATASET"]) / 'malnet-images-tiny' / 'malnet_train'
with open(train_data_file, 'rb') as pickle_file_train:
    train_data = pickle.load(pickle_file_train)
print('Train pickle file loaded.')
X_train = np.array(train_data['X_train'])
Y_train_fine = np.array(train_data['Y_train_fine'])
Y_train_coarse = np.array(train_data['Y_train_coarse'])

test_data_file = Path(os.environ["ICHOR_INPUT_DATASET"]) / 'malnet-images-tiny' / 'malnet_test'
with open(test_data_file, 'rb') as pickle_file_test:
    test_data = pickle.load(pickle_file_test)
print('Test pickle file loaded.')
X_test = np.array(test_data['X_test'])
Y_test_fine = np.array(test_data['Y_test_fine'])
Y_test_coarse = np.array(test_data['Y_test_coarse'])

print('Train set', X_train.shape, Y_train_fine.shape, Y_train_coarse.shape)
print('Test set', X_test.shape, Y_test_fine.shape, Y_test_coarse.shape)

print('Working with small dataset and Y_coarse')
from sklearn.model_selection import train_test_split
X_train, _, Y_train, _ = train_test_split(X_train, Y_train_coarse, test_size=0.9, random_state=42)
_, X_test, _, Y_test = train_test_split(X_test, Y_test_coarse, test_size=0.1, random_state=42)

print('Normalizaing.')
X_train=X_train/255.0
X_test=X_test/255.0


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit_transform(Y_train)
Y_train = le.transform(Y_train)
Y_test = le.transform(Y_test)

le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
num_classes = max(le_name_mapping.values())+1
print(f'Number of classes = {num_classes}')

model = Sequential()

model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(256,256,3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))    # num_classes = 10

# Checking the model summary
print(model.summary())

Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=64, epochs=100,
                    validation_data=(X_test, Y_test))

pred = model.predict(X_test)
print(pred)

# Converting the predictions into label index 
pred_classes = np.argmax(pred, axis=1)
print('Encoded classes predicted',pred_classes)
print('Natural label predicted', le.inverse_transform(pred_classes))
