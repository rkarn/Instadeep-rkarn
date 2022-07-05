import os
from pathlib import Path
import pickle

train_data_file = Path(os.environ["ICHOR_INPUT_DATASET"]) / 'malnet-images-tiny' / 'malnet_train'
with open(train_data_file, 'rb') as pickle_file_train:
    train_data = pickle.load(pickle_file_train)
print('Train pickle file loaded.')
X_train = train_data['X_train']
Y_train_fine = train_data['Y_train_fine']
Y_train_coarse = train_data['Y_train_coarse']

test_data_file = Path(os.environ["ICHOR_INPUT_DATASET"]) / 'malnet-images-tiny' / 'malnet_test'
with open(test_data_file, 'rb') as pickle_file_test:
    test_data = pickle.load(pickle_file_test)
print('Test pickle file loaded.')
X_test = test_data['X_test']
Y_test_fine = test_data['Y_test_fine']
Y_test_coarse = test_data['Y_test_coarse']

print('Train set', X_train.shape, Y_train_fine.shape, Y_train_coarse.shape)
print('Test set', X_test.shape, Y_test_fine.shape, Y_test_coarse.shape)
