import os
from pathlib import Path
import pickle

train_data_file = Path(os.environ["ICHOR_INPUT_DATASET"]) / 'malnet-images-tiny' / 'malnet_train'
train_data = pickle.load(train_data_file)
X_train = train_data['X_train']
Y_train_fine = train_data['Y_train_fine']
Y_train_coarse = train_data['Y_train_coarse']

test_data_file = Path(os.environ["ICHOR_INPUT_DATASET"]) / 'malnet-images-tiny' / 'malnet_test'
test_data = pickle.load(test_data_file)
X_test = train_data['X_test']
Y_test_fine = train_data['Y_test_fine']
Y_test_coarse = train_data['Y_test_coarse']

print('Train set', X_train.shape, Y_train_fine.shape, Y_train_coarse.shape)
print('Test set', X_test.shape, Y_test_fine.shape, Y_test_coarse.shape)
