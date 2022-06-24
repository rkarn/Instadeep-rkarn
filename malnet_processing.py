import os
from pathlib import Path
from PIL import Image
import numpy as np

file_path_read = Path(os.environ["ICHOR_INPUT_DATASET"]) / "malnet-images-tiny" 
print('The details of the train directory files.',os.listdir(file_path_read/"train"))
print('The details of the test directory files.',os.listdir(file_path_read/"test"))
print('The details of the val directory files.',os.listdir(file_path_read/"val"))

train_dir = Path(os.environ["ICHOR_INPUT_DATASET"]) / "malnet-images-tiny" / "train"
test_dir = Path(os.environ["ICHOR_INPUT_DATASET"]) / "malnet-images-tiny" / "test"
val_dir = Path(os.environ["ICHOR_INPUT_DATASET"]) / "malnet-images-tiny" / "val"
X_train = []
Y_train_fine = []
Y_train_coarse = []
for dirs in train_dir:
    for sub_class in os.listdir(train_dir / dirs):
        for image_file in os.listdir(train_dir / dirs / sub_class):
            im = Image.open(train_dir / dirs / sub_class / image_file, 'r')
            print(f'Processing for dir {dirs}, subdir {sub_class}, file {image_file}.')
            X_train.append(np.asarray(im))
            Y_train_fine.append(sub_class)
            Y_train_coarse.append(dirs)

import pickle
train_dataset = {}
train_dataset['X_train'] = X_train
train_dataset['Y_train_fine'] = Y_train_fine
train_dataset['Y_train_coarse'] = Y_train_coarse
with open(file_path_read/"malnet_train", "w+") as fp:  
    pickle.dump(train_dataset, fp)        
