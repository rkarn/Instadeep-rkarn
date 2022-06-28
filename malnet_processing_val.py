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
X_val = []
Y_val_fine = []
Y_val_coarse = []
for dirs in os.listdir(val_dir):
    for sub_class in os.listdir(val_dir / dirs):
        for image_file in os.listdir(val_dir / dirs / sub_class):
            im = Image.open(val_dir / dirs / sub_class / image_file, 'r')
            print(f'Processing for dir {dirs}, subdir {sub_class}, file {image_file}.')
            X_val.append(np.asarray(im))
            Y_val_fine.append(sub_class)
            Y_val_coarse.append(dirs)
print('Completed processing for training dataset.')
import pickle
val_dataset = {}
val_dataset['X_val'] = X_val
val_dataset['Y_val_fine'] = Y_val_fine
val_dataset['Y_val_coarse'] = Y_val_coarse
print('Now pickling.')
fp = open(file_path_read/"malnet_val", "bw+")  
pickle.dump(val_dataset, fp)   
fp.close()
