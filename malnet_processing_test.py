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
X_test = []
Y_test_fine = []
Y_test_coarse = []
for dirs in os.listdir(test_dir):
    for sub_class in os.listdir(test_dir / dirs):
        for image_file in os.listdir(test_dir / dirs / sub_class):
            im = Image.open(test_dir / dirs / sub_class / image_file, 'r')
            print(f'Processing for dir {dirs}, subdir {sub_class}, file {image_file}.')
            X_test.append(np.asarray(im))
            Y_test_fine.append(sub_class)
            Y_test_coarse.append(dirs)
print('Completed processing for training dataset.')
import pickle
test_dataset = {}
test_dataset['X_test'] = X_test
test_dataset['Y_test_fine'] = Y_test_fine
test_dataset['Y_test_coarse'] = Y_test_coarse
print('Now pickling.')
fp = open(file_path_read/"malnet_test", "bw+")  
pickle.dump(test_dataset, fp)   
fp.close()
