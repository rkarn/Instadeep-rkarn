import os
from pathlib import Path

file_path_read = Path(os.environ["ICHOR_INPUT_DATASET"]) / "malnet-images-tiny" 
print('The details of the train directory files.',os.listdir(file_path_read/"train"))
print('The details of the test directory files.',os.listdir(file_path_read/"test"))
print('The details of the val directory files.',os.listdir(file_path_read/"val"))
