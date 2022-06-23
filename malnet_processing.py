import os
from pathlib import Path

file_path_read = Path(os.environ["ICHOR_INPUT_DATASET"]) / "malnet-images-tiny"
print('The details of the current directory files.',os.listdir(file_path_read))
