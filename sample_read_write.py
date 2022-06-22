import os
from pathlib import Path

file_path_read = Path(os.environ["ICHOR_INPUT_DATASET"]) / 'sample01.txt'
f = open(file_path_read, 'r')
file_content = f.readlines()
print(file_content)
f.close()

file_path_write = Path(os.environ["ICHOR_INPUT_DATASET"]) / 'sample_write.txt'
f = open(file_path_write, 'w+')
f.write('Hello World. I have written succesfully.')
f.close()
