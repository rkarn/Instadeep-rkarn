from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from tensorflow.keras import backend as K
print(K.tensorflow_backend._get_available_gpus())
