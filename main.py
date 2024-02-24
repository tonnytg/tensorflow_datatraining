import pathlib
import ssl
import urllib.request
import tensorflow as tf
from tensorflow.keras import utils

# Desabilitar verificação SSL
ssl._create_default_https_context = ssl._create_unverified_context

data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'

dataset_dir = utils.get_file(
    origin=data_url,
    untar=True,
    cache_dir='stack_overflow',
    cache_subdir='')

dataset_dir = pathlib.Path(dataset_dir).parent
list(dataset_dir.iterdir())
train_dir = dataset_dir/'train'
list(train_dir.iterdir())
sample_file = train_dir/'python/1755.txt'

with open(sample_file) as f:
  print(f.read())