import collections
import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.layers import TextVectorization

import os

files_dir = "./texts"

def load_and_preprocess_texts(file_path):
    with open(file_path, 'r') as file:
        files_content = file.read()
    return files_content

files_paths = [os.path.join(files_dir, filename) for filename in os.listdir(files_dir)]

files_data = [load_and_preprocess_texts(file_path) for file_path in files_paths]

dataset_dir = pathlib.Path(files_dir)
list(dataset_dir.iterdir())
train_dir = dataset_dir/'train'
list(train_dir.iterdir())