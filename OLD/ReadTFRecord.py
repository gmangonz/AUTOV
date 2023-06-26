import os
import h5py
import numpy as np
import tensorflow as tf
from tqdm.autonotebook import tqdm
from tensorflow.io import FixedLenFeature, parse_single_example
import mathplotlib.pyplot as plt

raw_dataset = tf.data.TFRecordDataset([str("/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/TFRecords/Data_2/2016-01-30--11-24-51.tfrecord")])

def _dtype_to_tf_feattype(dtype):
    """ convert tf dtype to correct tffeature format
    """
    if dtype in [tf.float32, tf.int64]:
        return dtype
    else:
        return tf.string

def _parse_function(example_proto, data_types):
    """ parse dataset from tfrecord, and convert to correct format
    """
    # list features
    features = {
        lab: FixedLenFeature([], _dtype_to_tf_feattype(dtype))
        for lab, dtype in data_types.items()
    }
    # parse features
    parsed_features = parse_single_example(example_proto, features)
    feat_dtypes = [tf.float32, tf.string, tf.int64]
    
    # convert the features if they are in the wrong format
    parse_list = [
        parsed_features[lab]
        if dtype in feat_dtypes
        else tf.io.decode_raw(parsed_features[lab], dtype)
        for lab, dtype in data_types.items()
    ]
    return parse_list

data_types = {
    "steering_angle": tf.float32,
    "speed": tf.float32,
    "X": tf.uint8,
}

files = os.listdir('/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/TFRecords/Data_2')
files = list(map(lambda x: os.path.join('/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/TFRecords/Data_2', x), files))
files = tf.data.Dataset.from_tensor_slices(files)
dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x),
                           cycle_length=11, block_length=1)

dataset = dataset.map(lambda x: _parse_function(x, data_types=data_types))
dataset = dataset.shuffle(buffer_size=100000)
dataset = dataset.batch(128)

ds = iter(dataset)
angle, speed, x  = next(ds)

x = tf.reshape(x, [128, 100, 100, 3])
plt.imshow(x[120])