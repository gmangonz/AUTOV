import tensorflow as tf
from tensorflow.io import FixedLenFeature, parse_single_example
import os

data_types = {
    "steering_angle": tf.float32,
    "speed": tf.float32,
    "X": tf.uint8,
}

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

def read_data(data_dir):

    files = os.listdir(data_dir)
    files = list(map(lambda x: os.path.join(data_dir, x), files))

    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
                                 cycle_length=len(files), block_length=1)
    dataset = dataset.map(lambda x: _parse_function(x, data_types=data_types))
    dataset = dataset.batch(128)

    ds = dataset.take(1)
    return ds

def read_tfrecord(tfrecord_path, return_values=False, coco=False):

    feature_dict = {}
    for rec in tf.data.TFRecordDataset([str(tfrecord_path)]):

        example_bytes = rec.numpy()
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
        
        for key in example.features.feature:

            feature = example.features.feature[key]
            if feature.HasField('bytes_list'):
                values = feature.bytes_list.value if return_values else tf.io.FixedLenFeature([], dtype=tf.string)
            elif feature.HasField('float_list'):
                values = feature.float_list.value if return_values else tf.io.FixedLenFeature([], dtype=tf.float32)
            elif feature.HasField('int64_list'):
                values = feature.int64_list.value if return_values else tf.io.FixedLenFeature([], dtype=tf.int64)
            else:
                values = feature.WhichOneof('kind')
                
            feature_dict[key] = values

    return feature_dict