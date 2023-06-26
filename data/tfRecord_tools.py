import tensorflow as tf

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