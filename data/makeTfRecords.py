import h5py
import numpy as np
import tensorflow as tf
from tqdm.autonotebook import tqdm
import os
from tensorflow.io import FixedLenFeature, parse_single_example

all_data = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-01-30--11-24-51.h5', # 8 GB xx
            '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-01-30--13-46-00.h5', # 9 GB xx
            '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-01-31--19-19-25.h5', # 3 GB xx
            '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-02-02--10-16-58.h5', # 8.5 GB xx
            '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-02-08--14-56-28.h5', # 4 GB xx
            '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-02-11--21-32-47.h5', # 13 GB xX
            '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-03-29--10-50-20.h5', # 12 GB x
            '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-04-21--14-48-08.h5', # 4.6 GB xx
            '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-05-12--22-20-00.h5', # 7.8 GB xx
            '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-06-02--21-39-29.h5', # 6.75 GB xx
            '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-06-08--11-46-01.h5'] # 2.7 GB xx

all_labels = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-01-30--11-24-51.h5', 
              '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-01-30--13-46-00.h5', 
              '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-01-31--19-19-25.h5', 
              '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-02-02--10-16-58.h5', 
              '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-02-08--14-56-28.h5', 
              '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-02-11--21-32-47.h5', 
              '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-03-29--10-50-20.h5', 
              '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-04-21--14-48-08.h5', 
              '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-05-12--22-20-00.h5', 
              '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-06-02--21-39-29.h5', 
              '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-06-08--11-46-01.h5']

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(example):
    """Serialize an item in a dataset
    Arguments:
      example {[list]} -- list of dictionaries with fields "name" , "_type", and "data"

    Returns:
      [type] -- [description]
    """
    dset_item = {}
    for key in example.keys():
        dset_item[key] = example[key]["_type"](example[key]["data"])
        example_proto = tf.train.Example(features=tf.train.Features(feature=dset_item))
    return example_proto.SerializeToString()

def makeTFRecord(all_data, all_labels):

    """
    Convert .h5 files to TFRecord files

    """

    for data_file, label_file in list(zip(all_data, all_labels)):

        print('Currently on: ', data_file)
        # Open Files
        h5_file = h5py.File(data_file, 'r')
        h5_labels = h5py.File(label_file, 'r')

        # Extract Data
        mydata = h5_file['X'] # uint8 -> _bytes_feature
        speed_labels = np.array(h5_labels['speed']) # float64 -> _float_feature
        angle_labels = np.array(h5_labels['steering_angle']) # float64 -> _float_feature
        gear_labels = np.array(h5_labels['gear_choice']) # float64 -> _float_feature

        # Get acceleration
        speed_shifted = np.roll(speed_labels, 1)
        speed_shifted[0] = 0
        acc_labels = (speed_labels - speed_shifted)/0.05 # float64 -> _float_feature

        print('Previous Lengths')
        print(speed_labels.shape[0])

        assert speed_labels.shape[0] == angle_labels.shape[0] == gear_labels.shape[0] == acc_labels.shape[0], f'File {label_file} has components with different shapes'
        # Sample
        idxs = np.linspace(0, speed_labels.shape[0] - 1, mydata.shape[0]).astype("int")
        speed_labels = speed_labels[idxs]
        angle_labels = angle_labels[idxs]
        gear_labels = gear_labels[idxs]
        acc_labels = acc_labels[idxs]

        print('New Lengths')
        print(speed_labels.shape[0])

        # Number of items
        assert speed_labels.shape[0] == angle_labels.shape[0] == gear_labels.shape[0] == acc_labels.shape[0]
        num_of_items = mydata.shape[0]
        # Write TFRecords
        print('Writing TFRecords File')

        # Make data directory
        filepath = os.path.dirname(data_file.replace('camera', 'TFRecords/Data'))
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        with tf.io.TFRecordWriter(str(filepath.replace('.h5', '.tfrecord'))) as writer:

            for row in tqdm(range(num_of_items)):

                gear = gear_labels[row]
                if gear == 0 or gear == 10: # Reversing or Parked
                    continue
                speed = speed_labels[row]
                angle = angle_labels[row]        
                acc = acc_labels[row]
                img = mydata[row]
                img_resized = tf.image.resize(np.moveaxis(img, 0, -1), [100, 100]).numpy().astype('uint8')
                assert len(img_resized.shape) == 3, 'Should have 3D img'

                fields = {
                    'X': {'data': img_resized.flatten().tobytes(), '_type': _bytes_feature},
                    'speed': {'data': speed, '_type': _float_feature}, 
                    'steering_angle': {'data': angle, '_type': _float_feature},
                    'acceleration': {'data': acc, '_type': _float_feature}
                    }

                example = serialize_example(fields)
                writer.write(example)
        h5_file.close()
        h5_labels.close()