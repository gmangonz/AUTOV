import h5py
import numpy as np
import os

Y_VALUE_1 = 'speed'
Y_VALUE_2 = 'steering_angle'

# Training Data
train_small = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/training_CAM_small.h5'
train_small_labels = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/training_LOG_small.h5'

train_medium = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/training_CAM_medium.h5'
train_medium_labels = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/training_LOG_medium.h5'

train_large = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/training_CAM_large.h5'
train_large_labels = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/training_LOG_large.h5'

# Validation Data
val_small = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/val_CAM_small.h5'
val_small_labels = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/val_LOG_small.h5'

val_medium = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/val_CAM_medium.h5'
val_medium_labels = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/val_LOG_medium.h5'

val_large = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/val_CAM_large.h5'
val_large_labels = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/val_LOG_large.h5'

# Testing Data
test_small = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/testing_CAM_small.h5'
test_small_labels = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/testing_LOG_small.h5'

test_rest = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/testing_CAM_rest.h5'
test_rest_labels = '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/data_partitioned/testing_LOG_rest.h5'

def combine_h5(base_path, filename, files_to_combine):

    """Combine H5 camera data and log data into a single file"""

    assert isinstance(files_to_combine, list), 'files_to_combine should be a list'
    assert len(files_to_combine) == 2, 'files_to_combine should countiain two files'

    if filename.split('.')[-1] != 'h5':
        filename = filename + '.h5'

    new_file = os.path.join(base_path, filename)
    with h5py.File(new_file, mode='w') as h5fw:
        row1 = 0
        for h5IMGS, h5name in list(zip(files_to_combine)):

            h5img = h5py.File(h5IMGS,'r') # Open corresponding camera file
            cam_data = h5img['X']
            print('Camera length: ', cam_data.shape[0])

            h5fr = h5py.File(h5name,'r') # Open labels file
            arr_data = h5fr[Y_VALUE_1][:] # Get speed
            arr_data2 = h5fr[Y_VALUE_2][:] # Get steering wheel

            print('Speed data type: ', arr_data.dtype)
            print('Steering Wheel data type: ', arr_data2.dtype)

            print('Original Length: ', arr_data.shape[0])
            assert arr_data.shape == arr_data2.shape, 'Lengths should be the same'
            
            idxs = np.linspace(0, arr_data.shape[0] - 1, cam_data.shape[0]).astype("int") # Return evenly spaced numbers (number of numbers: x.shape[0]) over a specified interval (0, steering_angle.shape[0]-1)

            arr_data = arr_data[idxs]
            arr_data2 = arr_data2[idxs]

            dslen = arr_data.shape[0]

            print('New Length: ', dslen)

            if row1 == 0: 
                h5fw.create_dataset('speed', dtype="<f8",  shape=(dslen,), maxshape=(None,))
                h5fw.create_dataset('steering_angle', dtype="<f8",  shape=(dslen,), maxshape=(None,))

            if row1+dslen <= len(h5fw['speed']) :
                h5fw['speed'][row1:row1+dslen,] = arr_data[:]
                h5fw['steering_angle'][row1:row1+dslen,] = arr_data2[:]
            else:
                h5fw['speed'].resize( (row1+dslen,) )
                h5fw['speed'][row1:row1+dslen,] = arr_data[:]

                h5fw['steering_angle'].resize( (row1+dslen,) )
                h5fw['steering_angle'][row1:row1+dslen,] = arr_data2[:]

            print(h5fw['speed'].dtype)
            row1 += dslen