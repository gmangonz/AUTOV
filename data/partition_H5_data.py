import h5py
import numpy as np
import os

Y_VALUE_1 = 'speed'
Y_VALUE_2 = 'steering_angle'

def create_CAM_dataset(basename, filename, img_files):

    """Partition multiple CAM files into a single CAM file"""

    assert isinstance(img_files, list), 'img_files must be a list'
    assert any(list(map(lambda x: not x.endswith('h5'), img_files))), 'img_files must containt cam files that end with .h5'

    new_filename = os.path.join(basename, filename)
    if not new_filename.endswith('.h5'):
        new_filename = new_filename + '.h5'

    with h5py.File(new_filename, mode='w') as h5fw:
        row1 = 0
        for h5name in img_files:
            print(f'Extracting from {h5name}')
            h5fr = h5py.File(h5name,'r') 
            dset1 = list(h5fr.keys())[0]
            arr_data = h5fr[dset1][:]
            dslen = arr_data.shape[0]
            cols = arr_data.shape[1]
            if row1 == 0: 
                h5fw.create_dataset('X', dtype="uint8",  shape=(dslen, cols, 160, 320), maxshape=(None, cols, 160, 320) )
            if row1+dslen <= len(h5fw['X']) :
                h5fw['X'][row1:row1+dslen,:, :, :] = arr_data[:]
            else :
                h5fw['X'].resize( (row1+dslen, cols, 160, 320) )
                h5fw['X'][row1:row1+dslen,:, :, :] = arr_data[:]
            row1 += dslen


def create_LOG_dataset(basename, filename, img_files, label_files):

    """Partition multiple LOG files into a single log file"""

    assert isinstance(img_files, list) and isinstance(label_files, list), 'img_files and label_files must be lists'
    assert any(list(map(lambda x: not x.endswith('h5'), img_files))), 'img_files must containt cam files that end with .h5'
    assert any(list(map(lambda x: not x.endswith('h5'), label_files))), 'label_files must containt log files that end with .h5'

    new_filename = os.path.join(basename, filename)
    if not new_filename.endswith('.h5'):
        new_filename = new_filename + '.h5'

    with h5py.File(new_filename, mode='w') as h5fw:
        row1 = 0
        for h5IMGS, h5name in list(zip(img_files, label_files)):
            print(f'Extracting from {h5name}')

            h5fr = h5py.File(h5name,'r') # Open labels file
            h5img = h5py.File(h5IMGS,'r') # Open corresponding camera file

            arr_data = h5fr[Y_VALUE_1][:] # Get speed
            arr_data2 = h5fr[Y_VALUE_2][:] # Get steering wheel

            print('Speed data type: ', arr_data.dtype)
            print('Steering Wheel data type: ', arr_data2.dtype)

            print('Original Length: ', arr_data.shape[0])
            assert arr_data.shape == arr_data2.shape, 'Lengths should be the same'
            
            idxs = np.linspace(0, arr_data.shape[0] - 1, h5img['X'].shape[0]).astype("int")

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

            row1 += dslen


### Files to use below, but they no longer exist cuz these were in google drive

# TRAINING
train_path_imgs = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-01-31--19-19-25.h5', # 2.93
                     '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-02-08--14-56-28.h5', # 3.81
                      '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-04-21--14-48-08.h5'] # 4.39
train_path_labels = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-01-31--19-19-25.h5',
                     '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-02-08--14-56-28.h5', 
                      '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-04-21--14-48-08.h5']


train_path_imgs_34 = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-02-11--21-32-47.h5', # 12.3 GB
                      '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-03-29--10-50-20.h5', # 11.28 GB
                      '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-05-12--22-20-00.h5', # 7.47 GB
                      '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-01-31--19-19-25.h5'] # 2.93
train_path_labels_34 = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-02-11--21-32-47.h5',
                        '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-03-29--10-50-20.h5',
                        '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-05-12--22-20-00.h5',
                        '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-01-31--19-19-25.h5'] 


train_path_imgs_51 = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-02-11--21-32-47.h5', # 12.3 GB
                      '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-03-29--10-50-20.h5', # 11.28 GB
                      '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-05-12--22-20-00.h5', # 7.47 GB
                      '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-01-31--19-19-25.h5', # 2.93
                      '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-01-30--13-46-00.h5', # 8.5
                      '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-02-02--10-16-58.h5'] # 8.06
train_path_labels_51 = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-02-11--21-32-47.h5',
                        '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-03-29--10-50-20.h5', 
                        '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-05-12--22-20-00.h5', 
                        '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-01-31--19-19-25.h5', 
                        '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-01-30--13-46-00.h5', 
                        '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-02-02--10-16-58.h5'] 
# VALIDATION

validation_path_imgs_s = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-06-08--11-46-01.h5'] # 2.64
validation_path_labels_s = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-06-08--11-46-01.h5']


validation_path_imgs_m = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-06-08--11-46-01.h5', # 2.64
                          '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-01-30--11-24-51.h5'] # 7.62
validation_path_labels_m = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-06-08--11-46-01.h5',
                            '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-01-30--11-24-51.h5']


validation_path_imgs_l = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-06-08--11-46-01.h5', # 2.64
                          '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-01-30--11-24-51.h5', # 7.62
                          '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-02-08--14-56-28.h5'] # 3.81
validation_path_labels_l = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-06-08--11-46-01.h5',
                            '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-01-30--11-24-51.h5',
                            '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-02-08--14-56-28.h5']

# TESTING
testing_path_imgs_s = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-04-21--14-48-08.h5'] # 4.39
testing_path_labels_s = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-04-21--14-48-08.h5'] 


testing_path_imgs_ml = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-04-21--14-48-08.h5', # 4.39
                        '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/camera/2016-06-02--21-39-29.h5'] # 6.45
testing_path_labels_ml = ['/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-04-21--14-48-08.h5',
                          '/content/drive/Shareddrives/ELEC 494 - Ω2Ω/Data/log/2016-06-02--21-39-29.h5']