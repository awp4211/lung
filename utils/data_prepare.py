import SimpleITK as sitk
import numpy as np
from glob import glob
import random
import pandas as pd
import os
import gc
from tqdm import tqdm
from multiprocessing import Pool

from utils.lung_segmentation import segment_lung_mask


def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return f


def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def set_window_width(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    image[image > MAX_BOUND] = MAX_BOUND
    image[image < MIN_BOUND] = MIN_BOUND
    return image


def get_nodule_mask(image_arr, mini_df, spacing, origin):
    """
    get nodule's mask:mask is a 3d numpy array(dtype=np.uint8)
    :param image_arr: 3d image arr 
    :param mini_df: pandas dataframe
    :param spacing: x,y,z ordered mm/pix of each dimension
    :param origin:  x,y,z ordered world coordinates mm
    :return: 
    """
    num_z, height, width = image_arr.shape
    masks = np.zeros([num_z, height, width], dtype=np.uint8)
    if mini_df.shape[0]>0:
        for node_idx, cur_row in mini_df.iterrows():
            print '------------------------------------------------'
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            # nodule center
            center = np.array([node_x, node_y, node_z])
            # nodule center in voxel space (still x,y,z ordering)
            v_center = np.rint((center - origin) / spacing)
            vx_diam = int(diam / spacing[0])
            vy_diam = int(diam / spacing[1])
            vz_diam = int(diam / spacing[2])
            print 'nodule in %s,nodule index = %d, world coordinate = %s, diam_mm = %f' % (cur_row['file'], node_idx, center, diam)
            print 'nodule v_center = %s, vx_diam = %d, vy_diam = %d, vz_diam = %d' % (v_center, vx_diam, vy_diam, vz_diam)
            # slices_indexes = np.arange(int(v_center[2])-1,int(v_center[2])+2).clip(0, num_z-1)
            slices = np.arange(int(v_center[2] - vz_diam / 2), int(v_center[2] + vz_diam / 2) + 1).clip(0, num_z - 1)
            for slice in slices:
                # slice is the z_index of volumn
                mask = np.zeros([height, width], dtype=np.uint8)
                v_xmin = np.max([0, int(v_center[0] - np.rint(vx_diam / 2))])
                v_xmax = np.min([width - 1, int(v_center[0] + np.rint(vx_diam / 2))])
                v_ymin = np.max([0, int(v_center[1] - np.rint(vy_diam / 2))])
                v_ymax = np.min([height - 1, int(v_center[1] + np.rint(vy_diam / 2))])

                v_xrange = range(v_xmin, v_xmax + 1)
                v_yrange = range(v_ymin, v_ymax + 1)

                for v_x in v_xrange:
                    for v_y in v_yrange:
                        mask[v_x, v_y] = 1

                masks[slice] += mask.T
    masks = masks.clip(0, 1)
    return masks


def _is_destnation(mask_volume, ratio):
    """
    identify weather the current volume is lung or nodule  
    :param mask_volume: 
    :param ratio:
    :return: 
    """
    total_voxels = mask_volume.shape[0]*mask_volume.shape[1]*mask_volume.shape[2]+0.#to float
    lung_voxels = np.sum(mask_volume==1)+0.#to float
    if lung_voxels/total_voxels > ratio:
        return True
    else:
        return False


def _shuffle_data(dataset_x, dataset_y, shuffle_times=10000):
    for i in range(shuffle_times):
        random_i = random.randint(0, len(dataset_x) - 1)
        random_j = random.randint(0, len(dataset_x) - 1)
        if random_i != random_j:
            # swap dataset_x
            temp1 = dataset_x[random_i]
            dataset_x[random_i] = dataset_x[random_j]
            dataset_x[random_j] = temp1
            # swap dataset_y
            temp2 = dataset_y[random_i]
            dataset_y[random_i] = dataset_y[random_j]
            dataset_y[random_j] = temp2


def sliding_window_scan_train_or_val(mhd_file_list, df_node,
                                     sw_x=20, sw_y=20, sw_z=5,
                                     stride_x=10, stride_y=10, stride_z=1,
                                     islung_ratio=0.5,
                                     isnodule_ratio=0.1,
                                     sample_rate=0.0015):
    """
    sliding window scan training or validation dateset
    :param mhd_file_list:mhd list
    :param df_node: pandas data frame
    :param sw_x: 
    :param sw_y: 
    :param sw_z: 
    :param stride_x: 
    :param stride_y: 
    :param stride_z: 
    :param islung_ratio:
    :param isnodule_ratio:
    :return: 
    """
    volume_list = []
    world_center_list = []
    labels = []

    for file_index, img_file in enumerate(tqdm(mhd_file_list)):
        print '====================================================================='
        print 'slinding window scanning file %d:%s' % (file_index, img_file)
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (np.int16)
        #num_z, height, width = img_array.shape
        origin = np.array(itk_img.GetOrigin())# x,y,z Origin in world coordinates
        spacing = np.array(itk_img.GetSpacing())#x,y,z spacing mm/px

        print 'image world coordinate origin = %s, spacing = %s' % (origin, spacing)

        segmented_lung_mask = segment_lung_mask(img_array, True)
        mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
        nodules_mask = get_nodule_mask(image_arr=img_array, mini_df=mini_df, spacing=spacing, origin=origin)

        # normalize image array
        img_array = set_window_width(img_array)
        img_array = normalize(img_array)
        img_array = np.asarray(img_array, dtype='float32')

        for z in range(0, img_array.shape[0]-sw_z, stride_z):
            for y in range(0, img_array.shape[1]-sw_y, stride_y):
                for x in range(0, img_array.shape[2]-sw_x, stride_x):
                    current_img_volume = img_array[z:z+sw_z, y:y+sw_y, x:x+sw_x]
                    current_lung_volume = segmented_lung_mask[z:z+sw_z, y:y+sw_y, x:x+sw_x]
                    current_nodule_volume = nodules_mask[z:z+sw_z, y:y+sw_y, x:x+sw_x]
                    current_v_center = np.array([x+sw_x/2, y+sw_y/2, z+sw_z/2]) #x,y,z
                    world_center = current_v_center*spacing + origin
                    #print 'world_center = %s' % world_center
                    if _is_destnation(current_lung_volume, islung_ratio):
                        label = 0
                        if _is_destnation(current_nodule_volume, isnodule_ratio):
                            # positive samples to save
                            label = 1
                            volume_list.append(current_img_volume.flatten())
                            world_center_list.append(world_center)
                            labels.append(label)
                        else:
                            # sampling negative samples
                            if np.random.random_sample() < sample_rate:
                                volume_list.append(current_img_volume.flatten())
                                world_center_list.append(world_center)
                                labels.append(label)


        print 'volume_list len = %d' % len(volume_list)
        print 'positive samples count = %d' % np.sum(np.asarray(labels) == 1)
        print 'negative samples count = %d' % np.sum(np.asarray(labels) == 0)

    return volume_list, labels


def sliding_window_scan_test(mhd_file_list,
                            sw_x=20, sw_y=20, sw_z=5,
                            stride_x=10, stride_y=10, stride_z=1,
                            islung_ratio=0.5):
    volume_list = []
    world_center_list = []
    seriesuids = []
    array_index_center_list = []


    for file_index, img_file in enumerate(tqdm(mhd_file_list)):
        print '====================================================================='
        print 'slinding window scanning file %d:%s' % (file_index, img_file)
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (np.int16)
        # num_z, height, width = img_array.shape
        origin = np.array(itk_img.GetOrigin())  # x,y,z Origin in world coordinates
        spacing = np.array(itk_img.GetSpacing())  # x,y,z spacing mm/px

        print 'image world coordinate origin = %s, spacing = %s' % (origin, spacing)

        segmented_lung_mask = segment_lung_mask(img_array, True)

        # normalize image array
        img_array = set_window_width(img_array)
        img_array = normalize(img_array)
        img_array = np.asarray(img_array, dtype='float32')

        for z in range(0, img_array.shape[0] - sw_z, stride_z):
            for y in range(0, img_array.shape[1] - sw_y, stride_y):
                for x in range(0, img_array.shape[2] - sw_x, stride_x):
                    current_img_volume = img_array[z:z + sw_z, y:y + sw_y, x:x + sw_x]
                    current_lung_volume = segmented_lung_mask[z:z + sw_z, y:y + sw_y, x:x + sw_x]
                    current_v_center = np.array([x + sw_x / 2, y + sw_y / 2, z + sw_z / 2])  # x,y,z
                    world_center = current_v_center * spacing + origin
                    # print 'world_center = %s' % world_center
                    if _is_destnation(current_lung_volume, islung_ratio):
                        volume_list.append(current_img_volume.flatten())
                        world_center_list.append(world_center)
                        seriesuids.append(img_file[img_file.rfind('/')+1:img_file.rfind('.')])
                        array_index_center_list.append(current_v_center)

        print 'volume_list len = %d' % len(volume_list)
        print 'world_center_list len =%d' % len(world_center_list)

    return volume_list, world_center_list, seriesuids, array_index_center_list


def _sub_process(train_sub_mhd_file,
                 train_datasetx_filename,
                 train_datasety_filename,
                 val_sub_mhd_file,
                 val_datasetx_filename,
                 val_datasety_filename,
                 sw_x, sw_y, sw_z,
                 stride_x, stride_y, stride_z,
                 islung_ratio,
                 isnodule_ratio,
                 sample_rate
                 ):
    train_x, train_y = sliding_window_scan_train_or_val(train_sub_mhd_file, train_df_node,
                                                        sw_x=sw_x, sw_y=sw_y, sw_z=sw_z,
                                                        stride_x=stride_x, stride_y=stride_y, stride_z=stride_z,
                                                        islung_ratio=islung_ratio,
                                                        isnodule_ratio=isnodule_ratio,
                                                        sample_rate=sample_rate)
    save_data_2_np(train_x, train_y, train_datasetx_filename, train_datasety_filename)
    # release
    del train_x
    del train_y
    gc.collect()
    val_luna_subset_path
    val_x, val_y = sliding_window_scan_train_or_val(val_sub_mhd_file, val_df_node,
                                                        sw_x=sw_x, sw_y=sw_y, sw_z=sw_z,
                                                        stride_x=stride_x, stride_y=stride_y, stride_z=stride_z,
                                                        islung_ratio=islung_ratio,
                                                        isnodule_ratio=isnodule_ratio,
                                                        sample_rate=sample_rate)
    save_data_2_np(val_x, val_y, val_datasetx_filename, val_datasety_filename)
    # release
    del val_x
    del val_y
    gc.collect()


def save_data_2_np(dataset_x,
                   dataset_y,
                   dataset_x_name,
                   dataset_y_name,
                   shuffle_times=100000):
    print 'save data to numpy'
    print 'dataset_x len = %d' % len(dataset_x)
    print 'dataset_y len = %d' % len(dataset_y)
    _shuffle_data(dataset_x, dataset_y, shuffle_times=shuffle_times)
    np.save(dataset_x_name, dataset_x)

    def one_hot_encoder(label, class_count=2):
        mat = np.asarray(np.zeros(class_count), dtype='float32').reshape(1, class_count)
        for i in range(class_count):
            if i == label:
                mat[0, i] = 1
        return mat.flatten()

    label_mat = []
    for l in dataset_y:
        label_mat.append(one_hot_encoder(l))
    np.save(dataset_y_name, label_mat)


def save_test_data_2_np(test_x_filename,
                        test_wc_filename,
                        test_ids_filename,
                        test_index_file,
                        test_x,
                        world_center_list,
                        seriesuids,
                        array_index_list
                        ):
    np.save(test_x_filename, test_x)
    np.save(test_wc_filename, world_center_list)
    np.save(test_ids_filename, seriesuids)
    np.save(test_index_file, array_index_list)


def _assert_xy_dim_equal():
    for file_index, img_file in enumerate(train_mhd_file_list):
        print 'scanning file %d:%s' % (file_index, img_file)
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (np.int16)
        num_z, height, width = img_array.shape
        origin = np.array(itk_img.GetOrigin())  # x,y,z Origin in world coordinates
        spacing = np.array(itk_img.GetSpacing())
        if spacing[0] != spacing[1]:
            print('spacing = %s' % spacing)

    for file_index, img_file in enumerate(test_mhd_file_list):
        print 'scanning file %d:%s' % (file_index, img_file)
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (np.int16)
        num_z, height, width = img_array.shape
        origin = np.array(itk_img.GetOrigin())  # x,y,z Origin in world coordinates
        spacing = np.array(itk_img.GetSpacing())
        if spacing[0] != spacing[1]:
            print('spacing = %s' % spacing)

    for file_index, img_file in enumerate(val_mhd_file_list):
        print 'scanning file %d:%s' % (file_index, img_file)
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (np.int16)
        num_z, height, width = img_array.shape
        origin = np.array(itk_img.GetOrigin())  # x,y,z Origin in world coordinates
        spacing = np.array(itk_img.GetSpacing())
        if spacing[0] != spacing[1]:
            print('spacing = %s' % spacing)


luna_path = './'
train_luna_subset_path = luna_path + 'train/'
test_luna_subset_path = luna_path + 'test/'
val_luna_subset_path = luna_path + 'val/'

# get *.mhd files
train_mhd_file_list = glob(train_luna_subset_path + '*.mhd')
test_mhd_file_list = glob(test_luna_subset_path + '*.mhd')
val_mhd_file_list = glob(val_luna_subset_path + '*.mhd')

# read csv files
train_df_node = pd.read_csv(luna_path + 'csv/train/annotations.csv')
val_df_node = pd.read_csv(luna_path + 'csv/val/annotations.csv')

# add file attribute to pandas
train_df_node['file'] = train_df_node['seriesuid'].map(lambda file_name: get_filename(train_mhd_file_list, file_name))
val_df_node['file'] = val_df_node['seriesuid'].map(lambda file_name: get_filename(val_mhd_file_list, file_name))


if __name__ == '__main__':

    sw_x = 40
    sw_y = 40
    sw_z = 10
    stride_x = 20
    stride_y = 20
    stride_z = 2
    islung_ratio = 0.5
    isnodule_ratio = 0.1
    sample_rate = 0.01

    # multi-processes
    num_process = 4

    print 'parent process %s' % os.getpid()
    print 'process training data and validation data ......'
    p = Pool()
    for i in range(num_process):
        train_sub_mhd_filelist = train_mhd_file_list[int(len(train_mhd_file_list)/num_process)*i:int(len(train_mhd_file_list)/num_process)*(i+1)]
        train_x_file = 'train_x_%d.npy' % i
        train_y_file = 'train_y_%d.npy' % i

        val_sub_mhd_filelist = val_mhd_file_list[int(len(val_mhd_file_list)/num_process)*i:int(len(val_mhd_file_list)/num_process)*(i+1)]
        val_x_file = 'val_x_%d.npy' % i
        val_y_file = 'val_y_%d.npy' % i

        p.apply_async(_sub_process, args=(train_sub_mhd_filelist, train_x_file, train_y_file,
                                          val_sub_mhd_filelist, val_x_file, val_y_file,
                                          sw_x, sw_y, sw_z, stride_x, stride_y, stride_z,
                                          islung_ratio, isnodule_ratio, sample_rate))

    print 'start %d process to process data ......' % num_process
    p.close()
    p.join()
    print 'all subprocesses done'
    print 'training data and validation data have been processed ......'


    print 'process test data ......'
    # single process
    num_test_data_part = 20
    for i in range(num_test_data_part):
        # split test data to num_test_data_part parts
        test_sub_mhd_filelist = test_mhd_file_list[int(len(test_mhd_file_list) / num_test_data_part) * i:int(
            len(test_mhd_file_list) / num_test_data_part) * (i + 1)]
        test_x_file = 'test_x_%d.npy' % i
        test_wc_file = 'test_wc_%d.npy' % i
        test_ids_file = 'test_ids_%d.npy' % i
        test_index_file = 'test_idx_%d.npy' % i

        test_x, world_center_list, seriesuids, array_index_list = sliding_window_scan_test(test_sub_mhd_filelist,
                                                                         sw_x=sw_x, sw_y=sw_y, sw_z=sw_z,
                                                                         stride_x=stride_x, stride_y=stride_y,
                                                                         stride_z=stride_z,
                                                                         islung_ratio=islung_ratio)
        print 'saving part %d test data' % i
        save_test_data_2_np(test_x_file, test_wc_file, test_ids_file, test_index_file,
                            test_x, world_center_list, seriesuids, array_index_list)
        # release
        del test_x
        del world_center_list
        del seriesuids
        gc.collect()

    print 'test data has been processed ......'
