import SimpleITK as sitk
import numpy as np
from glob import glob
import random
import pandas as pd
import os
import gc
import cv2
import selectivesearch
import h5py
import scipy
import scipy.ndimage


from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from lung_segmentation import segment_lung_mask


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

    nodules = []

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
            # nodule mask: z_min, z_max, y_min, y_max, x_min, x_max
            nodule = [slices[0], slices[len(slices)-1],
                      np.max([0, int(v_center[1] - np.rint(vy_diam / 2))]),
                      np.min([height - 1, int(v_center[1] + np.rint(vy_diam / 2))]),
                      np.max([0, int(v_center[0] - np.rint(vx_diam / 2))]),
                      np.min([width - 1, int(v_center[0] + np.rint(vx_diam / 2))])]
            nodules.append(nodule)
    masks = masks.clip(0, 1)
    return masks, nodules


def _resample(image, old_spacing, new_spacing=[1, 1, 1]):

    resize_factor = old_spacing[::-1] / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def selective_search_train_or_val(mhd_file_list, df_node, output=True, edge_size=48, sample_rate=0.2):
    """
    selective search scan training or validation dateset
    :param mhd_file_list:mhd list
    :param df_node: pandas data frame
    """
    volume_list = []
    labels = []

    for file_index, img_file in enumerate(tqdm(mhd_file_list)):
        print '====================================================================='
        print 'selective search scanning file %d:%s' % (file_index, img_file)
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (np.int16)
        #num_z, height, width = img_array.shape
        origin = np.array(itk_img.GetOrigin())# x,y,z Origin in world coordinates
        spacing = np.array(itk_img.GetSpacing())#x,y,z spacing mm/px

        print 'image world coordinate origin = %s, spacing = %s' % (origin, spacing)

        # reshape image with new spacing
        img_array, new_spacing = _resample(image=img_array, old_spacing=spacing, new_spacing=[1,1,1])
        print 'after resample, image shape = %s, new_spacing = %s' % (img_array.shape, new_spacing)

        segmented_lung_mask = segment_lung_mask(img_array, True)
        mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
        nodules_mask, postive_nodules = get_nodule_mask(image_arr=img_array, mini_df=mini_df,
                                                        spacing=new_spacing, origin=origin)

        mhd_name = img_file[img_file.rfind('/')+1: img_file.find('.')]

        # segmentation of lung
        lung_mask = np.multiply(img_array, segmented_lung_mask)
        lung_mask[lung_mask == 0] = -2048

        # save segment result to disk
        output_dir = os.path.join(os.path.dirname(os.getcwd()), 'output/%s/' % mhd_name)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        for i in range(len(lung_mask)):
            plt.imsave('%s_lung_img_%d.jpg' % (output_dir, i), lung_mask[i], cmap=plt.cm.bone)

        # normalize image array
        img_array = set_window_width(img_array)
        img_array = normalize(img_array)
        img_array = np.asarray(img_array, dtype='float32')

        # rotate postive sample
        for postive_nodule in postive_nodules:
            vz_min = postive_nodule[0]
            vz_max = postive_nodule[1]
            vy_min = postive_nodule[2]
            vy_max = postive_nodule[3]
            vx_min = postive_nodule[4]
            vx_max = postive_nodule[5]
            width = vx_max - vx_min
            height = vy_max - vy_min
            long = vz_max - vz_min

            z_min = np.max([0, int(vz_min + long/2 - edge_size/2)])
            z_max = np.min([img_array.shape[0], int(vz_min + long/2 + edge_size/2)])
            if z_max == img_array.shape[0]: z_min = z_max - edge_size
            if z_min == 0: z_max = z_min + edge_size

            y_min = np.max([0, int(vy_min + height/2 - edge_size/2)])
            y_max = np.min([img_array.shape[1], int(vy_min + height/2 + edge_size/2)])
            if y_max == img_array.shape[1]: y_min == y_max - edge_size
            if y_min == 0: y_max = y_min + edge_size

            x_min = np.max([0, int(vx_min + width/2 - edge_size/2)])
            x_max = np.min([img_array.shape[1], int(vx_min + width/2 + edge_size/2)])
            if x_max == img_array.shape[2]: x_min == x_max - edge_size
            if x_min == 0: x_max = x_min + edge_size

            roi_3d = img_array[z_min:z_max, y_min:y_max, x_min:x_max]

            print '-----(%d, %d), (%d, %d), (%d, %d)' % (z_min, z_max, y_min, y_max, x_min, x_max)

            for degree in range(0, 360, 10):
                # split data by z index
                new_roi3d = np.zeros(roi_3d.shape, dtype=np.uint8)
                for z_index, roi_2d in enumerate(roi_3d):
                    rows, cols = roi_2d.shape
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
                    dst = cv2.warpAffine(roi_2d, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=1.)
                    new_roi3d[z_index] = dst
                volume_list.append(new_roi3d)
                labels.append(1)

        print 'selective search images ......'

        lung_image_paths = glob(output_dir+'_lung_img_*.jpg')
        for z_index, lung_image_path in enumerate(lung_image_paths):
            lung_image = cv2.imread(lung_image_path, cv2.IMREAD_COLOR)
            img_lbl, regions = selectivesearch.selective_search(lung_image, scale=600, sigma=0.8, min_size=10)
            counter = 0

            for region in regions:
                min_x = region['rect'][0]
                min_y = region['rect'][1]
                width = region['rect'][2]
                height = region['rect'][3]

                if width<30 and height<30 and width >5 and height>5:
                    # we only select roi whose size between 5mm and 30mm in real world
                    roi_area = width * height
                    internel_area = region['size']
                    internel_ratio = (internel_area + 0.0) / roi_area
                    if internel_ratio > 0.1 and internel_ratio < 0.8:
                        if nodules_mask[z_index, (min_y+height/2), (min_x+width/2)] == 1:
                            # positive sample
                            pass
                        else:
                            # negative sample
                            counter += 1
                            cv2.rectangle(lung_image, (min_x, min_y), (min_x + width, min_y + height), (255, 0, 0))
                            # 2D ROI TO 3D
                            # roi_world_center = np.array([(min_x + width)/2, (min_y + height)/2, z_index])*spacing + origin
                            vx_min = np.max([0, (min_x + width/2) - edge_size/2])
                            vx_max = np.min([img_array.shape[2], vx_min + edge_size])
                            vy_min = np.max([0, (min_y + height/2) - edge_size/2])
                            vy_max = np.min([img_array.shape[1], vy_min + edge_size])
                            vz_min = np.max([0, z_index - edge_size/2])
                            vz_max = np.min([img_array.shape[0], vz_min + edge_size])
                            if (vx_max - vx_min) == edge_size and (vy_max - vy_min) == edge_size and (vz_max-vz_min) == edge_size:
                                if np.random.random_sample() < sample_rate:
                                    roi_3d = img_array[int(vz_min):int(vz_max),
                                                       int(vy_min):int(vy_max),
                                                       int(vx_min):int(vx_max)]
                                    volume_list.append(roi_3d)
                                    labels.append(0)

            if output:
                plt.imsave('%s_ss_%s.jpg' % (output_dir, lung_image_path[lung_image_path.find('_') + 1:lung_image_path.rfind('.')]), lung_image, cmap='gray')
                #print('lung_image %s, region count = %d, bounding box count = %d' % (lung_image_path, len(regions), counter))

        print 'volume_list len = %d' % len(volume_list)
        print 'positive samples count = %d' % np.sum(np.asarray(labels) == 1)
        print 'negative samples count = %d' % np.sum(np.asarray(labels) == 0)

    return volume_list, labels


def selective_search_test(mhd_file_list, output=True, edge_size=48):

    volume_list = []
    world_center_list = []
    seriesuids = []

    for file_index, img_file in enumerate(tqdm(mhd_file_list)):
        print '====================================================================='
        print 'selective search scanning file %d:%s' % (file_index, img_file)
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (np.int16)
        # num_z, height, width = img_array.shape
        origin = np.array(itk_img.GetOrigin())  # x,y,z Origin in world coordinates
        spacing = np.array(itk_img.GetSpacing())  # x,y,z spacing mm/px

        print 'image world coordinate origin = %s, spacing = %s' % (origin, spacing)

        img_array, new_spacing = _resample(image=img_array, old_spacing=spacing, new_spacing=[1, 1, 1])
        segmented_lung_mask = segment_lung_mask(img_array, True)
        mhd_name = img_file[img_file.rfind('/') + 1: img_file.find('.')]

        # segmentation of lung
        lung_mask = np.multiply(img_array, segmented_lung_mask)
        lung_mask[lung_mask == 0] = -2048

        # save segment result to disk
        output_dir = os.path.join(os.path.dirname(os.getcwd()), 'output/%s/' % mhd_name)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        for i in range(len(lung_mask)):
            plt.imsave('%s_lung_img_%d.jpg' % (output_dir, i), lung_mask[i], cmap=plt.cm.bone)

        # normalize image array
        img_array = set_window_width(img_array)
        img_array = normalize(img_array)
        img_array = np.asarray(img_array, dtype='float32')

        lung_image_paths = glob(output_dir + '_lung_img_*.jpg')
        for z_index, lung_image_path in enumerate(lung_image_paths):
            lung_image = cv2.imread(lung_image_path, cv2.IMREAD_COLOR)
            img_lbl, regions = selectivesearch.selective_search(lung_image, scale=600, sigma=0.8, min_size=10)
            counter = 0

            for region in regions:
                min_x = region['rect'][0]
                min_y = region['rect'][1]
                width = region['rect'][2]
                height = region['rect'][3]
                max_x = min_x + width
                max_y = min_y + height

                if width<30 and height<30 and width>5 and height>5:
                    # we only select roi whose size between 5mm and 30mm in real world
                    roi_area = width * height
                    internel_area = region['size']
                    internel_ratio = (internel_area + 0.0) / roi_area
                    if internel_ratio > 0.1 and internel_ratio < 0.8:
                        counter += 1
                        cv2.rectangle(lung_image, (min_x, min_y), (min_x + width, min_y + height), (255, 0, 0))
                        # 2D ROI TO 3D
                        roi_world_center = np.array([(min_x + width/2), (min_y + height/2), z_index])*new_spacing + origin
                        vx_min = np.max([0, (min_x + width/2) - edge_size / 2])
                        vx_max = np.min([img_array.shape[2], vx_min + edge_size])
                        vy_min = np.max([0, (min_y + height/2) - edge_size / 2])
                        vy_max = np.min([img_array.shape[1], vy_min + edge_size])
                        vz_min = np.max([0, z_index - edge_size / 2])
                        vz_max = np.min([img_array.shape[0], vz_min + edge_size])
                        if (vx_max - vx_min) == edge_size and (vy_max - vy_min) == edge_size and (
                            vz_max - vz_min) == edge_size:
                            roi_3d = img_array[int(vz_min):int(vz_max),
                                               int(vy_min):int(vy_max),
                                               int(vx_min):int(vx_max)]
                            volume_list.append(roi_3d)
                            world_center_list.append(roi_world_center)
                            seriesuids.append(img_file[img_file.rfind('/')+1:img_file.rfind('.')])

            if output:
                plt.imsave('%s_ss_%s.jpg' % (output_dir, lung_image_path[lung_image_path.find('_') + 1:lung_image_path.rfind('.')]), lung_image, cmap='gray')
                #print('lung_image %s, region count = %d, bounding box count = %d' % (lung_image_path, len(regions), counter))

        print 'volume_list len = %d' % len(volume_list)

    return volume_list, world_center_list, seriesuids


def _sub_process_train_val(sub_mhd_file,
                 df_node,
                 dataset_filename,
                 edge_size,
                 sample_rate):
    dataset_x, dataset_y = selective_search_train_or_val(sub_mhd_file, df_node, output=True, edge_size=edge_size, sample_rate=sample_rate)
    _save_data(dataset_x=dataset_x, dataset_y=dataset_y, dataset_name=dataset_filename)
    # release
    del dataset_x
    del dataset_y
    gc.collect()


def _sub_process_test(sub_mhd_file,
                      dataset_filename,
                      edge_size):
    test_x, world_center_list, seriesuids = selective_search_test(sub_mhd_file, output=True, edge_size=edge_size)

    _save_test_data(test_x, world_center_list, seriesuids, dataset_filename)

    # release
    del test_x
    del world_center_list
    del seriesuids
    gc.collect()



def _save_data(dataset_x,
              dataset_y,
              dataset_name,
              shuffle_times=100000):
    print 'save data to numpy'
    print 'dataset_x len = %d' % len(dataset_x)
    print 'dataset_y len = %d' % len(dataset_y)
    _shuffle_data(dataset_x, dataset_y, shuffle_times=shuffle_times)

    def one_hot_encoder(label, class_count=2):
        mat = np.asarray(np.zeros(class_count), dtype='float32').reshape(1, class_count)
        for i in range(class_count):
            if i == label:
                mat[0, i] = 1
        return mat.flatten()

    label_mat = []
    for l in dataset_y:
        label_mat.append(one_hot_encoder(l))
    with h5py.File(dataset_name, 'w') as hf:
        hf.create_dataset('data', data=dataset_x)
        hf.create_dataset('label', data=dataset_y)


def _save_test_data(test_x,
                   test_wc,
                   test_ids,
                   dataset_name):
    with h5py.File(dataset_name, 'w') as hf:
        hf.create_dataset('data', data=test_x)
        hf.create_dataset('wc', data=test_wc)
        hf.create_dataset('ids', data=test_ids)


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


def _statistic_data():
    y_postive_number = 0
    y_negative_number = 0
    from data_reader import load_train_or_val_dataset_single
    path = os.path.join(os.path.dirname(os.getcwd()), 'output')
    train_data_files = glob(path + '/val_*.h5')
    for i in tqdm(range(len(train_data_files))):
        print 'statistic file : %s' % train_data_files[i]
        train_set_x, train_set_y = load_train_or_val_dataset_single(train_data_files[i])
        y_postive_number += np.sum(np.array(train_set_y)[:, 1] == 1)
        y_negative_number += np.sum(np.array(train_set_y)[:, 1] == 0)
    print 'postive data : %d, negative data : %d' % (y_postive_number, y_negative_number)


def process_data(num_process, data_part, edge_size, sample_rate):
    # constants
    luna_path = os.path.dirname(os.getcwd())
    train_luna_subset_path = luna_path + '/train/'
    test_luna_subset_path = luna_path + '/test2/'
    val_luna_subset_path = luna_path + '/val/'

    # get *.mhd files
    train_mhd_file_list = glob(train_luna_subset_path + '*.mhd')
    test_mhd_file_list = glob(test_luna_subset_path + '*.mhd')
    val_mhd_file_list = glob(val_luna_subset_path + '*.mhd')

    # read csv files
    train_df_node = pd.read_csv(luna_path + '/csv/train/annotations.csv')
    val_df_node = pd.read_csv(luna_path + '/csv/val/annotations.csv')

    # add file attribute to pandas
    train_df_node['file'] = train_df_node['seriesuid'].map(
        lambda file_name: get_filename(train_mhd_file_list, file_name))
    val_df_node['file'] = val_df_node['seriesuid'].map(lambda file_name: get_filename(val_mhd_file_list, file_name))

    print 'parent process %s' % os.getpid()
    print '...... process training data ......'

    for j in range(data_part):

        _train_sub_mhd_filelist = train_mhd_file_list[int(len(train_mhd_file_list) / data_part) * j:
                                                      int(len(train_mhd_file_list) / data_part) * (j + 1)]

        p = Pool()
        for i in range(num_process):
            __train_sub_mhd_filelist = _train_sub_mhd_filelist[int(len(_train_sub_mhd_filelist) / num_process) * i:
                                                               int(len(_train_sub_mhd_filelist) / num_process) * (i + 1)]
            train_filename = '%s/output/train_%d_%d.h5' % (luna_path, j, i)
            p.apply_async(_sub_process_train_val, args=(__train_sub_mhd_filelist, train_df_node, train_filename, edge_size, sample_rate))
        p.close()
        p.join()
        del p
        gc.collect()

    print '...... process validation data ......'

    for j in range(data_part):

        _val_sub_mhd_filelist = val_mhd_file_list[int(len(val_mhd_file_list) / data_part) * j:
                                                  int(len(val_mhd_file_list) / data_part) * (j + 1)]

        p = Pool()
        for i in range(num_process):
            __val_sub_mhd_filelist = _val_sub_mhd_filelist[int(len(_val_sub_mhd_filelist) / num_process) * i:
                                                           int(len(_val_sub_mhd_filelist) / num_process) * (i + 1)]
            val_filename = '%s/output/val_%d_%d.h5' % (luna_path, j, i)
            p.apply_async(_sub_process_train_val, args=(__val_sub_mhd_filelist, val_df_node, val_filename, edge_size, sample_rate))
        p.close()
        p.join()
        del p
        gc.collect()

    print 'training data and validation data have been processed ......'

    print 'process test data ......'

    for j in range(data_part):

        _test_sub_mhd_filelist = test_mhd_file_list[int(len(test_mhd_file_list) / data_part) * j:
                                                    int(len(test_mhd_file_list) / data_part) * (j + 1)]

        p = Pool()
        for i in range(num_process):
            __test_sub_mhd_filelist = _test_sub_mhd_filelist[int(len(_val_sub_mhd_filelist) / num_process) * i:
                                                           int(len(_val_sub_mhd_filelist) / num_process) * (i + 1)]
            test_filename = '%s/output/test_%d_%d.h5' % (luna_path, j, i)
            p.apply_async(_sub_process_test, args=(__test_sub_mhd_filelist, test_filename, edge_size))
        p.close()
        p.join()
        del p
        gc.collect()

    print 'test data has been processed ......'


if __name__ == '__main__':
    #process_data(num_process=4, data_part=10, edge_size=48, sample_rate=0.05)
    _statistic_data()
