import numpy as np
from glob import glob

def read_train_and_val_data(num_part, rootdir='npy/'):
    """
    read training dataset and val dataset from folder rootdir
    :param num_part: number of data parts
    :param rootdir:  root data folder
    :return: 
    """
    train_x_files =[]
    train_y_files =[]
    val_x_files = []
    val_y_files = []
    for i in range(num_part):
        train_x_file = '%strain_x_%d.npy' % (rootdir, i)
        train_y_file = '%strain_y_%d.npy' % (rootdir, i)
        val_x_file = '%sval_x_%d.npy' % (rootdir, i)
        val_y_file = '%sval_y_%d.npy' % (rootdir, i)
        train_x_files.append(train_x_file)
        train_y_files.append(train_y_file)
        val_x_files.append(val_x_file)
        val_y_files.append(val_y_file)

    train_x = np.concatenate([np.load(f) for f in train_x_files]) #(428328,2000)
    train_y = np.concatenate([np.load(f) for f in train_y_files]) #(428328,2)
    val_x = np.concatenate([np.load(f) for f in val_x_files])     #(149904,2000)
    val_y = np.concatenate([np.load(f) for f in val_y_files])     #(149904,2)
    return train_x, train_y, val_x, val_y


def _merge_test_y_part(root_dir, num_data_part):
    """
    merge all tested ys
    :param root_dir: the root directory of npy files
    :param num_data_part: the number of splited data size
    :return: 
    """
    test_ys_files = []
    test_batchs = 150
    for i in range(num_data_part):
        for j in range(test_batchs):
            ys_file = '%stest_y_%d_%d.npy' % (root_dir, i, j)
            print '__retrive %s' % ys_file
            try:
                ys = np.load(ys_file)
                test_ys_files.append(ys_file)
            except Exception,e:
                break
    return test_ys_files


def read_tested_value(root_dir='npy/', num_data_part=20):
    """
    read tested datas
    :param root_dir: the npy root directory
    :param num_data_part: the number of splited data size
    :return: tested data id, world center list, array index center list, tested output on each dimension([0.xxx,0.xxx])
    """
    test_ids_files = []
    test_wcs_files =[]
    test_index_files = []

    test_ys_files = _merge_test_y_part(root_dir, num_data_part)

    for i in range(num_data_part):
        test_ids_file = '%stest_ids_%d.npy' % (root_dir, i)
        test_wcs_file = '%stest_wc_%d.npy' % (root_dir, i)
        test_index_file = '%stest_idx_%d.npy' % (root_dir, i)
        test_ids_files.append(test_ids_file)
        test_wcs_files.append(test_wcs_file)
        test_index_files.append(test_index_file)
        print 'retrive %s,%s,%s' % (test_ids_file, test_wcs_file, test_index_file)

    test_ids = np.concatenate([np.load(f) for f in test_ids_files])
    test_wcs = np.concatenate([np.load(f) for f in test_wcs_files])
    test_idxs = np.concatenate([np.load(f) for f in test_index_files])
    test_ys = np.concatenate([np.load(f)[0] for f in test_ys_files])

    print 'test_ids len = %d' % len(test_ids)
    print 'test_wcs len = %d' % len(test_wcs)
    print 'test_idx len = %d' % len(test_idxs)
    print 'test_ys len = %d' % len(test_ys)

    return test_ids, test_wcs, test_idxs, test_ys


if __name__ == '__main__':
    read_tested_value()