import h5py
import numpy as np

from tqdm import tqdm
from glob import glob

def load_train_or_val_dataset(data_dir, pattern):
    """
    loading training data or validation data to memory
    :param data_dir:  path = os.path.join(os.path.dirname(os.getcwd()), 'output')
    :param pattern: '/train_dataset.h5' or '/validation_dataset.h5'
    :return: 
    """
    data_files = glob(data_dir+pattern)
    assert len(data_files)==1

    with h5py.File(data_files[0], 'r') as hf:
        dataset_x = np.array(hf.get('data'))
        dataset_y = np.array(hf.get('label'))
    return dataset_x, dataset_y


def load_train_or_val_dataset_single(data_file):
    """
    read single training file or validation file
    :param data_file: 
    :return: 
    """
    with h5py.File(data_file, 'r') as hf:
        dataset_x = np.array(hf.get('data'))
        dataset_y = np.array(hf.get('label'))
        label_mat = []
        for l in dataset_y:
            label_mat.append(_one_hot_encoder(l))

    return dataset_x, label_mat


def load_train_or_val_dataset_full(data_files):
    """
    read all training or validation files and concat them
    :param data_files: 
    :return: 
    """
    dataset_x, dataset_y = None, None
    for i in tqdm(range(len(data_files))):
        tmp_x, tmp_y = load_train_or_val_dataset_single(data_files[i])
        dataset_x = tmp_x if dataset_x is None else np.concatenate((dataset_x, tmp_x), 0)
        dataset_y = tmp_y if dataset_y is None else np.concatenate((dataset_y, tmp_y), 0)

    return dataset_x, dataset_y


def _one_hot_encoder(label, class_count=2):
    mat = np.asarray(np.zeros(class_count), dtype='float32').reshape(1, class_count)
    for i in range(class_count):
        if i == label:
            mat[0, i] = 1
    return mat.flatten()



def load_test_dataset(data_file):
    """
    loading test data
    :param data_dir: path = os.path.join(os.path.dirname(os.getcwd()), 'output')
    :return: 
    """
    with h5py.File(data_file, 'r') as hf:
        dataset_x = np.array(hf.get('data'))
        world_centers = np.array(hf.get('wc'))
        seriesuids = np.array(hf.get('ids'))
    return dataset_x, world_centers, seriesuids
