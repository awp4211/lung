import data_reader as dr
import pandas as pd
import numpy as np
from utils import data_reader as dr
from pandas import Series, DataFrame
from tqdm import tqdm
from multiprocessing import Pool

def _vote_sub(root_dir, num_data_part, start_index, end_index, csv_filename):
    test_ids, test_wcs, test_idxs, test_ys = dr.read_tested_value(root_dir=root_dir, num_data_part=num_data_part)
    data = {'seriesuid': test_ids,
            'coordX': test_wcs[:, 0],
            'coordY': test_wcs[:, 1],
            'coordZ': test_wcs[:, 2],
            'indexX': test_idxs[:, 0],
            'indexY': test_idxs[:, 1],
            'indexZ': test_idxs[:, 2],
            'y': [np.argmax(y) for y in test_ys],
            'probability': np.zeros((len(test_ids)), dtype='float32')}
    df = DataFrame(data)

    stride_x = 10
    stride_y = 10
    stride_z = 1

    num_scan_x = range(-2, 3)
    num_scan_y = range(-2, 3)
    num_scan_z = range(-5, 6)

    positive_df = df[df['y'] == 1]

    #for i in tqdm(range(positive_df.shape[0])):
    for i in tqdm(range(start_index, end_index)):
        # find row
        row = positive_df.iloc[i]
        adj_count = 0.
        voted = 0.
        current_x_index = row.indexX
        current_y_index = row.indexY
        current_z_index = row.indexZ
        # find all seriesuid's data
        mini_df = df[df['seriesuid'] == row.seriesuid]
        for scan_x in num_scan_x:
            for scan_y in num_scan_y:
                for scan_z in num_scan_z:
                    selected_row_df = mini_df[(mini_df['indexX'] == scan_x * stride_x + current_x_index)
                                              & (mini_df['indexY'] == scan_y * stride_y + current_y_index)
                                              & (mini_df['indexZ'] == scan_z * stride_z + current_z_index)]
                    if selected_row_df.shape[0] > 0:
                        adj_count = adj_count + 1
                        voted = voted + selected_row_df.y.values[0]
        # end of for
        positive_df['probability'][i] = voted / adj_count

    computed_df = positive_df[positive_df['probability'] > 0]
    computed_df.to_csv('%s%s' % (root_dir, csv_filename),
                        columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'],
                        index=False, header=True)


def vote(root_dir, num_data_part, num_process=8):
    test_ids, test_wcs, test_idxs, test_ys = dr.read_tested_value(root_dir=root_dir, num_data_part=num_data_part)
    data = {'seriesuid': test_ids,
            'coordX': test_wcs[:, 0],
            'coordY': test_wcs[:, 1],
            'coordZ': test_wcs[:, 2],
            'indexX': test_idxs[:, 0],
            'indexY': test_idxs[:, 1],
            'indexZ': test_idxs[:, 2],
            'y': [np.argmax(y) for y in test_ys],
            'probability': np.zeros((len(test_ids)), dtype='float32')}
    df = DataFrame(data)

    positive_df = df[df['y'] == 1]

    p = Pool()
    for i in range(num_process):
        start = i * (positive_df.shape[0]//num_process)
        end = (i+1) * (positive_df.shape[0]//num_process)
        if i == num_process-1:
            end = positive_df.shape[0]

        csv_filename = 'result_%i.csv' % i
        p.apply_async(_vote_sub, args=(root_dir, num_data_part, start, end, csv_filename))

    p.close()
    p.join()


if __name__ == '__main__':
    root_dir = '/home/xiyou/Desktop/lung/npy/'
    vote(root_dir=root_dir, num_data_part=20, num_process=4)