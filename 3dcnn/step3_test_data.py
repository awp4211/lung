import tensorflow as tf
import os
import pandas as pd

from glob import glob
from tqdm import tqdm

import data_reader as dr

from step2_train_model1 import c3d_net


def _generate_pd_line(seriesuid, wc, prob, output=True):
    line = pd.DataFrame({'seriesuid': seriesuid,
                         'coordX': wc[0],
                         'coordY': wc[1],
                         'coordZ': wc[2],
                         'probability': prob}, index=['0'])
    if output:
        print '%s\t%f\t%f\t%f\t%f' % (seriesuid, wc[0], wc[1], wc[2], prob)
        #print 'seriesuid:%s, coordX:%f, coordY:%f, coordZ:%f, probability:%f' % (seriesuid, wc[0], wc[1], wc[2], prob)
    return line


def test_c3d(edge_size=36, model_save_path='trained_model/3dcnn_f1.ckpt', batch_size=100, start=0):

    x = tf.placeholder(tf.float32, [None, edge_size, edge_size, edge_size])
    keep_prob = tf.placeholder(tf.float32)

    data_files = glob(os.path.join(os.path.dirname(os.getcwd()), 'output')+'/test*.h5')
    print len(data_files)

    y_conv = c3d_net(x, keep_prob)
    y_prob = tf.nn.softmax(y_conv)

    print 'building sliding_window ......'
    saver = tf.train.Saver()
    #init = tf.initialize_all_variables()
    with tf.Session() as sess:
        print 'restore ......'
        saver.restore(sess, model_save_path)

        for i in tqdm(range(start, len(data_files))):
            print '...... init dataframe ......'
            result_pd = pd.DataFrame(columns=('seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'))

            print '...... loading dataset part %s _index = %d' % (data_files[i], i)
            dataset_x, world_centers, seriesuids = dr.load_test_dataset(data_files[i])

            for j in range(dataset_x.shape[0] // batch_size):
                try:
                    prob_y = sess.run([y_prob],
                                  feed_dict={x: dataset_x[j * batch_size:(j + 1) * batch_size], keep_prob: 1.0})
                except Exception:
                    print 'file %s, batch %d' % (data_files[i], j)

                # save data
                for idx, prob in enumerate(prob_y[0]):
                    #print 'index = ', j*batch_size+idx, ' prob = ', prob
                    if prob[1] > prob[0]:
                        result_pd = result_pd.append(_generate_pd_line(seriesuid=seriesuids[j*batch_size+idx],
                                                                   wc=world_centers[j*batch_size+idx],
                                                                   prob=prob[1]), ignore_index=True)

            try:
                # add last datas
                prob_y = sess.run([y_prob],
                              feed_dict={x: dataset_x[(dataset_x.shape[0] // batch_size) * batch_size:],
                                         keep_prob: 1.0})
            except Exception:
                print 'file %s, batch %d' % (data_files[i], (dataset_x.shape[0] // batch_size)+1)

            # save data
            for idx, prob in enumerate(prob_y[0]):
                #print prob
                if prob[1]>prob[0]:
                    result_pd = result_pd.append(_generate_pd_line(seriesuid=seriesuids[(dataset_x.shape[0] // batch_size)+idx],
                                                               wc=world_centers[(dataset_x.shape[0] // batch_size)+idx],
                                                               prob=prob[1]), ignore_index=True)
            #to csv
            path = os.path.join(os.path.dirname(os.getcwd()), 'output2')
            result_pd.to_csv('%s/%d_%s' % (path, i, 'result.csv'),
                             columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'],
                             index=False, header=True)


    print ' test net done'


if __name__ == '__main__':
    test_c3d(edge_size=48, batch_size=30, start=4)
