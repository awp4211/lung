import tensorflow as tf
import ploter as plt


def _conv3d(x, n_filters, kernels, strides, stddev=0.1,
           activation=None, bias=True, padding='SAME',name='conv3D'):
    """
    convolution 3D
    :param x:
    :param n_filters:
    :param kernels:
    :param strides:
    :param stddev:
    :param activation:
    :param bias:
    :param padding:
    :param name:
    :return:
    """
    assert len(kernels) == 3
    assert len(strides) == 3

    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [kernels[0], kernels[1], kernels[2], x.get_shape()[-1],n_filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(x, w,
                            strides=[1, strides[0], strides[1], strides[2], 1],
                            padding=padding)
        if bias:
            b = tf.get_variable('b', [n_filters],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.bias_add(conv, b)
        if activation:
            conv = activation(conv)
        return conv


def _spatial_reduction_block(net, name):
    """
    spatial reduction block used to reduct feature map size to half, meanwhile it will
    increase feature map size to twice.
    :param net:
    :param name:
    :return:
    """
    with tf.name_scope(name):
        with tf.name_scope(name+'/Maxpool3d_2_2'):
            branch_0 = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1],
                                        strides=[1, 2, 2, 2, 1],
                                        padding='SAME')
        with tf.name_scope(name+'/Conv3d_a'):
            branch_1 = _conv3d(net, n_filters=net.get_shape().as_list()[-1]/4, kernels=[3, 3, 3],
                              strides=[2, 2, 2], activation=tf.nn.relu, name=name+'/b1_conv3_3', padding='SAME')
        with tf.name_scope(name+'/Conv3d_b'):
            branch_2 = _conv3d(net, n_filters=net.get_shape().as_list()[-1], kernels=[1, 1, 1],
                              strides=[1, 1, 1], activation=tf.nn.relu, name=name+'/b2_conv1_1', padding='SAME')
            branch_2 = _conv3d(branch_2, n_filters=net.get_shape().as_list()[-1] * 5/16, kernels=[3, 3, 3],
                              strides=[2, 2, 2], activation=tf.nn.relu, name=name+'/b2_conv3_3', padding='SAME')
        with tf.name_scope(name+'/Conv3d_c'):
            branch_3 = _conv3d(net, n_filters=net.get_shape().as_list()[-1], kernels=[1, 1, 1],
                              strides=[1, 1, 1], activation=tf.nn.relu, name=name+'/b3_conv1_1', padding='SAME')
            branch_3 = _conv3d(branch_3, n_filters=net.get_shape().as_list()[-1] * 5/16, kernels=[3, 3, 3],
                              strides=[1, 1, 1], activation=tf.nn.relu, name=name+'/b3_conv3_3', padding='SAME')
            branch_3 = _conv3d(branch_3, n_filters=net.get_shape().as_list()[-1] * 7/16, kernels=[3, 3, 3],
                              strides=[2, 2, 2], activation=tf.nn.relu, name=name+'/b3_conv3_3_', padding='SAME')
        print '%s/branch_0, shape = %s' % (name, branch_0.get_shape())
        print '%s/branch_1, shape = %s' % (name, branch_1.get_shape())
        print '%s/branch_2, shape = %s' % (name, branch_2.get_shape())
        print '%s/branch_3, shape = %s' % (name, branch_3.get_shape())

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=4)
        return net


def _residual_conv_block(net, name):
    """
    residual block, the number of feature map's size is unchange
    :param net:
    :param name:
    :return:
    """
    with tf.name_scope(name):
        with tf.name_scope(name+'/Conv3d_a'):
            branch_0 = _conv3d(net, n_filters=net.get_shape().as_list()[-1]/2, kernels=[3, 3, 3],
                              strides=[1, 1, 1], activation=tf.nn.relu, name=name+'/b0_conv3_3', padding='SAME')
        with tf.name_scope(name+'/Conv3d_b'):
            branch_1 = _conv3d(net, n_filters=net.get_shape().as_list()[-1]/2, kernels=[1, 1, 1],
                              strides=[1, 1, 1], activation=tf.nn.relu, name=name+'/b1_conv1_1', padding='SAME')
            branch_1 = _conv3d(branch_1, n_filters=net.get_shape().as_list()[-1]/2, kernels=[3, 3, 3],
                              strides=[1, 1, 1], activation=tf.nn.relu, name=name+'/b1_conv3_3', padding='SAME')
        with tf.name_scope(name+'/Conv3d_c'):
            branch_2 = _conv3d(net, n_filters=net.get_shape().as_list()[-1]/2, kernels=[1, 1, 1],
                              strides=[1, 1, 1], activation=tf.nn.relu, name=name+'/b2_conv1_1', padding='SAME')
            branch_2 = _conv3d(branch_2, n_filters=net.get_shape().as_list()[-1]/2, kernels=[3, 3, 3],
                              strides=[1, 1, 1], activation=tf.nn.relu, name=name+'/b2_conv3_3', padding='SAME')
            branch_2 = _conv3d(branch_2, n_filters=net.get_shape().as_list()[-1]/2, kernels=[3, 3, 3],
                              strides=[1, 1, 1], activation=tf.nn.relu, name=name+'/b2_conv3_3_', padding='SAME')
        print '%s/branch_0, shape = %s' % (name, branch_0.get_shape())
        print '%s/branch_1, shape = %s' % (name, branch_1.get_shape())
        print '%s/branch_2, shape = %s' % (name, branch_2.get_shape())

        with tf.name_scope(name+'/Merge'):
            concated = tf.concat(values=[branch_0, branch_1, branch_2], axis=4)
        with tf.name_scope(name+'/Conv3d_d'):
            concated = _conv3d(concated, n_filters=net.get_shape().as_list()[-1], kernels=[1, 1, 1],
                         strides=[1, 1, 1], activation=tf.nn.relu, name=name+'/concate_conv1_1', padding='SAME')
        with tf.name_scope(name+'/Residual_merge'):
            net = net + concated
            net = tf.nn.relu(net)
        return net


def residual_inception_c3d_net(x, dropout_prob, n_hidden_unit=2000, n_classes=2):

    print 'volume shape = %s' % x.get_shape()

    with tf.name_scope('reshape'):
        x = tf.reshape(x, [-1,
                           x.get_shape().as_list()[1],
                           x.get_shape().as_list()[2],
                           x.get_shape().as_list()[3],
                           1])
    print 'after reshape, shape = %s' % x.get_shape()

    with tf.name_scope('conv1'):
        x = _conv3d(x, n_filters=64, kernels=[1, 1, 1], strides=[1, 1, 1],
                   activation=tf.nn.relu, name='conv1/conv3d', padding='SAME')
    print 'conv1, shape = %s' % x.get_shape()

    x = _spatial_reduction_block(x, 'spatial_reduction_1') # 18*18*18*128
    print 'spatial_reduction_1, shape = %s' % x.get_shape()

    x = _residual_conv_block(x, 'res_conv_block_1')  # 18*18*18*128
    print 'res_conv_block_1, shape = %s' % x.get_shape()

    x = _spatial_reduction_block(x, 'spatial_reduction_2')  # 9*9*9*256
    print 'spatial_reduction_2, shape = %s' % x.get_shape()

    x = _residual_conv_block(x, 'res_conv_block_2')  # 9*9*9*256
    print 'res_conv_block_2, shape = %s' % x.get_shape()

    with tf.name_scope('conv2'):
        x = _conv3d(x, n_filters=x.get_shape().as_list()[-1] * 2, kernels=[1, 1, 1], strides=[1, 1, 1],
                   activation=tf.nn.relu, name='conv2/conv3d', padding='SAME')  # 9*9*9*128
    print 'conv2, shape = %s' % x.get_shape()

    with tf.name_scope('maxpool1'):
        x = tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')  # 4*4*4*128
    print 'maxpool1, shape = %s' % x.get_shape()

    with tf.name_scope('conv3'):
        x = _conv3d(x, n_filters=x.get_shape().as_list()[-1] * 2, kernels=[1, 1, 1], strides=[1, 1, 1],
                   activation=tf.nn.relu, name='conv3/conv3d', padding='SAME')  # 9*9*9*128
    print 'conv3, shape = %s' % x.get_shape()

    with tf.name_scope('maxpool2'):
        x = tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')  # 4*4*4*128
    print 'maxpool2, shape = %s' % x.get_shape()

    with tf.name_scope('flatten'):
        x = tf.reshape(x, [-1, x.get_shape().as_list()[1]*x.get_shape().as_list()[2]
                           *x.get_shape().as_list()[3]*x.get_shape().as_list()[4]])
    print 'flatten, shape = %s' % x.get_shape()

    with tf.name_scope('dropout1'):
        x = tf.nn.dropout(x, keep_prob=dropout_prob)

    with tf.name_scope('fc1'):
        w1 = tf.Variable(tf.random_normal([x.get_shape().as_list()[-1], n_hidden_unit]))
        b1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_unit,]))
        x = tf.matmul(x, w1) + b1
        x = tf.nn.sigmoid(x)
    print 'fc1, shape = %s' % x.get_shape()

    with tf.name_scope('dropout2'):
        x = tf.nn.dropout(x, keep_prob=dropout_prob)

    with tf.name_scope('fc2'):
        w2 = tf.Variable(tf.random_normal([n_hidden_unit, n_classes]))
        b2 = tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
        x = tf.matmul(x, w2) + b2
        x = tf.nn.sigmoid(x)
    print 'fc2, shape = %s' % x.get_shape()

    return x


def _test_net_structure(edge_size=36, n_classes=2):
    x = tf.placeholder(tf.float32, [None, edge_size, edge_size, edge_size])
    y = tf.placeholder(tf.float32, [None, n_classes])
    y_conv = residual_inception_c3d_net(x, dropout_prob=0.8)
    y_prob = tf.nn.softmax(y_conv)


def train_c3d(dropout_keep_prob=0.8,
              edge_size=36,
              learning_rate=0.0001,
              n_epochs=10,
              batch_size=100,
              n_classes=2,
              show_batch_result=True):
    """
    
    :param dropout_keep_prob: 
    :param learning_rate: 
    :param n_epochs: 
    :param batch_size: 
    :param n_classes: 
    :return: 
    """
    print '...... loading dataset ......'

    from data_reader import load_train_or_val_dataset_single
    from metrics import metric, metric_
    from glob import glob
    from tqdm import tqdm

    import os
    import numpy as np

    path = os.path.join(os.path.dirname(os.getcwd()), 'output')

    train_data_files = glob(path+'/train_*.h5')
    val_data_files = glob(path+'/val_*.h5')

    print '...... building model ......'
    x = tf.placeholder(tf.float32, [None, edge_size, edge_size, edge_size])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)

    y_conv = residual_inception_c3d_net(x, dropout_prob=keep_prob)
    y_prob = tf.nn.softmax(y_conv)

    cost = tf.reduce_mean(tf.square(y - y_prob))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(y_prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    best_train_f1 = 0.
    best_train_epoch = 0

    best_val_f1 = 0.
    best_val_cost = 1000.
    best_val_epoch = 0

    train_accs = []
    val_accs = []
    train_costs = []
    val_costs = []
    train_f1s = []
    val_f1s = []

    print '...... initializing ......'
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print '...... initializing variables .....'
        sess.run(init)

        print '...... start to train sliding_window ......'
        for epoch_i in tqdm(range(n_epochs)):
            train_cost = 0.
            train_acc = 0.
            train_f1 = 0.
            n_train_examples_total = 0.
            y_probs_train = []
            y_true_train = []

            y_probs_val = []
            y_true_val = []

            # train files
            for i in range(len(train_data_files)):
                print 'training epoch %d, datafile %s : %d' % (epoch_i, train_data_files[i], i)
                train_set_x, train_set_y = load_train_or_val_dataset_single(train_data_files[i])
                n_train_examples = train_set_x.shape[0]
                n_train_examples_total += n_train_examples

                # train sliding_window
                for mini_batch in range(n_train_examples//batch_size):
                    #print 'training epoch %d, minibatch %d' % (epoch_i, mini_batch)
                    batch_xs = train_set_x[mini_batch*batch_size:(mini_batch+1)*batch_size]
                    batch_ys = train_set_y[mini_batch*batch_size:(mini_batch+1)*batch_size]
                    _, t_loss, t_acc, y_p = sess.run([optimizer, cost, accuracy, y_prob],
                                          feed_dict={
                                              x:batch_xs,
                                              y:batch_ys,
                                              keep_prob:dropout_keep_prob
                                          })
                    #print y_p
                    y_probs_train.append(y_p)
                    y_true_train.append(batch_ys)
                    train_cost += t_loss
                    train_acc += t_acc
                    if show_batch_result:
                        print 'epoch %d, data_file %s - %d, minibatch %d, train acc = %f %%, train_cost = %f ,train_f1 = %f' % (epoch_i,
                                train_data_files[i], i, mini_batch, t_acc*100, t_loss, metric_(y_p, batch_ys))

                if n_train_examples % batch_size!=0:
                    batch_xs = train_set_x[(train_set_x.shape[0] // batch_size) * batch_size:]
                    batch_ys = train_set_y[(train_set_x.shape[0] // batch_size) * batch_size:]
                    _, t_loss, t_acc, y_p = sess.run([optimizer, cost, accuracy, y_prob],
                                                     feed_dict={
                                                         x: batch_xs,
                                                         y: batch_ys,
                                                         keep_prob: dropout_keep_prob
                                                     })
                    #print y_p
                    y_probs_train.append(y_p)
                    y_true_train.append(batch_ys)
                    train_cost += t_loss
                    train_acc += t_acc
                    if show_batch_result:
                        print 'epoch %d, data_file %s - %d, minibatch %d, train acc = %f %%, train_cost = %f, train_f1 = %f' % (
                        epoch_i,
                        train_data_files[i], i, (train_set_x.shape[0] // batch_size)+1, t_acc * 100, t_loss, metric_(y_p, batch_ys))


            train_acc /= (n_train_examples_total // batch_size)
            train_cost /= (n_train_examples_total // batch_size)
            train_f1 = metric(y_probs_train, y_true_train)

            # validation files
            validation_cost = 0.
            validation_acc = 0.
            validation_f1 = 0.
            n_val_examples_total = 0.

            for i in range(len(val_data_files)):
                val_set_x, val_set_y = load_train_or_val_dataset_single(val_data_files[i])
                n_val_examples = val_set_x.shape[0]
                n_val_examples_total += n_val_examples

                # validate model on validation dataset
                for mini_batch in range(n_val_examples // batch_size):
                    batch_xs = val_set_x[batch_size * mini_batch:batch_size * (mini_batch + 1)]
                    batch_ys = val_set_y[batch_size * mini_batch:batch_size * (mini_batch + 1)]
                    v_loss, v_acc, v_prob = sess.run([cost, accuracy, y_prob],
                                             feed_dict={
                                                 x: batch_xs,
                                                 y: batch_ys,
                                                 keep_prob: 1.0
                                             })
                    y_probs_val.append(v_prob)
                    y_true_val.append(batch_ys)
                    validation_cost += v_loss
                    validation_acc += v_acc

            validation_acc /= (n_val_examples_total // batch_size)
            validation_cost /= (n_val_examples_total // batch_size)
            validation_f1 = metric(y_probs_val, y_true_val)

            if train_f1 > best_train_f1:
                best_train_f1 = train_f1
                best_train_epoch = epoch_i

            if validation_f1 > best_val_f1:
                best_val_f1 = validation_f1
                best_val_epoch = epoch_i
                saver.save(sess, 'trained_model/3dcnn_m2_f1_lr_%s_dp_%s.ckpt' % (learning_rate, dropout_keep_prob))

                if validation_cost < best_val_cost:
                    best_val_cost = validation_cost
                    saver.save(sess, 'trained_model/3dcnn_m2_cost_lr_%s_dp_%s.ckpt' % (learning_rate, dropout_keep_prob))

            train_accs.append(train_acc)
            train_costs.append(train_cost)
            train_f1s.append(train_f1)
            val_accs.append(validation_acc)
            val_costs.append(validation_cost)
            val_f1s.append(validation_f1)

            print '### epoch_%d,training data f1 = %f, cost = %f, acc = %f %%' % (epoch_i, train_f1, train_cost, train_acc * 100)
            print '### epoch_%d,validation data f1 = %f, cost = %f,acc = %f %%' % (epoch_i, validation_f1, validation_cost, validation_acc * 100)

        print 'done'
        print 'best f1 on training data = %f @ epoch %d' % (best_train_f1, best_train_epoch)
        print 'best f1 on validation data = %f @ epoch %d' % (best_val_f1, best_val_epoch)

        plt.plot_cost(train_accs, n_epochs, 'train_accs_m2_lr_%s_dp_%s' % (learning_rate, dropout_keep_prob))
        plt.plot_cost(train_costs, n_epochs, 'train_cost_m2_lr_%s_dp_%s' % (learning_rate, dropout_keep_prob))
        plt.plot_cost(train_f1s, n_epochs, 'train_f1_m2_lr_%s_dp_%s' % (learning_rate, dropout_keep_prob))

        plt.plot_cost(val_accs, n_epochs, 'val_accs_m2_lr_%s_dp_%s' % (learning_rate, dropout_keep_prob))
        plt.plot_cost(val_costs, n_epochs, 'val_costs_m2_lr_%s_dp_%s' % (learning_rate, dropout_keep_prob))
        plt.plot_cost(val_f1s, n_epochs, 'val_f1s_m2_lr_%s_dp_%s' % (learning_rate, dropout_keep_prob))

        return train_accs, train_costs, train_f1s, val_accs, val_costs, val_f1s, best_train_f1, best_val_f1


if __name__ == '__main__':
    tf.set_random_seed(12345)
    train_c3d(dropout_keep_prob=0.5,
              edge_size=48,
              learning_rate=0.0001,
              n_epochs=20,
              batch_size=15,
              n_classes=2,
              show_batch_result=True)
