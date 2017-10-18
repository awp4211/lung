import tensorflow as tf
import ploter as plt


def conv3d(x, n_filters, kernels, strides, stddev=0.1,
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
                            [kernels[0], kernels[1], kernels[2], x.get_shape()[-1], n_filters],
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


def get_weight_varible(name, shape):
    return tf.get_variable(name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def get_bias_varible(name, shape):
    return tf.get_variable(name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def fc(layer_name, x, inp_shape, out_shape):
    with tf.variable_scope(layer_name):
        inp_dim = inp_shape[-1]
        out_dim = out_shape[-1]
        y = tf.reshape(x, shape=inp_shape)
        w = get_weight_varible('w', [inp_dim, out_dim])
        b = get_bias_varible('b', [out_dim])
        y = tf.add(tf.matmul(y, w), b)
        return y


def c3d_net(x, dropout_keep_prob):
    print 'volume shape = %s' % x.get_shape()
    with tf.name_scope('input_reshape'):
        x = tf.reshape(x, [-1,
                           x.get_shape().as_list()[1],
                           x.get_shape().as_list()[2],
                           x.get_shape().as_list()[3],
                           1])
    print 'image reshape, shape = %s' % x.get_shape()

    # conv1 96C5*5*5
    with tf.name_scope('conv1'):
        net = conv3d(x, n_filters=96,
                     kernels=[5, 5, 5],
                     strides=[1, 1, 1],
                     activation=tf.nn.relu,
                     name='conv1',
                     padding='VALID')
    print 'conv1, shape = %s' % net.get_shape()

    # maxPool1 2*2*2
    with tf.name_scope('maxpool1'):
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1],
                               strides=[1, 2, 2, 2, 1],
                               padding='VALID')
    print 'maxpool1, shape = %s' % net.get_shape()

    # conv2 256C4*4*4
    with tf.name_scope('conv2'):
        net = conv3d(net, n_filters=256,
                     kernels=[4, 4, 4],
                     strides=[1, 1, 1],
                     activation=tf.nn.relu,
                     name='conv2',
                     padding='VALID')
    print 'conv2, shape = %s' % net.get_shape()

    # maxpool2 2*2*2
    with tf.name_scope('maxpool2'):
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1],
                               strides=[1, 2, 2, 2, 1],
                               padding='VALID')
    print 'maxpool2, shape = %s' % net.get_shape()

    # conv3 384C3*3*3
    with tf.name_scope('conv3'):
        net = conv3d(net, n_filters=384,
                     kernels=[3, 3, 3],
                     strides=[1, 1, 1],
                     activation=tf.nn.relu,
                     name='conv3',
                     padding='VALID')
    print 'conv3, shape = %s' % net.get_shape()

    # conv4 384C3*3*3
    with tf.name_scope('conv4'):
        net = conv3d(net, n_filters=384,
                     kernels=[3, 3, 3],
                     strides=[1, 1, 1],
                     activation=tf.nn.relu,
                     name='conv4',
                     padding='SAME')
    print 'conv4, shape = %s' % net.get_shape()

    # conv5 256C3*3*3
    with tf.name_scope('conv5'):
        net = conv3d(net, n_filters=512,
                     kernels=[3, 3, 3],
                     strides=[1, 1, 1],
                     activation=tf.nn.relu,
                     name='conv5',
                     padding='SAME')
    print 'conv5, shape = %s' % net.get_shape()

    # maxpool3 2*2*2
    with tf.name_scope('maxpool3'):
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1],
                               strides=[1, 2, 2, 2, 1],
                               padding='SAME')
    print 'maxpool3, shape = %s' % net.get_shape()

    # conv6 256C3*3*3
    with tf.name_scope('conv6'):
        net = conv3d(net, n_filters=1024,
                     kernels=[3, 3, 3],
                     strides=[1, 1, 1],
                     activation=tf.nn.relu,
                     name='conv6',
                     padding='SAME')
    print 'conv6, shape = %s' % net.get_shape()

    with tf.name_scope('maxpool4'):
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1],
                               strides=[1, 2, 2, 2, 1],
                               padding='SAME')
    print 'maxpool4, shape = %s' % net.get_shape()

    with tf.name_scope('flatten_layer'):
        net = tf.reshape(
            net,
            [-1, net.get_shape().as_list()[1]*net.get_shape().as_list()[2]*
             net.get_shape().as_list()[3] *net.get_shape().as_list()[4]]
        )
    print 'flatten_layer, shape = %s' % net.get_shape()

    # dropout
    with tf.name_scope('dropout'):
        net = tf.nn.dropout(net, keep_prob=dropout_keep_prob)

    # FC1VALID
    with tf.name_scope('fc1'):
        net = fc('fc1', net, [-1, net.get_shape().as_list()[1]], [-1, 4096])
        net = tf.nn.tanh(net)
    print 'fc1, shape = %s' % net.get_shape()

    # FC2
    with tf.name_scope('fc2'):
        net = fc('fc2', net, [-1, net.get_shape().as_list()[1]], [-1, 4096])
        net = tf.nn.tanh(net)
    print 'fc2, shape = %s' % net.get_shape()

    # FC3
    with tf.name_scope('fc3'):
        net = fc('fc3', net, [-1, net.get_shape().as_list()[1]], [-1, 2])
        net = tf.nn.tanh(net)
    print 'fc3, shape = %s' % net.get_shape()

    return net


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # note that each grad_and_vars looks like the following:
        # ((grad_gpu_0, var0_gpu0), ..., (grad_gpu_N, var0_gpuN) )
        grads = [g for g, _ in grad_and_vars]
        # average over the "tower" dimension
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # keep in mind that the variables are redundant because they are shared
        # accross towers. So we will just return the first tower's pointer to the variables
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def feed_all_gpu(input_dict, models, payload_per_gpu, batch_x, batch_y, dropout_keep_prob):
    for i in range(len(models)):
        x, y, keep_prob, _, _, _ = models[i]
        start_pos = i * payload_per_gpu
        stop_pos = (i+1) * payload_per_gpu
        input_dict[x] = batch_x[start_pos:stop_pos]
        input_dict[y] = batch_y[start_pos:stop_pos]
        input_dict[keep_prob] = dropout_keep_prob
    return input_dict


def paralled_train(dropout_keep_prob=0.5,
                   edge_size=48,
                   learning_rate=1e-6,
                   n_epochs=20,
                   batch_size=30,
                   n_classes=2,
                   show_batch_result=False,
                   num_gpus=4):
    print locals()
    batch_size *= num_gpus

    from data_reader import load_train_or_val_dataset_full
    from metrics import metric, metric_
    from glob import glob
    from tqdm import tqdm

    import os

    path = os.path.join(os.path.dirname(os.getcwd()), 'output')

    train_data_files = glob(path + '/train_*.h5')
    val_data_files = glob(path + '/val_*.h5')

    print "...... building model ......"

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

            print "[x] building model on gpu towers ......"
            models = []
            for gpu_id in range(num_gpus):
                with tf.device("/gpu:%d" % gpu_id):
                    print "[x] tower:%d ......" % gpu_id
                    with tf.name_scope("tower_%d" % gpu_id):
                        with tf.variable_scope("cpu_variables", reuse=gpu_id > 0):
                            x = tf.placeholder(tf.float32, [None, edge_size, edge_size, edge_size])
                            y = tf.placeholder(tf.float32, [None, n_classes])
                            keep_prob = tf.placeholder(tf.float32, name="keep_prob")

                            y_conv = c3d_net(x, dropout_keep_prob=keep_prob)
                            y_prob = tf.nn.softmax(y_conv)
                            cost = tf.reduce_mean(tf.square(y - y_prob))

                            grads = opt.compute_gradients(cost)
                            models.append((x, y, keep_prob, y_prob, cost, grads))
            print "[x] building model on gpu tower done ......"

            # constants
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

            print "[x] reduce model on cpu ......"
            print "[!] have been created %d models ......" % (len(models))
            tower_x, tower_y, tower_keep_prob, tower_probs, tower_costs, tower_grads = zip(*models)
            aver_loss_op = tf.reduce_mean(tower_costs)
            apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads))

            all_y = tf.reshape(tf.stack(tower_y, 0), [-1, n_classes])
            all_pred = tf.reshape(tf.stack(tower_probs, 0), [-1, n_classes])
            correct_pred = tf.equal(tf.argmax(all_y, 1), tf.argmax(all_pred, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
            print "[x] reduce model on cpu done ..."

            print "...... loading dataset ......"
            train_set_x, train_set_y = load_train_or_val_dataset_full(train_data_files)
            n_train_examples = train_set_x.shape[0]

            val_set_x, val_set_y = load_train_or_val_dataset_full(val_data_files)
            n_validation_examples = val_set_x.shape[0]

            print "[x] run init op ..."
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            print "[x] run train op ..."
            for epoch_i in tqdm(range(n_epochs)):
                payload_per_gpu = batch_size / num_gpus

                train_cost = 0.
                train_acc = 0.
                train_f1 = 0.
                n_train_examples_total = 0.
                y_probs_train = []
                y_true_train = []

                y_probs_val = []
                y_true_val = []

                for mini_batch in range(n_train_examples // batch_size):
                    batch_xs = train_set_x[mini_batch * batch_size:(mini_batch + 1) * batch_size]
                    batch_ys = train_set_y[mini_batch * batch_size:(mini_batch + 1) * batch_size]
                    input_dict = {}
                    input_dict = feed_all_gpu(input_dict, models, payload_per_gpu, batch_xs, batch_ys, dropout_keep_prob)
                    _, t_loss, t_acc, y_p = sess.run([apply_gradient_op,
                                                      aver_loss_op,
                                                      accuracy,
                                                      all_pred],
                                                     feed_dict=input_dict)
                    y_probs_train.append(y_p)
                    y_true_train.append(batch_ys)
                    train_cost += t_loss
                    train_acc += t_acc
                    if show_batch_result:
                        print 'epoch %d,, minibatch %d, train acc = %f %%, train_cost = %f ,train_f1 = %f' % (
                            epoch_i, mini_batch, t_acc * 100, t_loss, metric_(y_p, batch_ys))

                train_acc /= (n_train_examples_total // batch_size)
                train_cost /= (n_train_examples_total // batch_size)
                train_f1 = metric(y_probs_train, y_true_train)

                # validation files
                validation_cost = 0.
                validation_acc = 0.
                validation_f1 = 0.

                for mini_batch in range(n_validation_examples // batch_size):
                    batch_xs = val_set_x[mini_batch * batch_size:(mini_batch + 1) * batch_size]
                    batch_ys = val_set_y[mini_batch * batch_size:(mini_batch + 1) * batch_size]
                    input_dict = {}
                    input_dict = feed_all_gpu(input_dict, models, payload_per_gpu, batch_xs, batch_ys, 1.0)
                    v_loss, v_acc, v_prob = sess.run([aver_loss_op, accuracy, all_pred],
                                                     feed_dict=input_dict)
                    y_probs_val.append(v_prob)
                    y_true_val.append(batch_ys)
                    validation_cost += v_loss
                    validation_acc += v_acc

                validation_acc /= (n_validation_examples // batch_size)
                validation_cost /= (n_validation_examples // batch_size)
                validation_f1 = metric(y_probs_val, y_true_val)

                if train_f1 > best_train_f1:
                    best_train_f1 = train_f1
                    best_train_epoch = epoch_i

                if validation_f1 > best_val_f1:
                    best_val_f1 = validation_f1
                    best_val_epoch = epoch_i
                    saver.save(sess, 'trained_model/3dcnn_m1_f1_lr_%s_dp_%s.ckpt' % (learning_rate, dropout_keep_prob))

                if validation_cost < best_val_cost:
                    best_val_cost = validation_cost
                    saver.save(sess,
                               'trained_model/3dcnn_m1_cost_lr_%s_dp_%s.ckpt' % (learning_rate, dropout_keep_prob))

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

            plt.plot_cost(train_accs, n_epochs, 'train_accs_m1_lr_%s_dp_%s' % (learning_rate, dropout_keep_prob))
            plt.plot_cost(train_costs, n_epochs, 'train_cost_m1_lr_%s_dp_%s' % (learning_rate, dropout_keep_prob))
            plt.plot_cost(train_f1s, n_epochs, 'train_f1_m1_lr_%s_dp_%s' % (learning_rate, dropout_keep_prob))

            plt.plot_cost(val_accs, n_epochs, 'val_accs_m1_lr_%s_dp_%s' % (learning_rate, dropout_keep_prob))
            plt.plot_cost(val_costs, n_epochs, 'val_costs_m1_lr_%s_dp_%s' % (learning_rate, dropout_keep_prob))
            plt.plot_cost(val_f1s, n_epochs, 'val_f1s_m1_lr_%s_dp_%s' % (learning_rate, dropout_keep_prob))

            return train_accs, train_costs, train_f1s, val_accs, val_costs, val_f1s, best_train_f1, best_val_f1


if __name__ == '__main__':
    tf.set_random_seed(12345)
    paralled_train(dropout_keep_prob=0.5,
                   edge_size=48,
                   learning_rate=1e-6,
                   n_epochs=20,
                   batch_size=30,
                   n_classes=2,
                   show_batch_result=True,
                   num_gpus=4)