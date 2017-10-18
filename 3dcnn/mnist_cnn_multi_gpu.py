import os
import sys
import numpy as np
import time
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def get_weight_varible(name, shape):
    return tf.get_variable(name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def get_bias_varible(name, shape):
    return tf.get_variable(name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def conv2d(layer_name, x, filter_shape):
    with tf.variable_scope(layer_name):
        w = get_weight_varible('w', filter_shape)
        b = get_bias_varible('b', filter_shape[-1])
        y = tf.nn.bias_add(tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME'), b)
        return y


def pool2d(layer_name, x):
    with tf.variable_scope(layer_name):
        y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return y


def fc(layer_name, x, inp_shape, out_shape):
    with tf.variable_scope(layer_name):
        inp_dim = inp_shape[-1]
        out_dim = out_shape[-1]
        y = tf.reshape(x, shape=inp_shape)
        w = get_weight_varible('w', [inp_dim, out_dim])
        b = get_bias_varible('b', [out_dim])
        y = tf.add(tf.matmul(y, w), b)
        return y


def build_model(x):
    y = tf.reshape(x,shape=[-1, 28, 28, 1])
    #layer 1
    y = conv2d('conv_1', y, [3, 3, 1, 8])
    y = pool2d('pool_1', y)
    #layer 2
    y = conv2d('conv_2', y, [3, 3, 8, 16])
    y = pool2d('pool_2', y)
    #layer fc
    y = fc('fc', y, [-1, 7*7*16], [-1, 10])
    return y


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # note that each grad_and_vars looks like the following:
        # ((grad_gpu_0, var0_gpu0), ..., (grad_gpu_N, var0_gpuN) )
        grads = [g for g, _ in grad_and_vars]
        # average over the "tower" dimension
        print grads
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # keep in mind that the variables are redundant because they are shared
        # accross towers. So we will just return the first tower's pointer to the variables
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def feed_all_gpu(input_dict, models, payload_per_gpu, batch_x, batch_y):
    for i in range(len(models)):
        x, y, _, _, _ = models[i]
        start_pos = i * payload_per_gpu
        stop_pos = (i+1) * payload_per_gpu
        input_dict[x] = batch_x[start_pos:stop_pos]
        input_dict[y] = batch_y[start_pos:stop_pos]
    return input_dict


def multi_gpu(num_gpu, n_epoch=10):
    batch_size = 128*num_gpu
    mnist_data_path = os.path.join(os.path.dirname(os.getcwd()), 'MNIST_DATA')
    mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)

    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            learning_rate = tf.placeholder(tf.float32, shape=[])
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

            print "[x] build model on gpu tower ..."
            models = []
            for gpu_id in range(num_gpu):
                with tf.device("/gpu:%d" % gpu_id):
                    print "[x] tower:%d ..." % gpu_id
                    with tf.name_scope("tower_%d" % gpu_id):
                        with tf.variable_scope("cpu_variables", reuse=gpu_id > 0):
                            x = tf.placeholder(tf.float32, [None, 784])
                            y = tf.placeholder(tf.float32, [None, 10])
                            pred = build_model(x)
                            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
                            grads = opt.compute_gradients(loss)
                            models.append((x, y, pred, loss, grads))
            print "[x] build model on gpu tower done"

            print "[x] reduce model on cpu ..."
            print len(models)
            tower_x, tower_y, tower_preds, tower_losses, tower_grads = zip(*models)
            aver_loss_op = tf.reduce_mean(tower_losses)
            apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads))

            all_y = tf.reshape(tf.stack(tower_y, 0), [-1, 10])
            all_pred = tf.reshape(tf.stack(tower_preds, 0), [-1, 10])
            correct_pred = tf.equal(tf.argmax(all_y, 1), tf.argmax(all_pred, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
            print "[x] reduce model on cpu done ..."

            print "[x] run train op ..."
            sess.run(tf.global_variables_initializer())
            lr = 0.01
            for epoch in range(n_epoch):
                start_time = time.time()
                payload_per_gpu = batch_size / num_gpu
                total_batch = int(mnist.train.num_examples/batch_size)
                avg_loss = 0.0
                print "\n-----------------------"
                print "epoch:%d, lr:%.4f" % (epoch, lr)

                for batch_idx in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    input_dict = {}
                    input_dict[learning_rate] = lr
                    input_dict = feed_all_gpu(input_dict, models, payload_per_gpu, batch_x, batch_y)
                    _, _loss = sess.run([apply_gradient_op, aver_loss_op], input_dict)
                    avg_loss += _loss

                avg_loss /= total_batch
                print "training loss:%.4f" % avg_loss
                lr = max(lr*0.7, 0.00001)

                # validation
                val_payload_per_gpu = batch_size / num_gpu
                total_batch = int(mnist.validation.num_examples / batch_size)
                preds = None
                ys = None
                for batch_idx in range(total_batch):
                    batch_x, batch_y = mnist.validation.next_batch(batch_size)
                    input_dict = feed_all_gpu({}, models, val_payload_per_gpu, batch_x, batch_y)
                    batch_pred, batch_y = sess.run([all_pred, all_y], input_dict)

                    preds = batch_pred if preds is None else np.concatenate((preds, batch_pred),0)
                    ys = batch_y if ys is None else np.concatenate((ys, batch_y), 0)
                val_accuracy = sess.run([accuracy], {all_y: ys, all_pred:preds})[0]
                print "val accuracy: %0.4f%%" % (100.0 * val_accuracy)

                stop_time = time.time()
                elapsed_time = stop_time - start_time
                print "cost time: " + str(elapsed_time) + " sec."
            print "training done."

            # test model
            test_payload_per_gpu = batch_size / num_gpu
            total_batch = int(mnist.test.num_examples / batch_size)
            preds = None
            ys = None
            for batch_idx in range(total_batch):
                batch_x, batch_y = mnist.test.next_batch(batch_size)
                inp_dict = feed_all_gpu({}, models, test_payload_per_gpu, batch_x, batch_y)
                batch_pred, batch_y = sess.run([all_pred, all_y], inp_dict)
                if preds is None:
                    preds = batch_pred
                else:
                    preds = np.concatenate((preds, batch_pred), 0)
                if ys is None:
                    ys = batch_y
                else:
                    ys = np.concatenate((ys, batch_y), 0)
            test_accuracy = sess.run([accuracy], {all_y: ys, all_pred: preds})[0]
            print('Test Accuracy: %0.4f%%\n\n' % (100.0 * test_accuracy))


if __name__ == "__main__":
    #multi_gpu(1)
    #multi_gpu(2)
    #multi_gpu(3)
    multi_gpu(4)
