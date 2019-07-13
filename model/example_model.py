from model.rule_layer import *
from tensorflow.python.saved_model import tag_constants
import csv
import numpy as np
import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.reset_default_graph()
# model +++++++++++++++++++++++++++++++++++++++++++++++++


class network():
    def get_model(self, X, real_in, avg_pool, pretrain, activation, pool_by_stride, bn_before, bn_after, ind_scaling, pool_before, pool_after, skip, pool_skip):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Global settings
        n_filter, s_filter, drop = 128, 3, 1.

        pooling = pool_by_stride or pool_after or pool_before

        thres = []
        X = tf.reshape(X, (-1, 28, 28, 1))

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Input layer
        if not real_in:
            for i in range(1):
                thres.append(input_exp(X, i, activation=activation, ind_scaling=ind_scaling))
            X = tf.concat(thres, 3)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Layer 1
        if skip:
            if pool_skip:
                X = rule_conv_block(X, pretrain=pretrain, nr=1, n_filter=n_filter, s_filter=s_filter, pool_by_stride=pool_by_stride, activation=activation,
                                    bn_before=bn_before, bn_after=bn_after, ind_scaling=ind_scaling, pool_before=pool_before, pool_after=pool_after, avg_pool=avg_pool,
                                    skip=False)
            else:
                X = rule_conv_block(X, pretrain=pretrain, nr=1, n_filter=n_filter, s_filter=s_filter, activation=activation,
                                    bn_before=bn_before, bn_after=bn_after, ind_scaling=ind_scaling,
                                    skip=False)
        else:
            X = rule_conv_block(X, pretrain=pretrain, nr=1, n_filter=n_filter, s_filter=s_filter, pool_by_stride=pool_by_stride, activation=activation,
                                bn_before=bn_before, bn_after=bn_after, ind_scaling=ind_scaling, pool_before=pool_before, pool_after=pool_after, avg_pool=avg_pool)

        if skip:
            if pool_skip:
                X = rule_conv_block(X, pretrain=pretrain, nr=2, n_filter=n_filter, s_filter=s_filter, pool_by_stride=pool_by_stride, activation=activation,
                                    bn_before=bn_before, bn_after=bn_after, ind_scaling=ind_scaling, pool_before=pool_before, pool_after=pool_after, avg_pool=avg_pool,
                                    skip=False)
            else:
                X = rule_conv_block(X, pretrain=pretrain, nr=2, n_filter=n_filter, s_filter=s_filter, activation=activation,
                                    bn_before=bn_before, bn_after=bn_after, ind_scaling=ind_scaling,
                                    skip=False)
        else:
            X = rule_conv_block(X, pretrain=pretrain, nr=2, n_filter=n_filter, s_filter=s_filter, pool_by_stride=pool_by_stride, activation=activation,
                                bn_before=bn_before, bn_after=bn_after, ind_scaling=ind_scaling, pool_before=pool_before, pool_after=pool_after, avg_pool=avg_pool)

        if skip:
            if pool_skip:
                X = rule_conv_block(X, pretrain=pretrain, nr=3, n_filter=n_filter, s_filter=s_filter, pool_by_stride=pool_by_stride, activation=activation,
                                    bn_before=bn_before, bn_after=bn_after, ind_scaling=ind_scaling, pool_before=pool_before, pool_after=pool_after, avg_pool=avg_pool,
                                    skip=False)
            else:
                X = rule_conv_block(X, pretrain=pretrain, nr=3, n_filter=n_filter, s_filter=s_filter, activation=activation,
                                    bn_before=bn_before, bn_after=bn_after, ind_scaling=ind_scaling,
                                    skip=False)
        else:
            X = rule_conv_block(X, pretrain=pretrain, nr=3, n_filter=n_filter, s_filter=s_filter, pool_by_stride=pool_by_stride, activation=activation,
                                bn_before=bn_before, bn_after=bn_after, ind_scaling=ind_scaling, pool_before=pool_before, pool_after=pool_after, avg_pool=avg_pool)

        X = tf.nn.dropout(X, drop)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Fully connected layer and softmax output
        X = tf.layers.flatten(X)
        X = rule_dense_eff(X, 10, int(X.shape[1]), 1, activation=activation)

        X = tf.layers.dense(X, self.classes, tf.nn.softmax)

        return X

    def training(self, train, label_train, test, label_test, pretrain=False):
        loss_list, val_list = [], []
        with tf.Session() as sess:
            sess.run(self.init)
            best_acc_so_far = 0

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Pretrain
            if pretrain:
                for iteration in range(self.n_iterations):
                    indices = np.random.choice(len(train), self.batch_size)
                    batch_X = train[indices]
                    batch_Y = label_train[indices]
                    feed_dict = {self.X: batch_X, self.Y: batch_Y, self.pretrain: True}

                    _, lo = sess.run([self.step, self.loss], feed_dict=feed_dict)

                    if iteration % self.check_every == 0:
                        indices = np.random.choice(len(test), 500)
                        acc, lo = sess.run([self.accuracy, self.loss], feed_dict={
                            self.X: test[indices], self.Y: label_test[indices], self.pretrain: True})
                        if iteration % self.print_every == 0:
                            print("Iteration: ", iteration, "Acc.: ", acc, flush=True)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Train
            for iteration in range(self.n_iterations):
                indices = np.random.choice(len(train), self.batch_size)
                batch_X = train[indices]
                batch_Y = label_train[indices]
                feed_dict = {self.X: batch_X, self.Y: batch_Y,  self.pretrain: False}

                _, lo = sess.run([self.step, self.loss], feed_dict=feed_dict)

                if iteration % self.check_every == 0:
                    indices = np.random.choice(len(test), 2500)
                    acc, lo = sess.run([self.accuracy, self.loss], feed_dict={
                        self.X: test[indices], self.Y: label_test[indices], self.pretrain: False})

                    loss_list.append(lo)
                    val_list.append(acc)

                    if acc > best_acc_so_far:
                        best_acc_so_far = acc
                        save_path = self.saver.save(sess, self.folder_to_save)
                        # print("Model saved in path: %s" % save_path)

                    if iteration % self.print_every == 0:
                        print("Iteration: ", iteration, "Acc.: ", acc, flush=True)

        with open(os.path.dirname(os.path.realpath(__file__)) + "/stored_results/" + str(self.name), "w") as f:
            writer = csv.writer(f)
            writer.writerow(loss_list)
            writer.writerow(val_list)

    def __init__(self, name, avg_pool, real_in, lr=1E-3, batch_size=2**8, activation=binarize_STE, pool_by_stride=False, bn_before=False, bn_after=False, ind_scaling=False, pool_before=False, pool_after=False, skip=False, pool_skip=False):
        # config ++++++++++++++++++++++++++++++++++++++++++++++++
        tf.reset_default_graph()
        self.lr = lr
        self.classes = 10
        self.dtype, self.shape = tf.float32, [None, 28, 28]
        self.n_iterations, self.batch_size, self.print_every, self.check_every = 2**18, batch_size, 2**12, 2**7
        self.folder_to_save, self.name = os.path.dirname(
            os.path.realpath(__file__)) + "/stored_models/" + str(name), name

        # targets ++++++++++++++++++++++++++++++++++++++++++++++
        self.pretrain = tf.placeholder(dtype=tf.bool)
        self.X = tf.placeholder(dtype=self.dtype, shape=self.shape)
        self.Y = tf.placeholder(dtype=self.dtype, shape=[None, self.classes])
        self.y = self.get_model(self.X, real_in=real_in, avg_pool=avg_pool, pretrain=self.pretrain, pool_by_stride=pool_by_stride, activation=activation, bn_before=bn_before, bn_after=bn_after,
                                ind_scaling=ind_scaling, pool_before=pool_before, pool_after=pool_after, skip=skip, pool_skip=pool_skip)

        self.loss = tf.reduce_mean(-tf.reduce_sum(self.Y *
                                                  tf.log(self.y + 1E-10), reduction_indices=[1]))  # + reg2
        self.step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # Evaluate model
        self.one_hot_out = tf.argmax(self.y, 1)
        self.hits = tf.equal(self.one_hot_out, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.hits, tf.float32))

        # Initialize the variables
        self.init = tf.global_variables_initializer()

        # Save model
        self.saver = tf.train.Saver()


class stored_network(object):
    def __init__(self, nn):
        self.sess = tf.Session()
        self.nn = nn
        self.restore()

    def print_to_stderr(self, input, op_name="dcdl_conv_1/_out1"):
        #print(op_name, file=sys.stderr, flush=True)
        self.sess.run(tf.get_default_graph().get_operation_by_name(
            op_name), feed_dict={self.nn.X: input})

    def restore(self):
        # Restore variables from disk.
        self.nn.saver.restore(self.sess, self.nn.folder_to_save)

    def predict(self, input):
        return self.sess.run(self.nn.one_hot_out, feed_dict={self.nn.X: input,  self.pretrain: False})

    def evaluate(self, input, label):
        return self.sess.run([self.nn.accuracy], feed_dict={self.nn.X: input, self.nn.Y: label,  self.nn.pretrain: False})[0]

    def __del__(self):
        self.close()

    def close(self):
        self.sess.close()


def get_input_shape():
    return shape[1:]
