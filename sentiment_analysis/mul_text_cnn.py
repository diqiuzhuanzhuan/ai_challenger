# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""



import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class TextCNN(object):

    def __init__(self, batch_size, learning_rate, embedding_size, vocab_size, sequence_length, num_filters, filter_sizes, weight, labes_num=20, output_dimension=4, next_element=None):
        self._sequence_length = sequence_length
        self._num_filters = num_filters
        self._filter_sizes = filter_sizes
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._embedding_size = embedding_size
        self._vocab_size = vocab_size
        self._labels_num = labes_num
        self._output_dimension = output_dimension
        self.weights = weight

        # placeholder
#        self._feature = tf.placeholder(tf.int64, [None, self._sequence_length], name="feature")
#        self._label = tf.placeholder(tf.int64, [None, None, None], name="label")
#        self._feature_length = tf.placeholder(dtype=tf.int64, shape=[None, None], name="feature_length")

        self._feature = next_element[0]
        self._feature_length = next_element[1]
        self._label = next_element[2]
        self._actual_batch_size = tf.shape(self._feature)[0]

        self._keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # global_step
        self.global_step = tf.get_variable("global_step", initializer=tf.constant(0), trainable=False)

        # decay learning
        self._train_learning_rate = tf.train.exponential_decay(1e-1, self.global_step, 3000, 0.96, staircase=True)

        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self._vocab_size, self._embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self._feature)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self._filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self._embedding_size, 1, self._num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self._num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                # h1 = tf.layers.batch_normalization(tf.nn.bias_add(conv, b), training=)
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self._sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self._num_filters * len(self._filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        self.h_drop = tf.nn.dropout(h_pool_flat, self._keep_prob)

        # create output
        with tf.name_scope("output"):
            def my_dense(inputs, units, activation):
                return tf.contrib.layers.fully_connected(inputs, units, activation)

            input_1_layer = my_dense(self.h_drop, 256, tf.nn.relu)

            input_2_layer = [my_dense(input_1_layer, units=128, activation=tf.nn.relu) for _ in range(self._labels_num)]

            input_3_layer = [my_dense(inputs=input_2_layer[i], units=self._output_dimension, activation=None) for i in range(self._labels_num)]
            self.logits = input_3_layer

            predict = tf.argmax(tf.stack(input_3_layer), axis=2)

            self.predict = tf.transpose(predict, [1, 0], name="predict")

        with tf.name_scope("create_loss"):
            length = self._labels_num
            w = [tf.nn.embedding_lookup(self.weights[i], tf.argmax(self._label[:, i, :], axis=1), name="embedding_lookup" + str(i)) for i in range(length)]
            self._loss_ = [tf.losses.softmax_cross_entropy(onehot_labels=self._label[:, i, :], logits=self.logits[i], weights=w[i]) for i in range(length)]
            total_loss = tf.stack(self._loss_)
            self._total_loss = tf.reduce_mean(total_loss, axis=0)

        with tf.name_scope("create_optimizer"):
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._train_learning_rate)
            self._grads_total = optimizer.compute_gradients(self._total_loss)
            #self._grads_distribution = [optimizer.compute_gradients(self._loss_[i]) for i in range(self._labels_num)]
            self._train_total = optimizer.apply_gradients(self._grads_total, global_step=self.global_step)
            #self._train_distribution = [optimizer.apply_gradients(self._grads_distribution[i], global_step=self.global_step) for i in range(self._labels_num)]

        with tf.name_scope("create_summary"):
            grad_summaries = []
            for g, v in self._grads_total:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)

            tf.summary.scalar('learning_rate', self._learning_rate)
            tf.summary.scalar('loss', self._total_loss)
            [tf.summary.scalar('loss [t' + str(i) + ']', self._loss_[i]) for i in range(self._labels_num)]
            [tf.summary.histogram('histogram loss[' + str(i) + ']', self._loss_[i]) for i in range(self._labels_num)]
            tf.summary.histogram('histogram loss', self._total_loss)
            self._summary_op = tf.summary.merge_all()

