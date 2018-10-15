# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf
from sea import DataFiles, Data, Config
import time
import os
from sklearn.metrics import f1_score
os.environ['CUDA_VISIBLE_DEVICES']='1'


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, filter_sizes, num_filters, batch_size=128, l2_reg_lambda=0.0, embedding_size=256):

        self._embedding_size = embedding_size
        self._batch_size = tf.placeholder(dtype=tf.int64, name="batch_size")
        # Placeholders for input, output and dropout
        self._feature = tf.placeholder(tf.int64, [None, sequence_length], name="feature")
        self._label = tf.placeholder(dtype=tf.int64, shape=[None, None, None], name="label")
        self._feature_length = tf.placeholder(dtype=tf.int64, shape=[None, None], name="feature_length")
        self._keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self._num_filters = num_filters
        self._sequence_length = sequence_length
        self._batch_size = batch_size
        self._data = Data(self._batch_size, max_length=self._sequence_length)
        self._filter_sizes = filter_sizes
        self._l2_reg_lambda = l2_reg_lambda
        self.global_step = tf.get_variable("global_step", initializer=tf.constant(0), trainable=False)
        self._batch_normalization_trainable = tf.placeholder(tf.bool, name="batch_normalization_trainable")

        self._labels_num = 20
        self._output_dimension = 4
        self._learning_rate = tf.train.exponential_decay(1e-1, self.global_step, 3000, 0.96, staircase=True)
        self._checkpoint_path = os.path.dirname('checkpoint/checkpoint')
        self.graph = tf.Graph()


    def _load_data(self):
        self._train_iterator, self._train_iterator_initializer, self._validation_iterator, self._validation_iterator_initializer, self._test_iterator, self._test_iterator_initializer \
            = self._data.load_data()
        self._validation_next = self._validation_iterator.get_next()
        self._actual_batch_size = tf.shape(self._feature)[0]
        self.weights = tf.constant(self._data.weights)

    def _create_embedding(self):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self._data.get_vocab_size(), self._embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self._feature)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

    def _create_convolution(self):
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
                #h1 = tf.layers.batch_normalization(tf.nn.bias_add(conv, b), training=)
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
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        self.h_drop = tf.nn.dropout(self.h_pool_flat, self._keep_prob)

    def _create_output(self):
        with tf.name_scope("output"):
            self._input = tf.layers.dense(inputs=self.h_drop, units=256, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu)
            self._logits = [
                tf.layers.dense(inputs=self._input, units=128, kernel_initializer=tf.truncated_normal_initializer(seed=i, stddev=0.1), activation=tf.nn.relu)
                for i in range(self._labels_num)
            ]

            self._logits = [
                tf.layers.dense(inputs=self._logits[i], units=self._output_dimension, kernel_initializer=tf.truncated_normal_initializer(seed=i * 10, stddev=0.1), activation=tf.nn.sigmoid)
                for i in range(self._labels_num)
            ]
            #self._predict = tf.stack([tf.nn.softmax(logits=self._logits[i], name="softmax" + str(i)) for i in range(self._labels_num)])
            self._predict = tf.stack([self._logits[i] for i in range(self._labels_num)])
            self._predict = tf.argmax(self._predict, axis=2)
            self._predict = tf.one_hot(self._predict, depth=self._output_dimension, dtype=tf.int64)
            self._predict = tf.transpose(self._predict, [1, 0, 2])

    def _create_loss(self):
        with tf.name_scope("create_loss"):
            length = self._labels_num
            w = [tf.nn.embedding_lookup(self.weights[i], tf.argmax(self._label[:, i, :], axis=1), name="embedding_lookup" + str(i)) for i in range(length)]
            self._loss_ = [tf.losses.softmax_cross_entropy(onehot_labels=self._label[:, i, :], logits=self._logits[i], weights=w[i]) for i in range(length)]
            self._loss = tf.stack(self._loss_)
            self._total_loss = tf.reduce_mean(self._loss, axis=0)

    def _create_optimizer(self):
        with tf.name_scope("create_optimizer"):
            #self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._learning_rate)
            self._grads_total = self._optimizer.compute_gradients(self._total_loss)
            self._grads_distribution = [self._optimizer.compute_gradients(self._loss_[i]) for i in range(self._labels_num)]
            self._train_total = self._optimizer.apply_gradients(self._grads_total, global_step=self.global_step)
            self._train_distribution = [self._optimizer.apply_gradients(self._grads_distribution[i], global_step=self.global_step) for i in range(self._labels_num)]

    def _create_summary(self):
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
            [tf.summary.scalar('loss [' + str(i) + ']', self._loss_[i]) for i in range(self._labels_num)]
            [tf.summary.histogram('histogram loss[' + str(i) + ']', self._loss_[i]) for i in range(self._labels_num)]
            tf.summary.histogram('histogram loss', self._total_loss)
            self._summary_op = tf.summary.merge_all()

    def _build_graph(self):
        self._create_embedding()
        self._create_convolution()
        self._create_output()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()

    def build(self):
        self._load_data()
        self._build_graph()

    def validation(self, sess):
        f1 = 0
        samples = 0
        sess.run(self._validation_iterator_initializer)
        all_lab = []
        all_res = []
        print("对验证集进行验证....")
        while True:
            try:
                delta_t = time.time()
                feature, len, label = sess.run(self._validation_next)
                predict, actual_batch_size, lab, res = sess.run(
                    [self._predict, self._actual_batch_size, tf.argmax(label, axis=2) - 2, tf.argmax(self._predict, axis=2) - 2],
                    feed_dict={self._keep_prob: 1.0, self._feature: feature, self._feature_length: len, self._label: label}
                )

                all_lab.extend(lab)
                all_res.extend(res)
                samples += actual_batch_size
                delta_t = time.time() - delta_t
                print("cost time {} sec".format(delta_t))
            except tf.errors.OutOfRangeError:
                print("正在计算f1 score, 请稍等")
                for l1, l2 in zip(all_res, all_lab):
                    f1 += f1_score(l2, l1, average="macro")
                average_f1 = f1 / samples
                print("验证集运行完毕，平均f1为: {}".format(average_f1))
                break

    def train(self, epoches=10):
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")

        saver = tf.train.Saver(sharded=True)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                print("正在从{}加载模型".format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('graphs/ai_challenger/text_cnn/learning_rate' + str(self._learning_rate), self.graph)
            initial_step = self.global_step.eval()
            print("initial_step is {}".format(initial_step))
            total_loss = 0.0
            iteration = 0
            train_next = self._train_iterator.get_next()
            max_loss_indice = None
            total_time = 0
            for i in range(initial_step, initial_step + epoches):
                sess.run(self._train_iterator_initializer)
                while True:
                    try:
                        delta_t = time.time()
                        feature, len, label = sess.run(train_next)
                        if iteration < 20000 or not max_loss_indice:
                            _, loss, summary, global_step = sess.run(
                                [self._train_total, self._total_loss, self._summary_op, self.global_step],
                                feed_dict={
                                    self._keep_prob: 0.5, self._feature: feature, self._feature_length: len, self._label: label
                                }
                            )
                        else:
                            _, _, loss, summary, max_loss_indice, global_step = sess.run(
                                [self._train_distribution[max_loss_indice], self._train_total, self._total_loss, self._summary_op, tf.argmax(self._loss, axis=0), self.global_step],
                                feed_dict={
                                    self._keep_prob: 0.5, self._feature: feature, self._feature_length: len, self._label: label
                                }

                            )
                        total_loss += loss
                        iteration = iteration + 1
                        average_loss = total_loss / iteration
                        writer.add_summary(summary, global_step=global_step)
                        total_time += time.time() - delta_t
                        print("iteration is {}, average_loss is {}, total_time is {}, cost time {}sec/batch".format(iteration, average_loss, total_time, total_time / iteration))

                        if iteration % 1000 == 0:
                            saver.save(sess, save_path="checkpoint/text_cnn", global_step=self.global_step)
                            self.validation(sess)
                        if (global_step + 1) % 30000 == 0:
                            self._test(sess, global_step)

                    except tf.errors.OutOfRangeError:
                        saver.save(sess, save_path="checkpoint/text_cnn", global_step=self.global_step)
                        break

    def _test(self, sess, global_step):
        test_next = self._test_iterator.get_next()
        sess.run(self._test_iterator_initializer, feed_dict={})
        while True:
            try:
                feature, len, label = sess.run(test_next)
                res = sess.run(
                    tf.argmax(self._predict, axis=2, output_type=tf.int64) - 2,
                    feed_dict={self._keep_prob: 1.0, self._feature: feature, self._feature_length: len, self._label: label}
                )
                self._data.feed_output(res)

            except tf.errors.OutOfRangeError:
                break
        self._data.persist("result_{}.csv".format(global_step))

    def test(self):
        self.build()
        saver = tf.train.Saver(sharded=True)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("no model!")
                exit(0)
            test_next = self._test_iterator.get_next()
            sess.run(self._test_iterator_initializer, feed_dict={})
            while True:
                try:
                    feature, len, label = sess.run(test_next)
                    predict = sess.run(
                        self._predict, feed_dict={self._keep_prob: 1.0, self._feature: feature, self._feature_length: len, self._label: label}
                    )
                    res = sess.run(tf.argmax(predict, axis=2, output_type=tf.int64) - 2)
                    self._data.feed_output(res)

                except tf.errors.OutOfRangeError:
                    break
            self._data.persist()


if __name__ == "__main__":
    Config._use_lemma = False
    model = TextCNN(sequence_length=3000, filter_sizes=[3, 4, 5], num_filters=256)
    model.build()
    model.train(300)