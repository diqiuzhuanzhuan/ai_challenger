# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf
from tensorflow.contrib import *
import os
from sea import DataFiles, Data
from sklearn.metrics import f1_score

os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2'


class MoonLight(object):
    _train_file_names = DataFiles._train_file_names
    _validation_file_names = DataFiles._validation_file_names
    _test_file_names = DataFiles._test_file_names

    _batch_size = 64
    _table = None
    _vocab_size = 0

    _train_batch = None

    _embedding_dimension = 50
    _lstm_unit = 128
    _lstm_layers = 1
    _keep_prob = None
    _attention_length = 40
    #训练数据中标签总共有20个
    _labels_num = 20
    _output_dimension = 4

    def __init__(self, embedding_dimension=256):
        self._embedding_dimension = embedding_dimension
        self._keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        self._feature = tf.placeholder(dtype=tf.int64, shape=[None, None], name="feature")
        self._feature_length = tf.placeholder(dtype=tf.int64, shape=[None, None], name="feature_length")
        self._label = tf.placeholder(dtype=tf.int64, shape=[None, None, None], name="label")
        self.global_step = tf.get_variable("global_step", initializer=tf.constant(0), trainable=False)
        self._checkpoint_path = os.path.dirname('checkpoint/checkpoint')
        self._batch_size = tf.placeholder(name="batch_size", shape=[], dtype=tf.int64)
        self._learning_rate = tf.placeholder(name="learning_rate", dtype=tf.float32)
        self._actual_batch_size = None
        self._batch_size = 32
        self._data = Data(self._batch_size)
        self.weights = None
        self.graph = tf.Graph()

    def load_data(self):

        self._train_iterator, self._train_iterator_initializer, self._validation_iterator, self._validation_iterator_initializer, self._test_iterator, self._test_iterator_initializer\
                                                                                                                                                                = self._data.load_data()
        self._validation_next = self._validation_iterator.get_next()
        self._actual_batch_size = tf.shape(self._feature)[0]
        self.weights = tf.constant(self._data.weights)

    def _create_embedding(self):
        with tf.name_scope("create_embedding"):
            self._embedding_matrix = tf.get_variable(name="embedding_matrix", shape=[self._data.get_vocab_size(), self._embedding_dimension],
                                              initializer=tf.variance_scaling_initializer)

            self._embedding = tf.nn.embedding_lookup(self._embedding_matrix, self._feature, name="embedding")

    def _create_bilstm(self):

        def lstm_cell(lstm_unit):
            cell = tf.nn.rnn_cell.GRUCell(num_units=lstm_unit)
#            cell = rnn.AttentionCellWrapper(cell=cell, attn_length=self._attention_length, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=self._keep_prob)
            return cell

        with tf.name_scope("create_bilstm"):
            stack_fw_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_unit=self._lstm_unit) for _ in range(self._lstm_layers)])
            initial_state_fw = stack_fw_lstm.zero_state(tf.to_int32(self._actual_batch_size), tf.float32)
            stack_bw_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_unit=self._lstm_unit) for _ in range(self._lstm_layers)])
            initial_state_bw = stack_bw_lstm.zero_state(tf.to_int32(self._actual_batch_size), tf.float32)
            self._sentence_encoder_output, self._states = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_fw_lstm, cell_bw=stack_bw_lstm, inputs=self._embedding, sequence_length=tf.reshape(self._feature_length, [-1]),
                                            initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw)

            self.s = self._sentence_encoder_output
            self._sentence_encoder_output = tf.concat(self._sentence_encoder_output, 2)

    def _create_output(self):
        with tf.name_scope("create_output"):
            length = self._labels_num
            output_dimension = self._output_dimension
            indices = tf.stack([tf.range(self._actual_batch_size, dtype=tf.int32), tf.to_int32(tf.reshape(self._feature_length-1, [-1]))], axis=1)
            self._input = tf.gather_nd(self._sentence_encoder_output, indices)
            self._input = tf.layers.dense(inputs=self._input, units=256, kernel_initializer=tf.truncated_normal_initializer, activation=tf.nn.sigmoid)

            self._logits = [
                tf.layers.dense(inputs=self._input, units=128, kernel_initializer=tf.truncated_normal_initializer(seed=i, stddev=0.01, mean=0), activation=tf.nn.sigmoid)
                for i in range(length)
            ]
            self._logits = [
                tf.layers.dense(inputs=self._logits[i], units=output_dimension, kernel_initializer=tf.truncated_normal_initializer(seed=i*10, stddev=0.01, mean=0), activation=tf.nn.sigmoid)
                for i in range(length)
            ]
            self._predict = tf.stack([tf.nn.softmax(logits=self._logits[i], name="softmax"+str(i)) for i in range(length)])
            self._predict = tf.argmax(self._predict, axis=2)
            self._predict = tf.one_hot(self._predict, depth=output_dimension, dtype=tf.int64)
            self._predict = tf.transpose(self._predict, [1, 0, 2])

    def _create_metrics(self):
        with tf.name_scope("create_metrics"):
            pass
            """
            self._train_accuracy = tf.metrics.accuracy(tf.reshape(self._label, shape=[self._actual_batch_size, -1]), tf.reshape(self._predict, shape=[self._actual_batch_size, -1]), name="train_accuracy")
            self._train_recall = tf.metrics.recall(tf.reshape(self._label, shape=[self._actual_batch_size, -1]), tf.reshape(self._predict, shape=[self._actual_batch_size, -1]), name="train_recall")
            self._train_f1_score = 2 * self._train_accuracy[0] * self._train_recall[0] / (self._train_accuracy[0] + self._train_recall[0])
            self._validation_accuracy = tf.metrics.accuracy(tf.reshape(self._label, shape=[self._actual_batch_size, -1]), tf.reshape(self._predict, shape=[self._actual_batch_size, -1]), name="validation_accuracy")
            self._validation_recall = tf.metrics.recall(tf.reshape(self._label, shape=[self._actual_batch_size, -1]), tf.reshape(self._predict, shape=[self._actual_batch_size, -1]), name="validation_recall")
            self._validation_f1_score = 2 * self._validation_accuracy[0] * self._validation_recall[0] / (self._validation_accuracy[0] + self._validation_recall[0])
            print(self._next_element[2].get_shape())
            next_element = tf.reshape(self._next_element[2], shape=[self._actual_batch_size, self._labels_num, -1])
            self._train_accuracy = [tf.metrics.accuracy(name="train_acc"+str(i),labels=tf.reshape(next_element[:][i], shape=[self._actual_batch_size, -1]), predictions=tf.reshape(self._predict[:][i], shape=[self._actual_batch_size, -1])) for i in range(self._labels_num)]
            self._train_recall = [tf.metrics.recall(name="train_recall"+str(i),labels=tf.reshape(next_element[:][i], shape=[self._actual_batch_size, -1]), predictions=tf.reshape(self._predict[:][i], shape=[self._actual_batch_size, -1])) for i in range(self._labels_num)]
            self._train_f1_score = tf.stack([2 * self._train_accuracy[i][0] * self._train_recall[i][0] for i in range(self._labels_num)])
            self._train_f1_score = tf.reduce_mean(self._train_f1_score, axis=0)
            """

    def _create_loss(self):
        with tf.name_scope("create_loss"):
            length = self._labels_num
            w = [tf.nn.embedding_lookup(self.weights[i], tf.argmax(self._label[:, i, :], axis=1)) for i in range(length)]
            self._loss_ = [tf.losses.softmax_cross_entropy(onehot_labels=self._label[:, i, :], logits=self._logits[i], weights=w[i]) for i in range(length)]
            self._loss = tf.stack(self._loss_)
            self._total_loss = tf.reduce_mean(self._loss, axis=0)

    def _create_optimizer(self):
        with tf.name_scope("create_optimizer"):
            self._optimizer = tf.train.AdagradOptimizer(learning_rate=0.5).minimize(self._loss, global_step=self.global_step)
            self._all_optimizer = [tf.train.AdagradOptimizer(learning_rate=0.5).minimize(self._loss_[i], global_step=self.global_step) for i in range(self._labels_num)]

    def _create_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar('loss', self._total_loss)
            tf.summary.histogram('histogram loss', self._total_loss)
            self._summary_op = tf.summary.merge_all()

    def build(self):
        self.load_data()
        self._create_embedding()
        self._create_bilstm()
        self._create_output()
        self._create_metrics()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()

    def validation(self, sess):
        f1 = 0
        iteration = 0
        samples = 0
        average_f1 = 0
        sess.run(self._validation_iterator_initializer)
        while True:
            try:
                feature, len, label = sess.run(self._validation_next)
                predict, actual_batch_size, logits = sess.run(
                    [self._predict, self._actual_batch_size, self._logits],
                    feed_dict={self._keep_prob: 1.0, self._feature: feature, self._feature_length: len, self._label: label}
                )

                lab, res = sess.run([tf.argmax(label, axis=2) - 2, tf.argmax(predict, axis=2) - 2])
                for l1, l2 in zip(res, lab):
                    f1 += f1_score(l2, l1, average="macro")
                    print(l1, l2)
                iteration += 1
                samples += actual_batch_size
                average_f1 = f1 / samples
                print("average_f1 is {}".format(average_f1))
            except tf.errors.OutOfRangeError:
                print("验证集运行完毕，平均f1为: {}".format(average_f1))
                break

    def train(self, epoches=10):
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")

        self.build()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            writer = tf.summary.FileWriter('graphs/ai_challenger/learning_rate' + str(self._learning_rate), self.graph)
            initial_step = self.global_step.eval()
            print("initial_step is {}".format(initial_step))
            total_loss = 0.0
            iteration = 0
            train_next = self._train_iterator.get_next()
            max_loss_indice = None
            for i in range(initial_step, initial_step+epoches):
                sess.run(self._train_iterator_initializer)
                while True:
                    try:
                        feature, len, label = sess.run(train_next)
                        if iteration < 10000 or not max_loss_indice:
                            _, loss, summary, _loss = sess.run(
                                [self._optimizer, self._total_loss, self._summary_op, self._loss_],
                                feed_dict={
                                    self._keep_prob: 0.5, self._feature: feature, self._feature_length: len, self._label: label, self._learning_rate: 0.5
                                }
                            )
                        else:
                            print("最大loss的是{}".format(i))
                            _, loss, summary, _loss, max_loss_indice = sess.run(
                                [self._all_optimizer[max_loss_indice], self._total_loss, self._summary_op, self._loss_, tf.argmax(self._loss, axis=0)],
                                feed_dict={
                                    self._keep_prob: 0.5, self._feature: feature, self._feature_length: len, self._label: label, self._learning_rate: 0.5
                                }

                            )

                        print(_loss)
                        total_loss += loss
                        iteration = iteration + 1
                        average_loss = total_loss/iteration
                        print("average_loss is {}".format(average_loss))
                        writer.add_summary(summary, global_step=i)
                        if iteration % 400 == 0:
                            saver.save(sess, save_path="checkpoint/moon", global_step=self.global_step)
                            self.validation(sess)

                    except tf.errors.OutOfRangeError:
                        saver.save(sess, save_path="checkpoint/moon", global_step=self.global_step)
                        break

    def test(self):
        self.build()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("no model!")
                exit(0)
            test_next = self._test_iterator.get_next()
            sess.run(self._test_iterator)
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
    ml = MoonLight()
    ml.train(50000)
