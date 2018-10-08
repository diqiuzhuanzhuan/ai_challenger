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
    _lstm_unit = 256
    _lstm_layers = 3
    _keep_prob = None
    _attention_length = 40
    _learning_rate = 0.01
    #训练数据中标签总共有20个
    _labels_num = 20
    _output_dimension = 4

    def __init__(self, embedding_dimension=128):
        self._embedding_dimension = embedding_dimension
        self._keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        self.global_step = tf.get_variable("global_step", initializer=tf.constant(0), trainable=False)
        self._checkpoint_path = os.path.dirname('checkpoint/checkpoint')
        self._batch_size = tf.placeholder(name="batch_size", shape=[], dtype=tf.int64)
        self._actual_batch_size = None
        self._batch_size = 1
        self._data = Data(self._batch_size)
        self.weights = None

    def load_data(self):
        self._train_iterator, self._validation_iterator, self._test_iterator, self._next_element = self._data.load_data()
        self._actual_batch_size = tf.shape(self._next_element[0])[0]
        self.weights = tf.constant(self._data.weights)

    def _create_embedding(self):
        with tf.name_scope("create_embedding"):
            self._embedding_matrix = tf.get_variable(name="embedding_matrix", shape=[self._data._vocab_size, self._embedding_dimension],
                                              initializer=tf.variance_scaling_initializer)

            self._embedding = tf.nn.embedding_lookup(self._embedding_matrix, self._next_element[0], name="embedding")

    def _create_bilstm(self):

        def lstm_cell(lstm_unit):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_unit)
            #cell = rnn.AttentionCellWrapper(cell=cell, attn_length=self._attention_length, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=self._keep_prob)
            return cell

        with tf.name_scope("create_bilstm"):
            stack_fw_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_unit=self._lstm_unit) for _ in range(self._lstm_layers)])
            initial_state_fw = stack_fw_lstm.zero_state(tf.to_int32(self._actual_batch_size), tf.float32)
            stack_bw_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_unit=self._lstm_unit) for _ in range(self._lstm_layers)])
            initial_state_bw = stack_bw_lstm.zero_state(tf.to_int32(self._actual_batch_size), tf.float32)
            self._sentence_encoder_output, self._fw_state, self._bw_state = \
                rnn.stack_bidirectional_dynamic_rnn(cells_fw=[stack_fw_lstm], cells_bw=[stack_bw_lstm], initial_states_fw=[initial_state_fw],\
                                                    initial_states_bw=[initial_state_bw], inputs=self._embedding, sequence_length=tf.reshape(self._next_element[1], [-1]), parallel_iterations=128)

    def _create_output(self):
        with tf.name_scope("create_output"):
            length = self._labels_num
            output_dimension = self._output_dimension
            input = self._sentence_encoder_output[:, -1, :]
            self._logits = [
                tf.layers.dense(inputs=input, units=output_dimension, kernel_initializer=tf.truncated_normal_initializer(seed=i), activation=tf.nn.leaky_relu)
                for i in range(length)
            ]
            self._predict = tf.stack([tf.nn.softmax(logits=self._logits[i]) for i in range(length)])
            self._predict = tf.argmax(self._predict, axis=2)
            self._predict = tf.one_hot(self._predict, depth=output_dimension, dtype=tf.int64)
            self._predict = tf.transpose(self._predict, [1, 0, 2])

    def _create_metrics(self):
            self._train_accuracy = tf.metrics.accuracy(tf.reshape(self._next_element[2], shape=[-1]), tf.reshape(self._predict, shape=[-1]), name="train_accuracy")
            self._train_recall = tf.metrics.recall(tf.reshape(self._next_element[2], shape=[-1]), tf.reshape(self._predict, shape=[-1]), name="train_recall")
            self._train_f1_score = 2 * self._train_accuracy[0] * self._train_recall[0] / (self._train_accuracy[0] + self._train_recall[0])
            self._validation_accuracy = tf.metrics.accuracy(tf.reshape(self._next_element[2], shape=[-1]), tf.reshape(self._predict, shape=[-1]), name="validation_accuracy")
            self._validation_recall = tf.metrics.recall(tf.reshape(self._next_element[2], shape=[-1]), tf.reshape(self._predict, shape=[-1]), name="validation_recall")
            self._validation_f1_score = 2 * self._validation_accuracy[0] * self._validation_recall[0] / (self._validation_accuracy[0] + self._validation_recall[0])

    def _create_loss(self):
        with tf.name_scope("create_loss"):
            length = self._labels_num
            w = [tf.nn.embedding_lookup(self.weights[i], tf.argmax(self._next_element[2][:, i, :], axis=1)) for i in range(length)]
            self._loss = tf.stack([tf.losses.softmax_cross_entropy(onehot_labels=self._next_element[2][:, i, :], logits=self._logits[i], weights=w[i]) for i in range(length)])
            self._total_loss = tf.reduce_mean(self._loss, axis=0)

    def _create_optimizer(self):
        with tf.name_scope("create_optimizer"):
           self._optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate).minimize(self._total_loss, global_step=self.global_step)

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

    def train(self, epoches=10):
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.build()
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            writer = tf.summary.FileWriter('graphs/ai_challenger/learning_rate' + str(self._learning_rate), sess.graph)
            initial_step = self.global_step.eval()
            print("initial_step is {}".format(initial_step))
            total_loss = 0.0
            iteration = 0
            for i in range(initial_step, initial_step+epoches):
                sess.run(self._train_iterator)
                while True:
                    try:
                        _, loss, summary, f1_score, accuracy, recall, predict, labels = sess.run(
                            [self._optimizer, self._total_loss, self._summary_op, self._train_f1_score, self._train_accuracy, self._train_recall, self._predict, self._next_element[2]],
                            feed_dict={self._keep_prob: 0.4}
                        )
                        lab = sess.run(tf.argmax(labels, axis=2) - 2)
                        res = sess.run(tf.argmax(predict, axis=2) - 2)
                        for l1, l2 in zip(res, lab):
                            print("{}, {}".format(l1, l2))
                        total_loss += loss
                        iteration = iteration + 1
                        average_loss = total_loss/iteration
                        print("average_loss is {}".format(average_loss))
                        writer.add_summary(summary, global_step=i)
                        print("train f1_score is {}, accuracy is {}, recall is {}".format(f1_score, accuracy[0], recall[0]))

                    except tf.errors.OutOfRangeError:
                        break
                saver.save(sess, save_path="checkpoint/moon")
                sess.run(self._validation_iterator)
                while True:
                    try:
                        f1_score, accuracy, recall, predict = sess.run(
                            [self._validation_f1_score, self._validation_accuracy, self._validation_recall, self._predict],
                            feed_dict={self._keep_prob: 1.0})
                        print("validation f1_score is {}, accurancy is {}, recall is {}".format(f1_score, accuracy[0], recall[0]))
                        res = sess.run(tf.argmax(predict, axis=2) - 2)
                    except tf.errors.OutOfRangeError:
                        break

                sess.run(self._test_iterator)
                while True:
                    try:
                        predict = sess.run(
                            self._predict, feed_dict={self._keep_prob: 1.0}
                         )
                        res = sess.run(tf.argmax(predict, axis=2, output_type=tf.int64) - 2)
                        self._data.feed_output(res)
                    except tf.errors.OutOfRangeError:
                        break

                self._data.persist()

    def test(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.build()
#            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("no model!")
                exit(0)
            sess.run(self._test_iterator)
            while True:
                try:
                    predict = sess.run(
                        self._predict, feed_dict={self._keep_prob: 1.0}
                    )
                    res = sess.run(tf.argmax(predict, axis=2) - 2)
                    self._data.feed_output(res)

                except tf.errors.OutOfRangeError:
                    break
            self._data.persist()


if __name__ == "__main__":
    ml = MoonLight()
    ml.train(2)
