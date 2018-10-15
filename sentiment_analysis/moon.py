# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import time

import tensorflow as tf
import os
from sea import DataFiles, Data, Config
from sklearn.metrics import f1_score

os.environ['CUDA_VISIBLE_DEVICES']='1'


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

    def __init__(self, learning_rate=0.8, embedding_dimension=256):
        self._embedding_dimension = embedding_dimension
        self._keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        self._feature = tf.placeholder(dtype=tf.int64, shape=[None, None], name="feature")
        self._feature_length = tf.placeholder(dtype=tf.int64, shape=[None, None], name="feature_length")
        self._label = tf.placeholder(dtype=tf.int64, shape=[None, None, None], name="label")
        self.global_step = tf.get_variable("global_step", initializer=tf.constant(0), trainable=False)
        self._checkpoint_path = os.path.dirname('checkpoint/checkpoint')
        self._batch_size = tf.placeholder(name="batch_size", shape=[], dtype=tf.int64)
        self._learning_rate = tf.placeholder(name="learning_rate", dtype=tf.float32)
        self._learning_rate = learning_rate
        self._learning_rate = tf.train.exponential_decay(1e-1, self.global_step, 1000, 0.96, staircase=True)
        self._actual_batch_size = None
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
        with tf.device("/cpu:0"), tf.name_scope("create_embedding"):
            self._embedding_matrix = tf.get_variable(name="embedding_matrix", shape=[self._data.get_vocab_size(), self._embedding_dimension],
                                              initializer=tf.variance_scaling_initializer)

            self._embedding = tf.nn.embedding_lookup(self._embedding_matrix, self._feature, name="embedding")

    def _create_bilstm(self):

        with tf.name_scope("create_bilstm"):
            def lstm_cell(lstm_unit):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_unit)
#            cell = rnn.AttentionCellWrapper(cell=cell, attn_length=self._attention_length, state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=self._keep_prob)
                return cell

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
            self._input = tf.layers.dense(inputs=self._input, units=256, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu)

            self._logits = [
                tf.layers.dense(inputs=self._input, units=128, kernel_initializer=tf.truncated_normal_initializer(seed=i, stddev=0.1), activation=tf.nn.relu)
                for i in range(length)
            ]
            self._logits = [
                tf.layers.dense(inputs=self._logits[i], units=output_dimension, kernel_initializer=tf.truncated_normal_initializer(seed=i*10, stddev=0.01, mean=0), activation=tf.nn.sigmoid)
                for i in range(length)
            ]
            self._predict = tf.stack(self._logits)
            self._predict = tf.argmax(self._predict, axis=2)
            self._predict = tf.one_hot(self._predict, depth=output_dimension, dtype=tf.int64)
            self._predict = tf.transpose(self._predict, [1, 0, 2])

    def _create_loss(self):
        with tf.name_scope("create_loss"):
            length = self._labels_num
            w = [tf.nn.embedding_lookup(self.weights[i], tf.argmax(self._label[:, i, :], axis=1), name="embedding_lookup"+str(i)) for i in range(length)]
            self._loss_ = [tf.losses.softmax_cross_entropy(onehot_labels=self._label[:, i, :], logits=self._logits[i], weights=w[i]) for i in range(length)]
            self._loss = tf.stack(self._loss_)
            self._total_loss = tf.reduce_mean(self._loss, axis=0)

    def _create_optimizer(self):
        with tf.name_scope("create_optimizer"):
            #self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._loss, global_step=self.global_step)
            self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._learning_rate)
            self._grads_total = self._optimizer.compute_gradients(self._loss)
            self._grads_distribution = [self._optimizer.compute_gradients(self._loss_[i]) for i in range(self._labels_num)]
            self._train_total = self._optimizer.apply_gradients(self._grads_total, global_step=self.global_step)
            self._train_distribution = [self._optimizer.apply_gradients(self._grads_distribution[i], global_step=self.global_step) for i in range(self._labels_num)]

    def _create_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar('loss', self._total_loss)
            [tf.summary.scalar('loss ['+str(i)+']', self._loss_[i]) for i in range(self._labels_num)]
            [tf.summary.histogram('histogram loss[' +str(i) + ']', self._loss_[i]) for i in range(self._labels_num)]
            tf.summary.histogram('histogram loss', self._total_loss)
            self._summary_op = tf.summary.merge_all()

    def build(self):
        self.load_data()
        self._create_embedding()
        self._create_bilstm()
        self._create_output()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()

    def validation(self, sess):
        f1 = 0
        samples = 0
        sess.run(self._validation_iterator_initializer, feed_dict={self._batch_size: 512})
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

        self.build()
        saver = tf.train.Saver(sharded=True)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                print("正在从{}加载模型".format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('graphs/ai_challenger/learning_rate' + str(self._learning_rate), self.graph)
            initial_step = self.global_step.eval()
            print("initial_step is {}".format(initial_step))
            total_loss = 0.0
            iteration = 0
            train_next = self._train_iterator.get_next()
            max_loss_indice = None
            global_step = initial_step
            total_time = 0
            for i in range(initial_step, initial_step+epoches):
                sess.run(self._train_iterator_initializer, feed_dict={self._batch_size: 32})
                while True:
                    try:
                        delta_t = time.time()
                        feature, len, label = sess.run(train_next)
                        if global_step < 10000 or not max_loss_indice:
                            _, loss, summary, global_step = sess.run(
                                [self._train_total, self._total_loss, self._summary_op, self.global_step],
                                feed_dict={
                                    self._keep_prob: 1.0, self._feature: feature, self._feature_length: len, self._label: label
                                }
                            )
                        else:
                            _, _, loss, summary, max_loss_indice, global_step = sess.run(
                                [self._train_distribution[max_loss_indice], self._train_total, self._total_loss, self._summary_op, tf.argmax(self._loss, axis=0), self.global_step],
                                feed_dict={
                                    self._keep_prob: 0.7, self._feature: feature, self._feature_length: len, self._label: label
                                }

                            )
                        total_loss += loss
                        iteration = iteration + 1
                        average_loss = total_loss/iteration
                        writer.add_summary(summary, global_step=global_step)
                        total_time += time.time() - delta_t
                        print("iteration is {}, current_loss is {}, average_loss is {}, total_time is {}, cost time {}sec/batch".format(iteration, loss, average_loss, total_time, total_time/iteration))
                        if iteration % 1000 == 0:
                            saver.save(sess, save_path="checkpoint/moon", global_step=self.global_step)
                            self.validation(sess)
                        if global_step % 30000 == 0:
                            self._test(sess, global_step)

                    except tf.errors.OutOfRangeError:
                        saver.save(sess, save_path="checkpoint/moon", global_step=self.global_step)
                        break

    def _test(self, sess, global_step):
        test_next = self._test_iterator.get_next()
        sess.run(self._test_iterator_initializer, feed_dict={self._batch_size: 256})
        while True:
            try:
                feature, len, label = sess.run(test_next)
                res = sess.run(
                    tf.argmax(self._predict, axis=2, output_type=tf.int64)-2,
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
            sess.run(self._test_iterator_initializer, feed_dict={self._batch_size: 256})
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
    ml = MoonLight()
    ml.train(50000)
