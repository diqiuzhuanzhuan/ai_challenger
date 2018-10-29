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
import numpy as np
from tensorflow.contrib import rnn

os.environ['CUDA_VISIBLE_DEVICES']='0, 1'


class MoonLight(object):
    _train_file_names = DataFiles._train_file_names
    _validation_file_names = DataFiles._validation_file_names
    _test_file_names = DataFiles._test_file_names

    _table = None


    _embedding_dimension = 50
    _lstm_unit = 128
    _lstm_layers = 1
    _keep_prob = None
    _attention_length = 40
    #训练数据中标签总共有20个
    _labels_num = 20
    _output_dimension = 4

    def __init__(self, batch_size, learning_rate, embedding_size, vocab_size, weight, labes_num=20, output_dimension=4, next_element=None):
        self._embedding_size = embedding_size
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        self.global_step = tf.get_variable("global_step", initializer=tf.constant(0), trainable=False)
        self._checkpoint_path = os.path.dirname('checkpoint/checkpoint')
        self._batch_size = tf.placeholder(name="batch_size", shape=[], dtype=tf.int64)
        self._learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 1000, 0.96, staircase=True)
        self.weight = weight
        self._feature = next_element[0]
        self._feature_length = next_element[1]
        self._label = next_element[2]
        self._actual_batch_size = tf.shape(self._feature)[0]

        with tf.device("/cpu:0"), tf.name_scope("create_embedding"):
            self._embedding_matrix = tf.get_variable(name="embedding_matrix", shape=[self._vocab_size,
                                                                                     self._embedding_dimension], initializer=tf.variance_scaling_initializer)
            self._embedding = tf.nn.embedding_lookup(self._embedding_matrix, self._feature, name="embedding")

        with tf.name_scope("create_bilstm"):

            def lstm_cell(lstm_unit=256):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_unit)
                cell = rnn.AttentionCellWrapper(cell=cell, attn_length=self._attention_length, state_is_tuple=True)
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
                tf.layers.dense(inputs=self._logits[i], units=output_dimension, kernel_initializer=tf.truncated_normal_initializer(seed=i*10, stddev=0.01, mean=0), activation=None)
                for i in range(length)
            ]
            self.predict = tf.stack(self._logits)
            self.predict = tf.argmax(self.predict, axis=2)
            self.predict = tf.transpose(self.predict, [1, 0], name="predict")

        with tf.name_scope("create_loss"):
            length = self._labels_num
            w = [tf.nn.embedding_lookup(self.weight[i], tf.argmax(self._label[:, i, :], axis=1), name="embedding_lookup"+str(i)) for i in range(length)]
            self._loss_ = [tf.losses.softmax_cross_entropy(onehot_labels=self._label[:, i, :], logits=self._logits[i], weights=w[i]) for i in range(length)]
            self._loss = tf.stack(self._loss_)
            self._total_loss = tf.reduce_mean(self._loss, axis=0)

        with tf.name_scope("create_optimizer"):
            #self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._loss, global_step=self.global_step)
            self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._learning_rate)
            self._grads_total = self._optimizer.compute_gradients(self._loss)
            self._grads_distribution = [self._optimizer.compute_gradients(self._loss_[i]) for i in range(self._labels_num)]
            self._train_total = self._optimizer.apply_gradients(self._grads_total, global_step=self.global_step)
            self._train_distribution = [self._optimizer.apply_gradients(self._grads_distribution[i], global_step=self.global_step) for i in range(self._labels_num)]

        with tf.name_scope("summary"):
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


tf.flags.DEFINE_integer("batch_size", 32, "batch_size")
tf.flags.DEFINE_integer("labels_num", 20, "class num of task")
tf.flags.DEFINE_integer("output_dimension", 4, "output dimension")
tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")

tf.flags.DEFINE_integer("step_bypass_validation", 2000, "how many steps was run before we start to run the first validation?")
tf.flags.DEFINE_integer("step_validation", 3000, "validation run every many steps")

tf.flags.DEFINE_string("summary_path", "./graphs/", "summary path")
tf.flags.DEFINE_string("checkpoint_path", "./checkpoint/moon/", "Checkpoint file path for saving")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def load_data():
    data = Data(batch_size=FLAGS.batch_size)
    data.load_data()
    return data


def main():
    tf.reset_default_graph()
    data = load_data()
    vocab_size = data.get_vocab_size()
    weight = data.weights
    train_iterator_initializer = data._train_iterator_initializer
    validation_iterator_initializer = data._validation_iterator_initializer
    test_iterator_initializer = data._test_iterator_initializer
    handle = data.handle
    next_element = data.next_element

    model = MoonLight(batch_size=64, learning_rate=0.8, embedding_size=256, vocab_size=vocab_size, weight=weight, labes_num=FLAGS.labels_num,
                      output_dimension=FLAGS.output_dimension, next_element=next_element)

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        writer = tf.summary.FileWriter('graphs/bilstm/learning_rate' + str(model._learning_rate))
        if not os.path.exists(FLAGS.checkpoint_path):
            os.makedirs(FLAGS.checkpoint_path)
        checkpoints_prefix = os.path.join(FLAGS.checkpoint_path, "bilstm")
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        train_handle = sess.run(data._train_iterator.string_handle())
        validation_handle = sess.run(data._validation_iterator.string_handle())
        test_handle = sess.run(data._test_iterator.string_handle())

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("正在从{}加载模型".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

        def validation_step():
            feed_dict = {
                model._keep_prob: 1.0, handle: validation_handle
            }

            loss, actual_batch_size, lab, res = sess.run(
                [model._total_loss, model._actual_batch_size, tf.argmax(model._label, axis=2) - 2, model.predict - 2],
                feed_dict=feed_dict
            )
            return loss, actual_batch_size, lab, res

        def validation():
            f1 = 0
            samples = 0
            total_validation_loss = 0
            total_time = 0
            sess.run(validation_iterator_initializer)
            iteration = 0
            while True:
                try:
                    t1 = time.time()
                    loss, actual_batch_size, lab, res = validation_step()
                    f1 += np.sum(list(map(lambda x: f1_score(x[0], x[1], average="macro"), zip(lab.tolist(), res.tolist()))))
                    samples += actual_batch_size
                    iteration += 1
                    total_validation_loss += loss
                    delta_t = time.time() - t1
                    total_time += delta_t
                    print("当前f1为:{}, loss 为{}, 花费{}秒".format(f1 / samples, loss, delta_t))

                except tf.errors.OutOfRangeError:
                    print("平均f1为:{}, 平均loss为{}, 总耗时{}秒".format(f1 / samples, total_validation_loss / iteration, total_time))
                    break
            return f1 / samples

        def test_step(test_handle):
            feed_dict = {
                model._keep_prob: 1.0, handle: test_handle
            }
            res = sess.run(
                model.predict - 2,
                feed_dict=feed_dict
            )
            data.feed_output(res)

        def test():
            global_step = sess.run(model.global_step)
            sess.run(test_iterator_initializer)
            while True:
                try:
                    test_step(test_handle)
                except tf.errors.OutOfRangeError:
                    data.persist(filename="result_{}.csv".format(global_step))
                    print("测试结果已经保存, result_{}.csv".format(global_step))
                    break

        def train_step():
            feed_dict = {
                model._keep_prob: 1.0, handle: train_handle
            }
            _, loss, global_step, summary_op, actual_batch_size = sess.run(
                [model._train_total, model._total_loss, model.global_step, model._summary_op, model._actual_batch_size],
                feed_dict=feed_dict
            )
            writer.add_summary(summary_op, global_step=global_step)
            return loss, global_step, actual_batch_size

        def train():
            while True:
                try:
                    t1 = time.time()
                    loss, step, actual_batch_size = train_step()
                    delta_t = time.time() - t1
                    print("training: step is {}, loss is {}, cost {} 秒".format(step, loss, delta_t))
                    if step > FLAGS.step_bypass_validation and step % FLAGS.step_validation == 0:
                        saver.save(sess, save_path=checkpoints_prefix, global_step=step)
                        average_f1 = validation()
                        if average_f1 > 0.68:
                            test()

                    if step % FLAGS.step_validation == 0:
                        saver.save(sess, save_path=checkpoints_prefix, global_step=step)

                except tf.errors.OutOfRangeError:
                    break

        for i in range(FLAGS.num_epochs):
            sess.run(train_iterator_initializer)
            print("第{}个epoch".format(i))
            train()


if __name__ == "__main__":
    Config._use_lemma = False
    main()
