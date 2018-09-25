# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf
from tensorflow.contrib import *
from tensorflow.contrib import data
import pandas as pd


class DataFiles:
    _train_file_names = ["./data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv"]
    _validation_file_names = ["./data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv"]
    _test_file_names = ["./data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv"]
    _dict_file = "./data/words.dict"

    _record_defaults = [tf.constant([0], dtype=tf.int32), tf.constant([], dtype=tf.string), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32)]


def build_vocab():
    import os
    if os.path.exists(DataFiles._dict_file):
        print("字典文件已经存在，不必再生成, 如果你想重新生成，请先手动删除该文件：{}".format(DataFiles._dict_file))
        return

    print("字典文件还没有构建，正在为您构建字典文件，请稍等......")
    files = DataFiles._train_file_names + DataFiles._validation_file_names + DataFiles._test_file_names
    dataset = data.CsvDataset(files, DataFiles._record_defaults, header=True)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    words = set()
    lines_count = 0
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        while True:
            try:
                line = sess.run(next_element)
                line = str(line[1], encoding="utf-8")
                line = line.strip("\"\t \r")
                for i in line:
                    i = i.strip("\"\t \r\n")
                    if i.__len__() == 0:
                        continue
                    words.add(i)
                lines_count = lines_count + 1
                print("\r已读取{}行".format(lines_count), end="")
            except tf.errors.OutOfRangeError as e:
                break
            except Exception as e:
                print(e)
        tf.assert_equal(lines_count, 135000)

        sess.run(tf.write_file(filename=DataFiles._dict_file, contents="\n".join(words)))

        print("字符数一共有:{} ".format(words.__len__()))
        print("一共有{}行数据".format(lines_count))
        print("字典文件构建完毕~~~")


class LookMan(object):

    _box = set()
    _table = {}
    _num_oov = -1

    def __init__(self, dict_file, num_oov_buckets=1):
        self._num_oov = num_oov_buckets
        with open(dict_file, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip("\n")
                self._box.add(line)

        k = 0
        for i in self._box:
            if k == num_oov_buckets:
                k = k + 1
            self._table[i] = k
            k = k + 1

    def lookup(self, key):
        if key not in self._table:
            return self._num_oov
        return self._table[key]

    def size(self):
        return self._table.__len__() + 1



class MoonLight(object):
    _train_file_names = DataFiles._train_file_names
    _validation_file_names = DataFiles._validation_file_names
    _test_file_names = DataFiles._validation_file_names

    _batch_size = 32
    _table = None
    _vocab_size = 0

    _train_iterator = None
    _train_batch = None

    _embedding_dimension = 50
    _lstm_unit = 256
    _lstm_layers = 2
    _keep_prob = None
    _attention_length = 2
    _learning_rate = 0.01

    def __init__(self, embedding_dimension=100):
        self._embedding_dimension = embedding_dimension
        self._keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

    def load(self):
        build_vocab()
        self._table = LookMan(DataFiles._dict_file, num_oov_buckets=1)
        self._vocab_size = self._table.size()
        print("字典文件加载完成，字典大小为: {}".format(self._vocab_size))

    def _get_has_label_data(self, file_names):
        for file in file_names:
            lines = pd.read_csv(file, delimiter=",", skiprows=1)
            for i in range(len(lines)):
                sentence = lines.iloc[i, 1].strip("\"")
                ids = [self._table.lookup(i) for i in sentence]
                labels = []
                for j in range(2, 22, 1):
                    code = lines.iloc[1, j]
                    # 标注数据可能取值为-2, -1, 0, 1, 为了onehot编码，归一化为0， 1， 2， 3
                    code = code + 2
                    labels.append(code)
                #print(labels)
                yield ids, [len(ids)], labels

    def _get_no_label_data(self, file_names):
        for file in file_names:
            lines = pd.read_csv(file, delimiter=",", skiprows=1)
            for i in range(len(lines)):
                sentence = lines.iloc[i, 1].strip("\"")
                ids = [self._table.lookup(i) for i in sentence]
                yield ids, [len(ids)]

    def _get_validation_data(self):
        file_names = self._validation_file_names
        yield from self._get_has_label_data(file_names)

    def _gen_train_data(self):
        file_names = self._train_file_names
        yield from self._get_has_label_data(file_names)

    def _get_test_data(self):
        file_names = self._test_file_names
        yield from self._get_no_label_data(file_names)

    def _load_train_data(self):
        train_dataset = tf.data.Dataset.from_generator(self._gen_train_data, (tf.int64, tf.int64, tf.int64), ([None], [None], [20]))
        train_dataset = train_dataset.padded_batch(self._batch_size, padded_shapes=([None], [None], [20]),
                                                   padding_values=(tf.constant(1, dtype=tf.int64), tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64)))
        train_dataset = train_dataset.map(lambda *x: (x[0], x[1], tf.one_hot(indices=x[2], depth=4, dtype=tf.float32)))
        self._train_iterator = train_dataset.make_initializable_iterator()
        self._train_feature, self._train_feature_len, self._train_labels = self._train_iterator.get_next()

    def _create_embedding(self):
        with tf.name_scope("create_embedding"):
            self._embedding_matrix = tf.get_variable(name="embedding_matrix", shape=[self._vocab_size, self._embedding_dimension],
                                              initializer=tf.variance_scaling_initializer)
            self._embedding = tf.nn.embedding_lookup(self._embedding_matrix, self._train_feature, name="embedding")

    def _create_bilstm(self):

        def lstm_cell(lstm_unit):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_unit, activation=tf.nn.relu)
            cell = rnn.AttentionCellWrapper(cell=cell, attn_length=self._attention_length)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=self._keep_prob)
            return cell

        with tf.name_scope("create_bilstm"):
            stack_fw_lstm = [lstm_cell(lstm_unit=self._lstm_unit) for _ in range(self._lstm_layers)]
            initial_state_fw = [stack_fw_lstm_unit.zero_state(self._batch_size, tf.float32) for stack_fw_lstm_unit in stack_fw_lstm]
            stack_bw_lstm = [lstm_cell(lstm_unit=self._lstm_unit) for _ in range(self._lstm_layers)]
            initial_state_bw = [stack_bw_lstm_unit.zero_state(self._batch_size, tf.float32) for stack_bw_lstm_unit in stack_bw_lstm]
            self._sentence_encoder_output, self._fw_state, self._bw_state = \
                rnn.stack_bidirectional_dynamic_rnn(cells_fw=stack_fw_lstm, cells_bw=stack_bw_lstm, initial_states_fw=initial_state_fw,\
                                                    initial_states_bw=initial_state_bw, inputs=self._embedding)

    def _create_loss(self):
        with tf.name_scope("create_loss"):
            length = self._train_labels.get_shape()[1].value
            output_dimension = self._train_labels.get_shape()[2].value
            input = tf.concat([self._bw_state[-1][-1], self._fw_state[-1][-1]], 1)
            logits = tf.layers.dense(inputs=input, units=output_dimension, kernel_initializer=tf.truncated_normal_initializer(seed=29))
            self._loss = [tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self._train_labels[:, i, :]) for i in range(length)]
            self._total_loss = tf.reduce_sum(self._loss, axis=0)
            self._total_loss = tf.reduce_sum(self._total_loss, 1)
            self._total_loss = tf.reduce_mean(self._total_loss, axis=0)

    def _create_optimizer(self):
        with tf.name_scope("create_optimizer"):
           self._optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate).minimize(self._total_loss)

    def _create_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar('loss', self._total_loss)
            tf.summary.histogram('histogram loss', self._total_loss)
            self._summary_op = tf.summary.merge_all()

    def build(self, sess):
        self.load()
        self._load_train_data()
        self._create_embedding()
        self._create_bilstm()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()
        sess.run(ml._train_iterator.initializer)


if __name__ == "__main__":
    tf.Graph()
    ml = MoonLight()
    with tf.Session() as sess:
        ml.build(sess)
        while True:
            try:
                sess.run(tf.global_variables_initializer())
                _, loss = sess.run([ml._optimizer, ml._total_loss], feed_dict={ml._keep_prob: 0.6})
                print("loss is {}".format(loss))
            except tf.errors.OutOfRangeError:
                break
