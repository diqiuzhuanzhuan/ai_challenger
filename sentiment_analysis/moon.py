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
import os
import urllib

os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2'


class DataFiles:
    _train_file_names = ["./data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv"]
    _train_file_url = ["http://www.diqiuzhuanzhuan.com/download/344/"]
    _validation_file_names = ["./data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv"]
    _validation_file_url = ["http://www.diqiuzhuanzhuan.com/download/346/"]
    _test_file_names = ["./data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv"]
    _test_file_url = ["http://www.diqiuzhuanzhuan.com/download/334/"]
    _dict_file = "./data/words.dict"

    _record_defaults = [tf.constant([0], dtype=tf.int32), tf.constant([], dtype=tf.string), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32),
                        tf.constant([0], dtype=tf.int32)]

    @classmethod
    def _prepare_data(cls):
        for url, file in zip(cls._train_file_url, cls._train_file_names):
            cls._download(url, file)
        for url, file in zip(cls._validation_file_url, cls._validation_file_names):
            cls._download(url, file)
        for url, file in zip(cls._test_file_url, cls._test_file_names):
            cls._download(url, file)

    @classmethod
    def _download(cls, url, file_name):
        if not os.path.exists(file_name):
            try:
                print("正在下载文件{}".format(file_name))
                dest, _ = urllib.request.urlretrieve(url=url, filename=file_name)
                print("{}下载完成".format(file_name))
            except Exception as e:
                os.remove(file_name)


def build_vocab():
    import os
    print("正在准备数据集")
    DataFiles._prepare_data()
    print("数据集已准备完毕")
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

    _batch_size = 64
    _table = None
    _vocab_size = 0

    _train_iterator = None
    _train_batch = None

    _embedding_dimension = 50
    _lstm_unit = 256
    _lstm_layers = 1
    _keep_prob = None
    _attention_length = 40
    _learning_rate = 0.01
    #训练数据中标签总共有20个
    _labels_num = 20

    def __init__(self, embedding_dimension=100):
        self._embedding_dimension = embedding_dimension
        self._keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        self.global_step = tf.get_variable("global_step", initializer=tf.constant(0), trainable=False)
        self._checkpoint_path = os.path.dirname('checkpoint/checkpoint')
        self._batch_size = tf.placeholder(name="batch_size", shape=[], dtype=tf.int64)

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
                for j in range(2, 2+self._labels_num, 1):
                    code = lines.iloc[i, j]
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

    def _gen_test_data(self):
        file_names = self._test_file_names
        yield from self._get_no_label_data(file_names)

    def _load_train_data(self):
        train_dataset = tf.data.Dataset.from_generator(self._gen_train_data, (tf.int64, tf.int64, tf.int64), ([None], [None], [self._labels_num]))
        train_dataset = train_dataset.padded_batch(self._batch_size, padded_shapes=([None], [None], [None]),
                                                   padding_values=(tf.constant(1, dtype=tf.int64), tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64)), drop_remainder=True)
        train_dataset = train_dataset.map(lambda *x: (x[0], x[1], tf.one_hot(indices=x[2], depth=4, dtype=tf.int64)))
        self._iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        self._next_element = self._iterator.get_next()
        self._train_iterator = self._iterator.make_initializer(train_dataset)

    def _load_validation_data(self):
        validation_dataset = tf.data.Dataset.from_generator(self._gen_train_data, (tf.int64, tf.int64, tf.int64), ([None], [None], [self._labels_num]))
        validation_dataset = validation_dataset.padded_batch(self._batch_size, padded_shapes=([None], [None], [self._labels_num]),
                                                   padding_values=(tf.constant(1, dtype=tf.int64), tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64)), drop_remainder=True)
        validation_dataset = validation_dataset.map(lambda *x: (x[0], x[1], tf.one_hot(indices=x[2], depth=4, dtype=tf.int64)))
        self._validation_iterator = self._iterator.make_initializer(validation_dataset)

    def _load_test_data(self):
        test_dataset = tf.data.Dataset.from_generator(self._gen_test_data, (tf.int64, tf.int64), ([None], [None]))
        test_dataset = test_dataset.padded_batch(self._batch_size, padded_shapes=([None], [None]),
                                                             padding_values=(tf.constant(1, dtype=tf.int64), tf.constant(0, dtype=tf.int64)))
        test_dataset = test_dataset.map(lambda *x: (x[0], x[1], tf.one_hot(indices=x[2], depth=4, dtype=tf.int64)))
        self._iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
        self._next_element = self._iterator.get_next()
        self._test_iterator = test_dataset.make_initializable_iterator()
        self._test_feature, self._test_feature_len = self._validation_iterator.get_next()

    def _create_embedding(self):
        with tf.name_scope("create_embedding"):
            self._embedding_matrix = tf.get_variable(name="embedding_matrix", shape=[self._vocab_size, self._embedding_dimension],
                                              initializer=tf.variance_scaling_initializer)

            self._embedding = tf.nn.embedding_lookup(self._embedding_matrix, self._next_element[0], name="embedding")

    def _create_bilstm(self):

        def lstm_cell(lstm_unit):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_unit)
           # cell = rnn.AttentionCellWrapper(cell=cell, attn_length=self._attention_length, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=self._keep_prob)
            return cell

        with tf.name_scope("create_bilstm"):
            stack_fw_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_unit=self._lstm_unit) for _ in range(self._lstm_layers)])
            initial_state_fw = stack_fw_lstm.zero_state(tf.to_int32(self._batch_size), tf.float32)
            stack_bw_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_unit=self._lstm_unit) for _ in range(self._lstm_layers)])
            initial_state_bw = stack_bw_lstm.zero_state(tf.to_int32(self._batch_size), tf.float32)
            self._sentence_encoder_output, self._fw_state, self._bw_state = \
                rnn.stack_bidirectional_dynamic_rnn(cells_fw=[stack_fw_lstm], cells_bw=[stack_bw_lstm], initial_states_fw=[initial_state_fw],\
                                                    initial_states_bw=[initial_state_bw], inputs=self._embedding)

    def _create_output(self):
        with tf.name_scope("create_output"):
            length = self._labels_num
            output_dimension = self._next_element[2].get_shape()[2].value
            input = self._sentence_encoder_output[:, -1, :]
            self._logits = [
                tf.layers.dense(inputs=input, units=output_dimension, kernel_initializer=tf.truncated_normal_initializer(seed=i), activation=tf.nn.relu)
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
            self._loss = tf.stack([tf.losses.softmax_cross_entropy(onehot_labels=self._next_element[2][:, i, :], logits=self._logits[i]) for i in range(length)])
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
        self.load()
        self._load_train_data()
        self._load_validation_data()
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
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoint/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            writer = tf.summary.FileWriter('graphs/ai_challenger/learning_rate' + str(self._learning_rate), sess.graph)
            initial_step = self.global_step.eval()
            print("initial_step is {}".format(initial_step))
            total_loss = 0.0
            iteration = 0
            for i in range(initial_step, initial_step+epoches):
                sess.run(self._train_iterator, feed_dict={self._batch_size: 64})
                while True:
                    try:
                        _, loss, summary, f1_score, accuracy, recall = sess.run(
                            [self._optimizer, self._total_loss, self._summary_op, self._train_f1_score, self._train_accuracy, self._train_recall],
                            feed_dict={self._keep_prob: 0.6, self._batch_size: 64}
                        )
                        total_loss += loss
                        iteration = iteration + 1
                        average_loss = total_loss/iteration
                        print("average_loss is {}".format(average_loss))
                        writer.add_summary(summary, global_step=i)
                        print("train f1_score is {}, accuracy is {}, recall is {}".format(f1_score, accuracy[0], recall[0]))

                    except tf.errors.OutOfRangeError:
                        break
                sess.run(self._validation_iterator, feed_dict={self._batch_size: 32})
                while True:
                    try:
                        f1_score, accuracy, recall = sess.run(
                            [self._validation_f1_score, self._validation_accuracy, self._validation_recall],
                            feed_dict={self._keep_prob: 1.0, self._batch_size: 32})
                        print("validation f1_score is {}, accurancy is {}, recall is {}".format(f1_score, accuracy[0], recall[0]))

                    except tf.errors.OutOfRangeError:
                        break

    def test(self):
        saver = tf.train.Saver()


if __name__ == "__main__":
    tf.Graph()
    ml = MoonLight()
    ml.train(10)
