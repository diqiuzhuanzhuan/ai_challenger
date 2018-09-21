# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf
from tensorflow.contrib import data
from tensorflow.contrib import lookup
import numpy as np
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
            word = f.readline().strip("\n")
            self._box.add(word)

        k = 0
        for i in self._box:
            if k == num_oov_buckets:
                k = k + 1
            self._table[i] = k
            k = k + 1

    def lookup(self, key):
        try:
            id = self._table[key]
        except KeyError as e:
            id = self._num_oov
        return id


class MoonLight(object):
    _train_file_names = DataFiles._train_file_names
    _validation_file_names = DataFiles._validation_file_names
    _test_file_names = DataFiles._validation_file_names

    _batch_size = 32
    _table = None

    _train_iterator = None
    _train_batch = None

    _embedding_dimension = 50

    def __init__(self, embedding_dimension=100):
        self._embedding_dimension = embedding_dimension

    def load(self):
        build_vocab()
        self._table = LookMan(DataFiles._dict_file, num_oov_buckets=1)

    def _gen_train_data(self):
        filenames = self._train_file_names
        for file in filenames:
            lines = pd.read_csv(file, delimiter=",", skiprows=1)
            for i in range(len(lines)):
                sentence = lines.iloc[i, 1].strip("\"")
                ids = [self._table.lookup(i) for i in sentence]
                yield ids, lines.iloc[i, range(2, 22, 1)]

    def _load_train_data(self):
        train_dataset = tf.data.Dataset.from_generator(self._gen_train_data, (tf.int64, tf.int64), ([None], [20]))
        train_dataset = train_dataset.padded_batch(self._batch_size, padded_shapes=([None], [None]),
                                                   padding_values=(tf.constant(-1, dtype=tf.int64), tf.constant(0, dtype=tf.int64)))
        self._train_iterator = train_dataset.make_initializable_iterator()
        self._train_batch = self._train_iterator.get_next()

    def _create_embedding(self):
        with tf.name_scope("create——embedding"):
            self._embedding = tf.Variable()


if __name__ == "__main__":
    tf.Graph()
    ml = MoonLight()
    with tf.Session() as sess:
        ml.load()
        ml._load_train_data()
        sess.run(ml._train_iterator.initializer)
        sess.run(tf.tables_initializer())
        while True:
            try:
                print(sess.run(ml._train_batch))
            except tf.errors.OutOfRangeError:
                break
