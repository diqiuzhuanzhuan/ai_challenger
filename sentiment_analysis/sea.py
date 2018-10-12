# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import json
import tensorflow as tf
import os
import urllib
from tensorflow.contrib import data
import pandas as pd
from collections import defaultdict
import numpy as np
import jieba


class DataFiles:
    _train_file_names = ["./data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv"]
    #_train_file_names = ["./data/ai_challenger_sentiment_analysis_trainingset_20180816/train.csv"]
    _train_file_url = ["http://www.diqiuzhuanzhuan.com/download/344/"]
    _validation_file_names = ["./data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv"]
    #_validation_file_names = ["./data/ai_challenger_sentiment_analysis_validationset_20180816/train.csv"]
    _validation_file_url = ["http://www.diqiuzhuanzhuan.com/download/346/"]
    _test_file_names = ["./data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv"]
    _test_file_url = ["http://www.diqiuzhuanzhuan.com/download/334/"]
    _dict_file = "./data/words.dict"
    _lemma_file = "./data/words.lemma"

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


class LookMan(object):

    def __init__(self, dict_file, num_oov_buckets=1):
        self._box = set()
        self._table = {}
        self._num_oov = -1
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


class Config(object):
    _use_lemma = True


class Data(object):
    _labels_num = 20

    _table = None
    _vocab_size = None
    _batch_size = None

    _will_output = None
    _output_cursor = 0
    _weight_file = "weight.txt"

    def __init__(self, batch_size=32, max_length=None):
        self._batch_size = batch_size
        self._max_length = max_length
        self.load_dict()
        self.weights = None

    def _load_weight(self):
        if not os.path.exists(self._weight_file+".npy"):
            self.calc_weight()
        else:
            print("weight文件已经存在，不再重新计算，如需重新计算，请删除文件:{}".format(self._weight_file))
            self.weights = np.load(self._weight_file+".npy")
            print(self.weights)

    def get_vocab_size(self):
        if Config._use_lemma:
            return self._lemma_size
        else:
            return self._vocab_size

    def _build_lemma(self):
        import os
        print("正在准备数据集")
        DataFiles._prepare_data()
        print("数据集已准备完毕")
        if os.path.exists(DataFiles._lemma_file):
            print("词典文件已经存在，不必再生成, 如果你想重新生成，请先手动删除该文件：{}".format(DataFiles._lemma_file))
            return
        print("词典文件还没有构建，正在为您构建字典文件，请稍等......")
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

                    s = jieba.cut(line)
                    [words.add(x) for x in s]
                    lines_count = lines_count + 1
                    print("\r已读取{}行".format(lines_count), end="")
                except tf.errors.OutOfRangeError as e:
                    break
                except Exception as e:
                    print(e)
            tf.assert_equal(lines_count, 135000)

            sess.run(tf.write_file(filename=DataFiles._lemma_file, contents="\n".join(words)))

            print("词语数一共有:{} ".format(words.__len__()))
            print("一共有{}行数据".format(lines_count))
            print("词典典文件构建完毕~~~")

    def _build_vocab(self):
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

    def load_dict(self):
        self._build_vocab()
        self._build_lemma()
        self._table = LookMan(DataFiles._dict_file, num_oov_buckets=1)
        self._lemma_table = LookMan(DataFiles._lemma_file, num_oov_buckets=1)
        self._vocab_size = self._table.size()
        self._lemma_size = self._lemma_table.size()
        print("字典文件加载完成，字典大小为: {}".format(self._vocab_size))
        print("词典文件加载完成，词典大小为: {}".format(self._lemma_size))


    def _get_has_label_data(self, file_names):
        for file in file_names:
            lines = pd.read_csv(file, delimiter=",")
            for i in range(len(lines)):
                sentence = lines.iloc[i, 1].strip("\"")
                if Config._use_lemma:
                    ids = [self._lemma_table.lookup(i) for i in jieba.cut(sentence) ]
                else:
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
        self._will_output = pd.DataFrame()
        for file in file_names:
            lines = pd.read_csv(file, delimiter=",")
            if self._will_output is None:
                self._will_output = pd.DataFrame(lines.columns)
            self._will_output = self._will_output.append(lines.iloc[:])
            self._will_output = self._will_output.fillna(int(0))

            for i in range(len(lines)):
                sentence = lines.iloc[i, 1].strip("\"")
                if Config._use_lemma:
                    ids = [self._lemma_table.lookup(i) for i in jieba.cut(sentence)]
                else:
                    ids = [self._table.lookup(i) for i in sentence]

                yield ids, [len(ids)], [0] * self._labels_num

    def _get_validation_data(self):
        file_names = DataFiles._validation_file_names
        yield from self._get_has_label_data(file_names)

    def _gen_train_data(self):
        file_names = DataFiles._train_file_names
        yield from self._get_has_label_data(file_names)

    def _gen_test_data(self):
        file_names = DataFiles._test_file_names
        yield from self._get_no_label_data(file_names)

    def _load_train_data(self):
        train_dataset = tf.data.Dataset.from_generator(self._gen_train_data, (tf.int64, tf.int64, tf.int64), ([None], [None], [self._labels_num]))
        train_dataset = train_dataset.padded_batch(self._batch_size, padded_shapes=([self._max_length], [None], [None]),
                                                   padding_values=(tf.constant(1, dtype=tf.int64), tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64)))
        train_dataset = train_dataset.map(lambda *x: (x[0], x[1], tf.one_hot(indices=x[2], depth=4, dtype=tf.int64)))
        self._train_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        self._train_iterator_initializer = self._train_iterator.make_initializer(train_dataset)

    def _load_validation_data(self):
        validation_dataset = tf.data.Dataset.from_generator(self._gen_train_data, (tf.int64, tf.int64, tf.int64), ([None], [None], [self._labels_num]))
        validation_dataset = validation_dataset.padded_batch(self._batch_size, padded_shapes=([self._max_length], [None], [self._labels_num]),
                                                             padding_values=(tf.constant(1, dtype=tf.int64), tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64)))
        validation_dataset = validation_dataset.map(lambda *x: (x[0], x[1], tf.one_hot(indices=x[2], depth=4, dtype=tf.int64)))
        self._validation_iterator = tf.data.Iterator.from_structure(validation_dataset.output_types, validation_dataset.output_shapes)
        self._validation_iterator_initializer = self._validation_iterator.make_initializer(validation_dataset)

    def _load_test_data(self):
        test_dataset = tf.data.Dataset.from_generator(self._gen_test_data, (tf.int64, tf.int64, tf.int64), ([None], [None], [self._labels_num]))
        test_dataset = test_dataset.padded_batch(self._batch_size, padded_shapes=([self._max_length], [None], [None]),
                                                 padding_values=(tf.constant(1, dtype=tf.int64), tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64)))
        test_dataset = test_dataset.map(lambda *x: (x[0], x[1], tf.one_hot(indices=x[2], depth=4, dtype=tf.int64)))
        self._test_iterator = tf.data.Iterator.from_structure(test_dataset.output_types,test_dataset.output_shapes)
        self._test_iterator_initializer = self._test_iterator.make_initializer(test_dataset)

    def load_data(self):
        self._load_weight()
        self._load_train_data()
        self._load_validation_data()
        self._load_test_data()
        return self._train_iterator, self._train_iterator_initializer, self._validation_iterator, self._validation_iterator_initializer, self._test_iterator, self._test_iterator_initializer

    def feed_output(self, labels):
        for line in labels:
            self._will_output.iloc[self._output_cursor, 2:] = line
            self._output_cursor = self._output_cursor + 1

    def persist(self, filename="result.csv"):
        if self._output_cursor != len(self._will_output):
            print("测试数据结果不完整")
        columns = self._will_output.columns
        self._will_output[columns[2:]] = self._will_output[columns[2:]].astype(int)
        self._will_output.to_csv(filename, index=False, encoding="utf_8_sig")
        self._output_cursor = 0

    def calc_weight(self):
        self.load_data()
        all = []
        with tf.Session() as sess:
            sess.run(self._train_iterator_initializer)
            while True:
                try:
                    res = sess.run(self._train_iterator.get_next())
                    all.extend(res[2])
                except tf.errors.OutOfRangeError:
                    break

            sess.run(self._validation_iterator_initializer)
            while True:
                try:
                    res = sess.run(self._validation_iterator.get_next())
                    all.extend(res[2])
                except tf.errors.OutOfRangeError:
                    break

            print(all)
            total = len(all)
            self.weights = 1 - (np.sum(all, axis=0)/ total)
            print("weights is{}".format(self.weights))
            np.save(self._weight_file, self.weights)

    def test(self):
        self.load_data()
        with tf.Session() as sess:
            sess.run(self._validation_iterator_initializer)
            while True:
                try:
                    res = sess.run(self._validation_iterator.get_next())
                    print(res)
                except tf.errors.OutOfRangeError:
                    break


if __name__ == "__main__":
    d = Data()
    d.test()
