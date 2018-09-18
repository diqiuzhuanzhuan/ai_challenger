# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf
from tensorflow.contrib import data
from tensorflow.contrib import lookup


class DataFiles:
    _train_file_names = ["./data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv"]
    _validation_file_names = ["./data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv"]
    _test_file_names = ["./data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv"]
    _dict_file = "./data/words.dict"

    _record_defaults = [tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.string), tf.constant([0], dtype=tf.int32),
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


class MoonLight(object):
    _train_file_names = DataFiles._train_file_names
    _validation_file_names = DataFiles._validation_file_names
    _test_file_names = DataFiles._validation_file_names

    _batch_size = 32
    _table = None

    _train_batch = None

    def load(self):
        build_vocab()
        self._table = lookup.index_table_from_file(DataFiles._dict_file, num_oov_buckets=1)

    def _load_train_data(self):
        train_files = self._train_file_names
        train_dataset = data.CsvDataset(train_files, record_defaults=DataFiles._record_defaults, header=True)
        train_dataset = train_dataset.map(lambda x: (x[1], x[1:]))
        train_dataset = train_dataset.padded_batch(self._batch_size, padded_shapes=[])
        train_iterator = train_dataset.make_initializable_iterator()
        self._train_batch = train_iterator.get_next()


if __name__ == "__main__":
    ml = MoonLight()
    ml.load()
    print(ml._table)
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        features = tf.constant(["他", "我", "and", "你"])
        ids = ml._table.lookup(features)
        print(sess.run(ids))
        ml._load_train_data()
        while True:
            try:
                print(sess.run(ml._train_batch))
            except tf.errors.OutOfRangeError:
                break
