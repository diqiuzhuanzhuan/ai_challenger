# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf
import os
import numpy as np
import time
from sea import Data, Config
from sklearn.metrics import f1_score
from mul_text_cnn import TextCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.flags.DEFINE_integer("embedding_size", 256, "Dimensionality of character embedding (default: 256)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 256)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_integer("max_length", 3000, "Max length of sentence")
tf.flags.DEFINE_integer("labels_num", 20, "class num of task")
tf.flags.DEFINE_integer("output_dimension", 4, "output dimension")
tf.flags.DEFINE_boolean("use_lemma", False, "if use lemma or not")

tf.flags.DEFINE_integer("step_bypass_validation", 30000, "how many steps was run before we start to run the first validation?")
tf.flags.DEFINE_integer("step_validation", 3000, "validation run every many steps")
tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("batch_size", 64, "batch_size")
tf.flags.DEFINE_float("learning_rate", 0.1, "initial learning rate")
tf.flags.DEFINE_string("summary_path", "./graphs/", "summary path")
tf.flags.DEFINE_string("checkpoint_path", "./checkpoint", "Checkpoint file path for saving")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def load_data():
    data = Data(batch_size=FLAGS.batch_size, max_length=FLAGS.max_length)
    data.load_data()
    return data


def main():
    Config._use_lemma = FLAGS.use_lemma

    with tf.Graph().as_default():
        data = load_data()
        weight = data.weights
        vocab_size = data.get_vocab_size()
        train_next = data._train_next
        train_iterator_initializer = data._train_iterator_initializer
        validation_next = data._validation_next
        validation_iterator_initializer = data._validation_iterator_initializer

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(batch_size=FLAGS.batch_size,
                          learning_rate=FLAGS.learning_rate,
                          embedding_size=FLAGS.embedding_size,
                          vocab_size=vocab_size,
                          sequence_length=FLAGS.max_length,
                          num_filters=FLAGS.num_filters,
                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                          weight=weight,
                          labes_num=FLAGS.labels_num,
                          output_dimension=FLAGS.output_dimension)

            writer = tf.summary.FileWriter('graphs/text_cnn/learning_rate' + str(cnn._learning_rate))
            if not os.path.exists(FLAGS.checkpoint_path):
                os.makedirs(FLAGS.checkpoint_path)
            checkpoints_prefix = os.path.join(FLAGS.checkpoint_path, "text_cnn")
            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())

            # 是否需要恢复模型
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                print("正在从{}加载模型".format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)

            def train_step(feature, feature_len, label):
                feed_dict = {
                    cnn._keep_prob: 1.0, cnn._feature: feature, cnn._feature_length: feature_len, cnn._label: label
                }
                _, loss, global_step, summary_op, actual_batch_size = sess.run(
                    [cnn._train_total, cnn._total_loss, cnn.global_step, cnn._summary_op, tf.shape(feature)[0]],
                    feed_dict=feed_dict
                )
                writer.add_summary(summary_op, global_step=global_step)
                return loss, global_step, actual_batch_size

            def validation_step(feature, feature_len, label):
                feed_dict = {
                    cnn._keep_prob: 1.0, cnn._feature: feature, cnn._feature_length: feature_len, cnn._label: label
                }

                loss, actual_batch_size, lab, res = sess.run(
                    [cnn._total_loss, tf.shape(feature)[0], tf.argmax(label, axis=2) - 2, cnn.predict - 2],
                    feed_dict=feed_dict
                )
                return loss, actual_batch_size, lab, res

            def train():
                while True:
                    try:
                        t1 = time.time()
                        feature, feature_len, label = sess.run(train_next)
                        loss, step, actual_batch_size = train_step(feature, feature_len, label)
                        delta_t = time.time() - t1
                        print("training: step is {}, loss is {}, cost {} 秒".format(step, loss, delta_t))
                        if step > FLAGS.step_bypass_validation and step % FLAGS.step_validation == 0:
                            saver.save(sess, save_path=checkpoints_prefix, global_step=step)
                            validation()
                        if step % FLAGS.step_validation == 0:
                            saver.save(sess, save_path=checkpoints_prefix, global_step=step)

                    except tf.errors.OutOfRangeError:
                        break

            def validation():
                f1 = 0
                samples = 0
                total_validation_loss = 0
                total_time = 0
                sess.run(validation_iterator_initializer)
                while True:
                    try:
                        t1 = time.time()
                        feature, feature_len, label = sess.run(validation_next)
                        loss, actual_batch_size, lab, res = validation_step(feature, feature_len, label)
                        f1 += np.sum(list(map(lambda x: f1_score(x[0], x[1], average="macro"), zip(lab.tolist(), res.tolist()))))
                        samples += actual_batch_size
                        total_validation_loss += loss
                        delta_t = time.time() - t1
                        total_time += delta_t
                        print("当前f1为:{}, loss 为{}, 花费{}秒".format(f1/samples, loss, delta_t))

                    except tf.errors.OutOfRangeError:
                        print("平均f1为:{}, 平均loss为{}, 总耗时{}秒".format(f1/samples, total_validation_loss/samples, total_time))
                        break

            for i in range(FLAGS.num_epochs):
                sess.run(train_iterator_initializer)
                print("第{}个epoch".format(i))
                train()


if __name__ == "__main__":
    main()
