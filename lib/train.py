from paths import add_lib_path
add_lib_path()

import tensorflow as tf
from datetime import datetime
import argparse
import os
import sys
from build_model import build_model
from get_data_provider import get_data_provider
from config_provider import get_config
from paths import Paths
import numpy as np
from sklearn.metrics import roc_auc_score


def train_model(model_name, data_name, cfg_name):
    cfg = get_config(cfg_name)
    print('Config:')
    print(cfg)
    data_provider = get_data_provider(data_name)

    x = tf.placeholder(tf.float32, shape=[None, data_provider.input_length], name='x')
    y = tf.placeholder(tf.int64, shape=[None], name='y')  # labels: 0, not repeat buyer; 1, is repeat buyer
    is_training = tf.placeholder(tf.bool, name='is_training')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    logits = build_model(model_name, x, is_training, dropout_keep_prob)
    predicts = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(cfg.initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=1,
                                               decay_rate=cfg.decay_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    global_step_increment = global_step.assign_add(1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('Start training')
        train_loss = 0.
        train_labels = np.ndarray(cfg.display_step * cfg.batch_size, dtype=np.int)
        train_scores = np.ndarray(cfg.display_step * cfg.batch_size, dtype=np.float)
        for step in range(1, cfg.train_iters + 1):
            if step % cfg.decay_step == 0:
                sess.run(global_step_increment)

            data, labels = data_provider.next_batch(cfg.batch_size, 'train')
            batch_loss, _, batch_predict = sess.run([loss, optimizer, predicts],
                                                    feed_dict={x: data,
                                                               y: labels,
                                                               is_training: True,
                                                               dropout_keep_prob: cfg.dropout_keep_prob})
            for train_index_y in range(cfg.batch_size):
                train_index = ((step-1) % cfg.display_step) * cfg.batch_size + train_index_y
                train_labels[train_index] = labels[train_index_y]
                train_scores[train_index] = batch_predict[train_index_y][1]
            train_loss += batch_loss
            # Display training status
            if step % cfg.display_step == 0:
                train_accuracy = roc_auc_score(train_labels, train_scores)
                print("{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}"
                      .format(datetime.now(), step, train_loss / cfg.display_step, train_accuracy))
                train_loss = 0.

            # Snapshot
            if step % cfg.snapshot_step == 0:
                save_path = os.path.join(Paths.output_path,
                                         '{}_{}_{}'.format(data_name, model_name, step))
                saver.save(sess, save_path)

            # Display validation status
            if step % cfg.val_step == 0:
                val_num = int(data_provider.val_size / cfg.batch_size)
                val_labels = np.ndarray(val_num * cfg.batch_size, dtype=np.int)
                val_scores = np.ndarray(val_num * cfg.batch_size, dtype=np.float)
                for val_index_x in range(val_num):
                    data, labels = data_provider.next_batch(cfg.batch_size, 'val')
                    val_predicts = sess.run(predicts, feed_dict={x: data,
                                                                 y: labels,
                                                                 is_training: False,
                                                                 dropout_keep_prob: cfg.dropout_keep_prob})
                    for val_index_y in range(cfg.batch_size):
                        val_index = val_index_x * cfg.batch_size + val_index_y
                        val_labels[val_index] = labels[val_index_y]
                        val_scores[val_index] = val_predicts[val_index_y][1]
                val_accuracy = roc_auc_score(val_labels, val_scores)
                print("{} Iter {}: Validation Accuracy = {:.4f}".format(datetime.now(), step, val_accuracy))

        print('Finish!')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train Repeat Buyer Prediction Model')
    parser.add_argument('--model', dest='model_name',
                        help='model to use',
                        default='dnn', type=str)
    parser.add_argument('--data', dest='data_name',
                        help='data to use',
                        default='', type=str)
    parser.add_argument('--cfg', dest='cfg_name',
                        help='train, val and test config to use',
                        default='', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    train_model(args.model_name, args.data_name, args.cfg_name)
