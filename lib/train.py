import sys
import os
lib_path = os.getcwd()
print(lib_path)
sys.path.insert(0, lib_path)
print('System path:')
print(sys.path)

import tensorflow as tf
from datetime import datetime
import argparse
import sys
from build_model import build_model
from data_provider import DataProvider
from config_provider import get_config
from paths import Paths


def train_model(model_name, data_name, cfg_name):
    cfg = get_config(cfg_name)
    print('Config:')
    print(cfg)
    input_data = DataProvider(data_name)

    x = tf.placeholder(tf.float32, shape=[cfg.batch_size, input_data.input_length], name='x')
    y = tf.placeholder(tf.int64, shape=[cfg.batch_size], name='y')  # labels: 0, not repeat buyer; 1, is repeat buyer
    logits = build_model(model_name, x)
    predicts = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=cfg.learning_rate).minimize(loss)
    correct_predict = tf.equal(tf.argmax(logits, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name='accuracy')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('Start training')
        train_loss = 0.
        train_accuracy = 0.
        for step in range(1, cfg.train_iters + 1):
            data, labels = input_data.next_batch(cfg.batch_size, 'train')
            batch_loss, _, batch_accuracy, batch_predict = sess.run([loss, optimizer, accuracy, predicts],
                                                                    feed_dict={x: data, y: labels})
            train_loss += batch_loss
            train_accuracy += batch_accuracy
            # Display training status
            if step % cfg.display_step == 0:
                print("{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}"
                      .format(datetime.now(), step, train_loss / cfg.display_step, train_accuracy / cfg.display_step))
                train_loss = 0.
                train_accuracy = 0.

            # Snapshot
            if step % cfg.snapshot_step == 0:
                save_path = os.path.join(Paths.output_path,
                                         '{}_{}_{}'.format(data_name, model_name, step))
                saver.save(sess, save_path)

            # Display validation status
            if step % cfg.val_step == 0:
                val_accuracy = 0.
                val_num = int(input_data.val_size / cfg.batch_size)
                for _ in range(val_num):
                    data, labels = input_data.next_batch(cfg.batch_size, 'val')
                    acc = sess.run(accuracy, feed_dict={x: data, y: labels})
                    val_accuracy += acc
                val_accuracy /= val_num
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
