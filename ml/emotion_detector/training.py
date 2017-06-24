""" Trains a model for facial emotion detection. """
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import logging
import coloredlogs

import tensorflow as tf

from models import dexpression as m
from models.dexpression.conf import dex_hyper_params as hyper_params
from datasets import face_expression_dataset as ds
import utils.tensor

FLAGS = None

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# disable TensorFlow C++ warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train(_):
    """Starts the training. Executed only if run as a script."""
    global_step = tf.Variable(0, trainable=False, name='global_step')
    ph_training = tf.placeholder_with_default(False, [], name='is_training')
    
    with tf.device('/cpu:0') and tf.name_scope('input-pipeline'):
        dataset = ds.FaceExpressionDataset(FLAGS.data_root)
        batch_images, batch_labels = dataset.train_inputs(FLAGS.batch_size,
                                                          augment_data=True)

    with tf.name_scope('inference'):
        classifier = m.DexpressionNet(FLAGS.weight_decay,
                                      hyper_params=hyper_params)
        predictions = classifier.inference(batch_images, batch_labels, ph_training)

    with tf.name_scope('loss-layer'):
        loss_op = classifier.loss(predictions, batch_labels)
        tf.summary.scalar('loss', loss_op)
        total_loss_op = classifier.total_loss(loss_op)

    with tf.name_scope('metrics'):
        metrics = classifier.metrics(predictions, batch_labels)
        accuracy_op = metrics['accuracy']
        tf.summary.scalar('accuracy', accuracy_op)

    with tf.name_scope('optimizer'):
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss_op,
                                                                        global_step=global_step)
    
    saver = tf.train.Saver()

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
       
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.summary_root, 'training'), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.summary_root, 'validation'))

        sess.run(tf.global_variables_initializer())
        logging.info('Model with {} trainable parameters.'.format(utils.tensor.get_num_trainable_params()))

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            logging.info('Start training ...')
            epoch = 0

            step_for_avg = 0
            loss_sum = acc_sum = 0.0

            while not coord.should_stop():
                epoch += 1

                if epoch > FLAGS.train_epochs:
                    break

                logging.info('Starting epoch {} / {}...'.format(epoch, FLAGS.train_epochs))
                sess.run(tf.local_variables_initializer())

                num_batches = int(dataset.train_size / FLAGS.batch_size)
                for b in range(num_batches):

                    _, loss, acc, summary, gstep = sess.run([train_op, loss_op, accuracy_op,
                                                             summary_op, global_step],
                                                            feed_dict={ph_training: True})
                    loss_sum += loss
                    acc_sum += acc
                    step_for_avg += 1

                    if gstep % 50 == 0:
                        # TRAIN LOGGING
                        loss_avg = loss_sum / step_for_avg
                        acc_avg = acc_sum / step_for_avg
                        logging.info('Step {:3d} with loss: {:.5f}, acc: {:.5f}'.format(gstep, loss_avg, acc_avg))
                        loss_sum = acc_sum = 0.0
                        step_for_avg = 0
                        # write to summary
                        train_writer.add_summary(summary, gstep)
                        train_writer.flush()

                # VALIDATION LOGGING
                dataset.valid_reset()
                num_batches = int(dataset.valid_size / FLAGS.batch_size)
                loss_sum = acc_sum = 0.0
                for step in range(num_batches):
                    batch_x, batch_y = dataset.valid_batch(FLAGS.batch_size)
                    loss, acc = sess.run([loss_op, accuracy_op],
                                         feed_dict={batch_images: batch_x, batch_labels: batch_y})
                    loss_sum += loss
                    acc_sum += acc

                gstep = sess.run(global_step)
                loss_avg = loss_sum / num_batches
                acc_avg = acc_sum / num_batches
                logging.info('VALIDATION > Step {:3d} with loss: {:.5f}, acc: {:.5f}'.format(gstep, loss_avg, acc_avg))
                valid_writer.add_summary(summary, gstep)
                valid_writer.flush()

                # if FLAGS.save_checkpoint:
                #    checkpoint_dir = 'checkpoint'  # FIXME different runs would override the same checkpoint!
                #    if not os.path.isdir(checkpoint_dir):
                #        os.makedirs(checkpoint_dir)
                #    # save checkpoint
                #    logging.info('Saving checkpoint...')
                #    save_path = saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'))
                #    logging.info('Model saved in file: {}'.format(save_path))

        except tf.errors.OutOfRangeError:
            logging.info('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--batch_size', type=int, default=64,
                        help='The batch size.')
    PARSER.add_argument('--learning_rate', type=float, default=0.001,
                        help='The initial learning rate.')
    PARSER.add_argument('--train_epochs', type=int, default=5,
                        help='The number of training epochs.')
    PARSER.add_argument('--dropout', type=float, default=0.5,
                        help='The keep probability of the dropout layer.')
    PARSER.add_argument('--weight_decay', type=float, default=0.001,
                        help='The lambda koefficient for weight decay regularization.')
    PARSER.add_argument('--augmentation', type=bool, default=False,
                        help='Whether data augmentation (rotate/shift/...) is used or not.')
    PARSER.add_argument('--dataset_check', type=bool, default=False,
                        help='Whether the dataset should be checked only.')
    PARSER.add_argument('--summary_root', type=str, default='summary',
                        help='The root directory for the summaries.')
    PARSER.add_argument('--data_root', type=str, default='emotions',
                        help='The root directory of the data.')
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=train, argv=[sys.argv[0]] + UNPARSED)