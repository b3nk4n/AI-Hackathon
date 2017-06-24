""" Trains a model for facial emotion detection. """
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse

import tensorflow as tf

from models import dexpression as m
from datasets import face_expression_dataset as ds
import utils.tensor

FLAGS = None

# disable TensorFlow C++ warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train(_):
    """Starts the training. Executed only if run as a script."""
    ph_training = tf.placeholder_with_default(False, [], name='is_training')
    
    with tf.device('/cpu:0') and tf.name_scope('input-pipeline'):
        dataset = ds.FaceExpressionDataset(FLAGS.data_root, FLAGS.batch_size)
        batch_images, batch_labels = dataset.inputs(augment_data=True)

    with tf.name_scope('inference'):
        classifier = m.DexpressionClassifier(FLAGS.weight_decay,
                                             hyper_params=None)  # TODO hyperparams
        predictions = classifier.inference(batch_images, batch_labels, ph_training)

    with tf.name_scope('loss-layer'):
        loss_op = classifier.loss(predictions, batch_labels)
        total_loss_op = classifier.total_loss(loss_op)

    with tf.name_scope('optimizer'):
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss_op)
    
    saver = tf.train.Saver()

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
       
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'training'), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'validation'))

        sess.run(tf.global_variables_initializer())
        print('Model with {} trainable parameters.'.format(utils.tensor.get_num_trainable_params()))

        print('Start training...')
        step = 1
        loss_sum = loss_n = 0

        for epoch in range(FLAGS.train_epochs):
            print('Starting epoch {} / {}...'.format(epoch + 1, FLAGS.train_epochs))
            sess.run(tf.local_variables_initializer())

            num_batches = 0 # TODO calc number of batches
            for b in range(num_batches):
                # TODO get next train batch and feed it, or use the queue
                
                _, loss, summary = sess.run([train_op, loss_op, summary_op],
                                            feed_dict={})

                loss_sum += loss
                loss_n += 1

                if step % 10 == 0:
                    loss_avg = loss_sum / loss_n
                    print('Step {:3d} with loss: {:.5f}'.format(step, loss_avg))
                    loss_sum = loss_n = 0
                    # write to summary
                    train_writer.add_summary(summary, step)
                    train_writer.flush()

                step += 1

            # validation step
            # TODO get next train batch and feed it, or use the queue, and use a loop in case we cannot read all data at once
            loss, summary = sess.run([loss_op, summary_op], 
                                     feed_dict={})
            print('VALIDATION > Step {:3d} with loss: {:.5f}'.format(step, loss))
            valid_writer.add_summary(summary, step)
            valid_writer.flush()

        if FLAGS.save_checkpoint:
            checkpoint_dir = 'checkpoint' # FIXME different runs would override the same checkpoint!
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # save checkpoint
            print('Saving checkpoint...')
            save_path = saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'))
            print('Model saved in file: {}'.format(save_path))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--batch_size', type=int, default=64,  # large batch size (>>100) gives much better results
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
    PARSER.add_argument('--data_root', type=str, default='tmp',
                        help='The root directory of the data.')
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=train, argv=[sys.argv[0]] + UNPARSED)