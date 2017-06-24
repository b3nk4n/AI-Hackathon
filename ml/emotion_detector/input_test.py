import numpy as np
import tensorflow as tf
from datasets import face_expression_dataset as ds

DATA_ROOT = 'emotions'
BATCH_SIZE = 4
TEST_TRAINING = True  # False: test validation


def main():
    dataset = ds.FaceExpressionDataset(DATA_ROOT, queue_num_threads=2, queue_min_examples=64)

    # dataset info
    print('Batch-size: {}'.format(BATCH_SIZE))
    print('Train-size: {}'.format(dataset.train_size))
    print('Valid-size: {}'.format(dataset.valid_size))

    # create input pipeline
    queue_images, queue_labels = dataset.train_inputs(BATCH_SIZE, augment_data=True)

    # Create a session for running operations in the Graph.
    with tf.Session() as sess:
        # Initialize the variables (like the epoch counter).
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if TEST_TRAINING:
            test_training_input(sess, queue_images, queue_labels)
        else:
            test_validation_input(dataset)

        sess.close()


def test_training_input(sess, input_images, input_labels):
    # Start input enqueue threads.
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        step = 1
        while not coord.should_stop():
            if step % 2 == 0:
                # get example from queue
                batch_images, batch_labels = sess.run([input_images, input_labels])
            else:
                # get example using feeding
                gen_image = np.random.rand(BATCH_SIZE, 48, 48, 1)
                gen_label = np.ones((BATCH_SIZE, 7))
                batch_images, batch_labels = sess.run([input_images, input_labels],
                                                      feed_dict={input_images: gen_image,
                                                                 input_labels: gen_label})
            print('@{:5d} Batch shape: {}, {}; min: {}, max: {}'.format(step, batch_images.shape, batch_labels,
                                                                        np.min(batch_images), np.max(batch_images)))
            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)


def test_validation_input(dataset):
    while True:
        dataset.valid_reset()

        num_batches = int(dataset.valid_size / BATCH_SIZE)
        for step in range(num_batches):
            batch_x, batch_y = dataset.valid_batch(BATCH_SIZE)
            print(batch_x.shape, batch_y.shape)
            print('image min/max:', np.min(batch_x), np.max(batch_x))


if __name__ == '__main__':
    main()



