import tensorflow as tf
from datasets import face_expression_dataset as ds

DATA_ROOT = 'tmp'
BATCH_SIZE = 4

dataset = ds.FaceExpressionDataset(DATA_ROOT, BATCH_SIZE, queue_num_threads=2, queue_min_examples=64)
input_op = dataset.inputs(augment_data=True)

# Create a session for running operations in the Graph.
with tf.Session() as sess:
    # Initialize the variables (like the epoch counter).
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Start input enqueue threads.
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while not coord.should_stop():
            batch_images, batch_labels = sess.run(input_op)
            print('Batch shape: {}, {}'.format(batch_images.shape, batch_labels))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
