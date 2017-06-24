import numpy as np
import tensorflow as tf
from datasets import face_expression_dataset as ds

DATA_ROOT = 'tmp'
BATCH_SIZE = 4

dataset = ds.FaceExpressionDataset(DATA_ROOT, BATCH_SIZE, queue_num_threads=2, queue_min_examples=64)
input_images, input_labels = dataset.inputs(augment_data=True)

# Create a session for running operations in the Graph.
with tf.Session() as sess:
    # Initialize the variables (like the epoch counter).
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

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
                gen_image = np.random.rand(BATCH_SIZE, 48, 48, 3)
                gen_label = np.ones((BATCH_SIZE,))
                batch_images, batch_labels = sess.run([input_images, input_labels],
                                                      feed_dict={input_images: gen_image,
                                                                 input_labels: gen_label})
            print('@{:5d} Batch shape: {}, {}'.format(step, batch_images.shape, batch_labels))
            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
