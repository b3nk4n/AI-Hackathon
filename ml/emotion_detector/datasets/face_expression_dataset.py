"""Merged face expression dataset that is provided using a multi-threaded queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


class FaceExpressionDataset(object):
    def __init__(self, data_root, batch_size,
                 queue_num_threads=8, queue_min_examples=4096):
        self.data_root = data_root
        self.batch_size = batch_size
        self.queue_num_threads = queue_num_threads
        self.queue_min_examples = queue_min_examples

    def inputs(self, augment_data=False):
        """Construct distorted input for training using the Reader ops.
        Args:
        Returns:
          images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
          labels: Labels. 1D tensor of [batch_size] size.
        """
        filenames = tf.train.match_filenames_once(os.path.join(self.data_root, '*.jpg'))

        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue
        input_example = DataExample(48, 48, 3)
        input_example.read_example(filename_queue)
        reshaped_image = tf.cast(input_example.image, tf.float32)

        if augment_data:
            # Image processing for training the network

            # Randomly crop a [height, width] section of the image
            distorted_image = tf.random_crop(reshaped_image,
                                             [input_example.height, input_example.width, 3])  # TODO no crop? crop center?

            # Randomly flip the image horizontally.
            distorted_image = tf.image.random_flip_left_right(distorted_image)

            # Because these operations are not commutative, consider randomizing
            # the order their operation
            # NOTE: since per_image_standardization zeros the mean and makes
            # the stddev unit, this likely has no effect see tensorflow#1458
            distorted_image = tf.image.random_brightness(distorted_image,
                                                         max_delta=50)
            distorted_image = tf.image.random_contrast(distorted_image,
                                                       lower=0.5, upper=1.5)
            # Limit pixel values to [0, 1]
            distorted_image = tf.minimum(distorted_image, 1.0)
            result_image = tf.maximum(distorted_image, 0.0)
        else:
            # Crop the central [height, width] of the image
            result_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                                  input_example.height, input_example.width)

        # Subtract off the mean and divide by the variance of the pixels
        processed_image = tf.image.per_image_standardization(result_image)

        # Set the shapes of tensors
        processed_image.set_shape([input_example.height, input_example.width, 3])
        input_example.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties
        print('Filling queue with {} images...'.format(self.queue_min_examples))

        # Generate a batch of images and labels by building up a queue of examples.
        return self._generate_image_and_label_batch(processed_image, input_example.label)

    def _generate_image_and_label_batch(self, image, label):
        """Construct a queued batch of images and labels.
        Args:
          image: 3-D Tensor of [height, width, 3] of type.float32.
          label: 1-D Tensor of type.int32
        Returns:
          images: Images. 4D tensor of [batch_size, height, width, 3] size.
          labels: Labels. 1D tensor of [batch_size] size.
        """
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=self.batch_size,
            num_threads=self.queue_num_threads,
            capacity=self.queue_min_examples + 32 * self.batch_size,
            min_after_dequeue=self.queue_min_examples)

        # Display the training images in the visualizer.
        tf.summary.image('images', images)

        return images, tf.reshape(label_batch, [self.batch_size])


class DataExample(object):
    """Model class for a single data example."""
    def __init__(self, height, width, depth):
        self._height = height
        self._width = width
        self._depth = depth
        self._key = None
        self._image = None
        self._label = None

    def read_example(self, filename_queue):
        """Reads and parses examples from the data files.
        Recommendation: if you want N-way read parallelism, call this function
        N times.  This will give you N independent Readers reading different
        files & positions within those files, which will give better mixing of
        examples.
        Args:
          filename_queue: A queue of strings with the filenames to read from.
        Returns:
          An object representing a single example, with the following fields:
            height: number of rows in the record (48)
            width: number of columns in the record (48)
            depth: number of color channels in the record (3)
            key: a scalar string Tensor describing the filename & record number
              for this example.
            label: an int32 Tensor with the label in the range 0..9.
            image: a [height, width, depth] uint8 Tensor with the image data
        """
        # Read a record, getting filenames from the filename_queue
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)

        # Decode the image
        self._image = tf.image.decode_png(value)

        # Decode the label
        self._label = 0  # TODO how to get the label, when this is implicitely encoded in the path (folder name)

    @property
    def height(self):
        return self.height

    @property
    def width(self):
        return self.width

    @property
    def depth(self):
        return self.depth

    @property
    def depth(self):
        return self.depth

    @property
    def image(self):
        return self.image

    @property
    def label(self):
        return self.label
