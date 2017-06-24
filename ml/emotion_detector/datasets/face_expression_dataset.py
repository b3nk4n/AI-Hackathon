"""Merged face expression dataset that is provided using a multi-threaded queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import utils.path

CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


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
        class_filenames = []
        class_labels = []
        for class_id, class_name in enumerate(CLASSES):
            filenames = utils.path.get_filenames(os.path.join(self.data_root, class_name), '*.jpg')
            for fname in filenames:
                class_filenames.append(fname)
                class_labels.append(class_id)

        image_filenames = tf.convert_to_tensor(class_filenames, dtype=tf.string)
        image_labels = tf.convert_to_tensor(class_labels, dtype=tf.int32)

        # Create a queue that produces the filenames plus labels to read
        input_queue = tf.train.slice_input_producer([image_filenames, image_labels])

        # Read examples from files in the filename queue
        input_example = DataExample(48, 48, 3)
        input_example.read_example(input_queue)
        result_image = tf.cast(input_example.image, tf.float32)

        if augment_data:
            # Image processing for training the network

            # Randomly crop a [height, width] section of the image
            result_image = tf.random_crop(result_image,
                                          [input_example.height, input_example.width, 3])  # TODO do crop? crop center?

            # Randomly flip the image horizontally.
            result_image = tf.image.random_flip_left_right(result_image)

            # Because these operations are not commutative, consider randomizing
            # the order their operation
            # NOTE: since per_image_standardization zeros the mean and makes
            # the stddev unit, this likely has no effect see tensorflow#1458
            result_image = tf.image.random_brightness(result_image,
                                                      max_delta=50)
            result_image = tf.image.random_contrast(result_image,
                                                    lower=0.5, upper=1.5)
            # Limit pixel values to [0, 1]
            result_image = tf.minimum(result_image, 1.0)
            result_image = tf.maximum(result_image, 0.0)
        else:
            # Crop the central [height, width] of the image
            result_image = tf.image.resize_image_with_crop_or_pad(result_image,
                                                                  input_example.height, input_example.width)

        # Subtract off the mean and divide by the variance of the pixels
        processed_image = tf.image.per_image_standardization(result_image)

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

    def read_example(self, input_queue):
        """Reads and parses examples from the data files.
        Recommendation: if you want N-way read parallelism, call this function
        N times.  This will give you N independent Readers reading different
        files & positions within those files, which will give better mixing of
        examples.
        Args:
          input_queue: An input queue of (strings, int) with the filenames plus label to read from.
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
        file_contents = tf.read_file(input_queue[0])

        # Decode the image
        self._image = tf.image.decode_jpeg(file_contents)

        # Decode the label
        self._label = input_queue[1]

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def depth(self):
        return self._depth

    @property
    def image(self):
        return self._image

    @property
    def label(self):
        return self._label
