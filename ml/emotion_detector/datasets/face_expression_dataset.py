"""Merged face expression dataset that is provided using a multi-threaded queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import tensorflow as tf
import utils.path
import utils.image

CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


class FaceExpressionDataset(object):
    def __init__(self, data_root, train_split=0.8,
                 queue_num_threads=8, queue_min_examples=4096):
        self._data_root = data_root
        self.queue_num_threads = queue_num_threads
        self.queue_min_examples = queue_min_examples

        all_filenames, all_labels = _read_files(data_root, '*.png')

        train_data, valid_data = _split_data(all_filenames, all_labels, train_split)
        self.train_filenames = train_data[0]
        self.train_labels = train_data[1]
        # self.valid_filenames = valid_data[0]
        # self.valid_labels = valid_data[1]

        self._valid_index = 0
        self.valid_images, self.valid_labels = _load_dataset_into_memory(valid_data[0], valid_data[1])

    def train_inputs(self, batch_size, augment_data=False):
        """Construct distorted input for training using the Reader ops.
        """
        image_filenames = tf.convert_to_tensor(self.train_filenames, dtype=tf.string)
        image_labels = tf.convert_to_tensor(self.train_labels, dtype=tf.int32)
        image_labels = tf.one_hot(image_labels, len(CLASSES))

        # Create a queue that produces the filenames plus labels to read
        input_queue = tf.train.slice_input_producer([image_filenames, image_labels])

        # Read examples from files in the filename queue
        input_example = DataExample(48, 48, 1)
        input_example.read_example(input_queue)
        result_image = tf.cast(input_example.image, tf.float32)

        # scale values to interval [-1, 1]
        result_image = _normalize_image(result_image)

        if augment_data:
            # Image processing for training the network

            # Randomly crop a [height, width] section of the image
            #result_image = tf.random_crop(result_image,
            #                              [input_example.height,
            #                               input_example.width,
            #                               input_example.depth])

            # Randomly flip the image horizontally.
            result_image = tf.image.random_flip_left_right(result_image)

            # Because these operations are not commutative, consider randomizing
            # the order their operation
            # NOTE: since per_image_standardization zeros the mean and makes
            # the stddev unit, this likely has no effect see tensorflow#1458
            #result_image = tf.image.random_brightness(result_image,
            #                                          max_delta=10)
            #result_image = tf.image.random_contrast(result_image,
            #                                        lower=0.9, upper=1.1)
            # Limit pixel values to [0, 1]
            #result_image = tf.minimum(result_image, 1.0)
            #result_image = tf.maximum(result_image, 0.0)
        ##else:
            # Crop the central [height, width] of the image
            #result_image = tf.image.resize_image_with_crop_or_pad(result_image,
            #                                                      input_example.height, input_example.width)

        result_image.set_shape([input_example.height, input_example.width, input_example.depth])
        # Subtract off the mean and divide by the variance of the pixels
        # result_image = tf.image.per_image_standardization(result_image)

        # Ensure that the random shuffling has good mixing properties
        print('Filling queue with {} images...'.format(self.queue_min_examples))

        # Generate a batch of images and labels by building up a queue of examples.
        return self._generate_image_and_label_batch(result_image, input_example.label,
                                                    batch_size)

    def valid_reset(self):
        self._valid_index = 0

    def valid_batch(self, valid_batch_size):
        end_index = self._valid_index + valid_batch_size
        if end_index > self.valid_size:
            raise "Validation index out of bounds. Forgot to call valid_reset()?"

        batch_images = self.valid_images[self._valid_index:end_index]
        batch_labels = self.valid_labels[self._valid_index:end_index]
        self._valid_index = end_index
        return batch_images, batch_labels

    def _generate_image_and_label_batch(self, image, label, batch_size):
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
            batch_size=batch_size,
            num_threads=self.queue_num_threads,
            capacity=self.queue_min_examples + 32 * batch_size,
            min_after_dequeue=self.queue_min_examples)

        # Display the training images in the visualizer.
        tf.summary.image('images', images)

        return images, tf.reshape(label_batch, [-1, len(CLASSES)])

    @property
    def data_root(self):
        return self._data_root

    @property
    def train_size(self):
        return len(self.train_filenames)

    @property
    def valid_size(self):
        return self.valid_images.shape[0]


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
        self._image = tf.image.decode_png(file_contents)

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


def _read_files(root, file_pattern):
    all_filenames = {}
    all_labels = {}
    for class_id, class_name in enumerate(CLASSES):
        class_filenames = []
        class_labels = []
        filenames = utils.path.get_filenames(os.path.join(root, class_name), file_pattern)
        for fname in filenames:
            class_filenames.append(fname)
            class_labels.append(class_id)
        all_filenames[class_name] = class_filenames
        all_labels[class_name] = class_labels

    return all_filenames, all_labels


def _split_data(filenames_dict, labels_dict, train_split):
    """Splits the data into train and test set.
    Args:
        filenames: List of filenames with shape [class, filenames]
        labels: List of labels with shape [class, label]
        train_split: The size of the train split. The validation data
                     is then 1 - train_split. 
    """
    train_filenames_list = []
    train_labels_list = []
    valid_filenames_list = []
    valid_labels_list = []

    for class_name in CLASSES:
            filenames = filenames_dict[class_name]
            labels = labels_dict[class_name]

            size = len(filenames)
            np_filenames = np.asarray(filenames, dtype=object)
            np_labels = np.asarray(labels)

            # shuffle
            perm = np.random.permutation(size)
            np_filenames = np_filenames[perm]
            np_labels = np_labels[perm]

            # split
            np_train_filenames = np_filenames[:int(size * train_split)]
            np_train_labels = np_labels[:int(size * train_split)]
            np_valid_filenames = np_filenames[int(size * train_split):]
            np_valid_labels = np_labels[int(size * train_split):]

            # add data to the train/valid list
            train_filenames_list += np_train_filenames.tolist()
            train_labels_list += np_train_labels.tolist()
            valid_filenames_list += np_valid_filenames.tolist()
            valid_labels_list += np_valid_labels.tolist()

    return (train_filenames_list, train_labels_list),\
           (valid_filenames_list, valid_labels_list)


def _load_dataset_into_memory(filenames, labels):
    one_hot_labels = np.zeros((len(labels), len(CLASSES)))
    one_hot_labels[np.arange(len(labels)), labels] = 1

    images = []
    for filename, label in zip(filenames, labels):
        image = utils.image.read(filename, cv2.IMREAD_GRAYSCALE)
        image = _normalize_image(image)
        images.append(image)

    return np.asarray(images), one_hot_labels


def _normalize_image(image):
    """Normalizes the image from [0, 255] to [-1, 1] with simple linear scaling."""
    return (image / 127.5) - 1.0
