
""" Routine for decoding images from the downloaded stack and converting them to usable tensors """

# The functions we will need from python 3.x
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_author_ = 'Simi'

# Libraries we will use in input.py
import os
import tensorflow as tf
import cv2                            # Open CV for image manipulation and importing
import numpy as np
import cPickle  as pickle             # Module for serializing data and saving it to disk for use in another script

FLAGS = tf.app.flags

# Define image sizes and color channels for the image data
tf.app.flags.DEFINE_integer('image_width', 256, """Width of the images.""")
tf.app.flags.DEFINE_integer('image_height', 256, """Height of the images.""")
tf.app.flags.DEFINE_integer('image_colors', 1, """Color channels of the images.""")

# Define the dimensions of our input data
tf.app.flags.DEFINE_integer('input_width', 256, """Width of the images.""")
tf.app.flags.DEFINE_integer('input_height', 256, """Height of the images.""")
tf.app.flags.DEFINE_integer('input_colors', 3, """Color channels of the images.""")

# How many classes you will have in the source data
tf.app.flags.DEFINE_integer('num_classes', 32, """The number of classes in the data set""")

# Define the sizes of the data set
tf.app.flags.DEFINE_integer('num_training_examples', 2000, """The amount of examples in the training set""")
tf.app.flags.DEFINE_integer('num_eval_examples', 100, """The amount of examples in the evaluation data set""")
tf.app.flags.DEFINE_integer('num_test_examples', 500, """The amount of examples set aside for testing""")

# Define filename for saving the image protobuffers
tf.app.flags.DEFINE_string('input_folder', 'Input', """Folder where inputs are stored""")

# Functions to define:
# Write image and label to TFRecords
# Read images and labels from protobuf: Later

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_image(filename,grayscale=True):
    """Reads an image and stores it in a numpy array that is returned.
        Automatically converts to greyscale unless specified """

    if grayscale==True:     # Make the image into a greyscale by default
        image = cv2.imread(filename, [cv2.CV_LOAD_IMAGE_GREYSCALE])
    else:                   # Else keep all the color channels
        image = cv2.imread(filename)

    return image


def read_labels(filename):
    """ Loads a list of serialized labels saved using the pickle module """
    with open(filename, 'rb') as file_holder:
        labels = pickle.load(file_holder)

    return labels


def pre_process_image(image, input_size=[FLAGS.input_width, FLAGS.input_height], padding=[0, 0],
                      interpolation=cv2.INTER_LINEAR, masking=False):
    """ Pre processes the image: Resizes based on the specified input size, padding, and interpolation method """

    # Center the data and divide by the standard deviation
    image = (image - np.mean(image)) / np.std(image)

    # Resize the image
    resize_dims = np.array(input_size) - np.array(padding)*2    # Different size arrays will be broadcast to the same
    pad_tuple = ((padding[0],padding[0]), (padding[1], padding[1]), (0, 0)) # set bilateral padding in X,Y and Z dims
    image = cv2.resize(image,tuple(resize_dims),interpolation=interpolation) # Use openCV to resize image
    image = np.pad(image, pad_tuple, mode='reflect') # pad all the dimensions with the pad-tuple

    # If defined, use masking to turn 'empty space' in the image into invalid entries that won't be calculated
    if masking==True:
        mask = np.zeros(image.shape, 'bool') # if masking, create an array with all values set to false
        mask = np.pad(mask, pad_tuple, mode='constant', constant_values=(True)) # pad the mask too
        if interpolation == cv2.INTER_NEAREST:
            image[mask] = 0
        return image, mask

    return image


def create_feature_dict(data_to_write, id=None):
    """ Create the features of each image:label pair we want to save to our TFRecord protobuf here instead of inline"""
    feature_dict_write = {}     # initialize an empty dictionary
    feature_dict_write['id'] = _int64_feature(id)   # id is the unique identifier of the image, make it an integer
    for key, feature in data_to_write.items():      # Loop over the dictionary and append the feature list for each id
        # To Do Need if statement to keep our already defined int64's as ints and not bytes
        feature_dict_write[key] = _bytes_feature(feature.tostring())

    return feature_dict_write


def img_protobuf(images, labels, num_examples, name):
    """ Combines the images and labels given and saves them to a TFRecords protocol buffer
        Will call this function one time each to save a training, validation and test set"""

    if images.shape[0] != num_examples:         # Check to see if batch size (# egs) matches the label vector size
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], num_examples))

    # Next we need to store the original image dimensions for when we restore the protobuf binary to a usable form
    rows = images.shape[1]
    columns = images.shape[2]
    depth = images.shape[3]

    filenames = os.path.join(FLAGS.input_folder, name + '.tfrecords') # Set the filenames for the protobuf

    # Define the class we will use to write the records to the .tfrecords protobuf. the init opens the file for writing
    writer = tf.python_io.TFRecordWriter(filenames)

    # Loop through each example and append the protobuf with the specified features
    for index in range(num_examples):
        # First create our dictionary of values to store: Added some dimensions values that may be useful later on
        data = { 'height': _int64_feature(rows),'width': _int64_feature(columns), 'depth': _int64_feature(depth),
                 'label': _int64_feature(int(labels[index])),'data': _bytes_feature(images[index])}


        example = tf.train.Example(features=tf.train.Features(feature=create_feature_dict(data,index)))
        writer.write(example.SerializeToString())

    writer.close()      # Close the file after writing