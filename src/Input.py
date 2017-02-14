
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
import pickle         # Module for serializing data and saving it to disk for use in another script

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
# Raw data is jpegs numbered from 3128 to 7293 with some entries missing
tf.app.flags.DEFINE_string('input_folder', 'data/raw', """Folder where our raw inputs are stored""")
tf.app.flags.DEFINE_string('records_file', 'data', """Where to store our records protobuf""")

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
    """ Loads a list of serialized labels saved using the pickle module. Returns a dictionary """
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


def img_protobuf(images, labels, num_examples, name):
    """ Combines the images and labels given and saves them to a TFRecords protocol buffer
        Will call this function one time each to save a training, validation and test set.
        Combine the image[index] as an element in the nested dictionary
        Args:
            Images: A Dictionary of our source images
            Labels: A 2D Dictionary with image id:{x,y,z} pairs """

    if images.shape[0] != num_examples:         # Check to see if batch size (# egs) matches the label vector size
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], num_examples))

    # Next we need to store the original image dimensions for when we restore the protobuf binary to a usable form
    rows = images[0].shape[1]
    columns = images[0].shape[2]
    depth = images[0].shape[3]

    filenames = os.path.join(FLAGS.records_file, name + '.tfrecords') # Set the filenames for the protobuf

    # Define the class we will use to write the records to the .tfrecords protobuf. the init opens the file for writing
    writer = tf.python_io.TFRecordWriter(filenames)

    # Loop through each example and append the protobuf with the specified features
    for index, feature in labels.iter():
        # First create our dictionary of values to store: Added some dimensions values that may be useful later on
        data = { 'data': _bytes_feature(images[index]),
                 'label1': _bytes_feature(labels[index]['Reading1']),'label2': _bytes_feature(labels[index]['Reading2']),
                 'height': _int64_feature(rows),'width': _int64_feature(columns), 'depth': _int64_feature(depth)}


        example = tf.train.Example(features=tf.train.Features(feature=create_feature_dict(data,index)))
        writer.write(example.SerializeToString())    # Converts data to serialized string and writes it in the protobuf

    writer.close()      # Close the file after writing

    return



def create_feature_dict(data_to_write, id=None):
    """ Create the features of each image:label pair we want to save to our TFRecord protobuf here instead of inline"""
    feature_dict_write = {}     # initialize an empty dictionary
    feature_dict_write['id'] = _int64_feature(id)   # id is the unique identifier of the image, make it an integer
    for key, feature in data_to_write.items():      # Loop over the dictionary and append the feature list for each id
        # To Do Need if statement to keep our already defined int64's as ints and not bytes
        feature_dict_write[key] = _bytes_feature(feature.tostring())

    return feature_dict_write


def load_protobuf(num_epochs, input_name):
    """ This function loads the previously saved protocol buffer and converts it's entries in to a Tensor for use
        in Training. For now, define the variables and dictionaries here locally and define Peters params[] class later
        Args
            filename_queue: the pointer to the protobuf containing all of our data
        Returns:
            Images and Labels: The tensors with the data"""

    # Outputs strings (filenames) to a queue for an input pipeline can do this in train and pass to this function
    # Will generate a random shuffle of the string tensors each epoch
    filenames = os.path.join(FLAGS.records_file, input_name + '.tfrecords')  # filenames for the protobuf
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

    reader = tf.TFRecordReader()        # Instantializes a TFRecordReader which outputs records from a TFRecords file
    _, serialized_example = reader.read(filename_queue)    # Returns the next record (key:value) produced by the reader

    # Tutorial implementation Below --------------------------------------------------------------

    # Create the feature dictionary to store the variables we will retrieve using the parse
    feature_dict = {'id':{'data': tf.FixedLenFeature([], tf.string),
                          'label1': tf.FixedLenFeature([], tf.float32),'label2': tf.FixedLenFeature([], tf.float32),
                          'height': tf.FixedLenFeature([], tf.int64), 'width': tf.FixedLenFeature([], tf.int64),
                          'depth': tf.FixedLenFeature([], tf.int64)}}
    # Q? Does defining the value in the feature dict as a type force that type on feed? if so why use decode raw

    # Parses one protocol buffer file into the features dictionary which maps keys to tensors with the data
    features = tf.parse_single_example(serialized_example, features=feature_dict)

    # Change the raw image data to floating point integer tensors stored in image
    # To do: Generalize this
    image = {}
    for index, value in features.iter():
        image[index] = tf.decode_raw(features['data'], tf.float32)      # Return a tensor with images as a float
        image[index] = image.reshape(image[index], [-1])                # flatten the image data to a linear array

    # Images is now a dictionary of index: flattened tensor pairs

    # Set the shape of the tensor for the image based on the values provided
    img_pixels = feature_dict['height'] * feature_dict['weight'] * feature_dict['width']
    image.set_shape(img_pixels)

    # Save your labels and ID as floating point integers as well
    label = tf.cast(features['label'], tf.float32)
    id = tf.cast(features['id'], tf.float32)

    # Peter implenetation below -----------------------------------------------------------------------

    # features = tf.parse_single_example(serialized_example, features=params['pp_options']['feature_dict_read'])
    #
    # # Since the image raw data comes in as a string, decode it to a vector of numbers
    # for field in params['pp_options']['dtype_key']:
    #     # Decode raw will reinterpret the bytes in the first argument to the type in the second
    #     images[field] = tf.decode_raw(features[field], params['pp_options']['dtype_key'][field][1])
    #     images[field] = self.reshape_images(images[field], field, num_classes=self.params['pp_options']['num_classes'],
    #                                         batch_size=0, squeeze=False)
    #
    #     images[field] = tf.cast(images[field], tf.float32)  # Store the image data as float32 type to stop errors
    #     images['id'] = tf.cast(features['id'], tf.int32)    # Store the Image identifiers as integers.

    return image, label, id

def randomize_batch(images, labels, batch_size, randomize_batch=True):
    """ This function takes our full data tensor of images and creates batches. The batches will be randomized if
        the Variable is set to true
        Args:
            Images: The tensor of all the images loaded
            randomize_batch: Whether to randomize or not
        Returns:
            train: a tensor of """

    # First implement the version with randomization:
    if randomize_batch==True:
        train = {}      # To store our training images as a dictionary of batches
        min_after_dq = 16       # Min elements to queue after a dequeue to ensure good mixing
        capacity = min_after_dq + 3*batch_size      # max # of elements in the queue
        keys, tensors = zip(*images.items())

        # This function creates batches by randomly shuffling the input tensors. returns a dict of tensors
        shuffled = tf.train.shuffle_batch(tensors, batch_size=batch_size,
                                              capacity=capacity, min_after_dequeue=min_after_dq)

        # Merge keys and shuffled into a tuple and set to equal. (I thought zip does this automatically?)
        for key, shuffle in zip(keys, shuffled): train[key] = shuffle

    else:   # Now the version without.
        train = images

    return train
