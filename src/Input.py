""" Routine for decoding images from the downloaded stack and converting them to usable tensors """

# The functions we will need from python 3.x
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_author_ = 'Simi'

# Libraries we will use in input.py
import os
import tensorflow as tf
import cv2  # Open CV for image manipulation and importing
import numpy as np
import pickle  # Module for serializing data and saving it to disk for use in another script
import glob

# Define the dimensions of our input data
image_width = 256,  # """Width of the images.""")
image_height = 256,  # """Height of the images.""")

# Define the sizes of the data set
num_training_examples = 2000  # """The amount of examples in the training set""")
num_eval_examples = 100,  # """The amount of examples in the evaluation data set""")
num_test_examples = 500  # """The amount of examples set aside for testing""")

# Define filename for saving the image protobuffers
# Raw data is jpegs numbered from 3128 to 7293 with some entries missing
input_folder = 'data/raw'  # Folder where our raw inputs are stored""")
records_file = 'data'  # """Where to store our records protobuf""")


# Functions to define:
# Write image and label to TFRecords
# Read images and labels from protobuf: Later

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_image(filename, grayscale=True):
    """Reads an image and stores it in a numpy array that is returned.
        Automatically converts to greyscale unless specified """

    if grayscale:  # Make the image into a greyscale by default
        image = cv2.imread(filename, 0)
    else:  # Else keep all the color channels
        image = cv2.imread(filename)

    return image


def read_labels(filename):
    """ Loads a list of serialized labels saved using the pickle module. Returns a dictionary """
    with open(filename, 'rb') as file_holder:
        labels = pickle.load(file_holder)

    return labels


def pre_process_image(image, input_size=[256, 256], padding=[0, 0],
                      interpolation=cv2.INTER_LINEAR, masking=False):
    """ Pre processes the image: Resizes based on the specified input size, padding, and interpolation method """

    # Center the data and divide by the standard deviation
    # image = (image - np.mean(image)) / np.std(image) removed, will have to do later to store smaller protobuf

    # Resize the image
    resize_dims = np.array(input_size) - np.array(padding) * 2  # Different size arrays will be broadcast to the same
    pad_tuple = ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0))  # set bilateral padding in X,Y and Z dims
    image = cv2.resize(image, tuple(resize_dims), interpolation=interpolation)  # Use openCV to resize image
    #    image = np.pad(image, pad_tuple, mode='reflect')  # pad all the dimensions with the pad-tuple OHNOES

    # If defined, use masking to turn 'empty space' in the image into invalid entries that won't be calculated
    if masking == True:
        mask = np.zeros(image.shape, 'bool')  # if masking, create an array with all values set to false
        mask = np.pad(mask, pad_tuple, mode='constant', constant_values=(True))  # pad the mask too
        if interpolation == cv2.INTER_NEAREST:
            image[mask] = 0
        return image, mask

    return image


def img_protobuf(images, labels, name):
    """ Combines the images and labels given and saves them to a TFRecords protocol buffer
        Will call this function one time each to save a training, validation and test set.
        Combine the image[index] as an element in the nested dictionary
        Args:
            images: A Dictionary of our source images
            labels: A 2D Dictionary with image id:{x,y,z} pairs """

    # Next we need to store the original image dimensions for when we restore the protobuf binary to a usable form
    # Pick a random entry in the images dict of arrays as our size model
    # rows = images[random.choice(list(images.keys()))].shape[0]
    # columns = images[random.choice(list(images.keys()))].shape[1]
    rows = 256
    columns = 256
    examples = len(images)
    # depth = images[0].shape[3]  # Depth is not defined since we have one color channel

    filenames = os.path.join(records_file, name + '.tfrecords')  # Set the filenames for the protobuf

    # Define the class we will use to write the records to the .tfrecords protobuf. the init opens the file for writing
    writer = tf.python_io.TFRecordWriter(filenames)

    # Loop through each example and append the protobuf with the specified features
    for index, feature in labels.items():
        if index not in images: continue  # Since we deleted some images, some of the labels won't exist
        # Create our dictionary of values to store: Added some dimensions values that may be useful later on
        data = {'data': images[index],
                'label1': labels[index]['Reading1'], 'label2': labels[index]['Reading2'],
                'height': rows, 'width': columns, 'examples': examples}

        # Create a dictionary for values that will be retreived when we restore the protobuf
        # create the feature data
        feature_data_pre = {'data': images[index],
                            'label1': labels[index]['Reading1'], 'label2': labels[index]['Reading2'],
                            'height': rows, 'width': columns, 'examples': examples}

        feature_data = create_feature_dict(feature_data_pre, index, True)

        example = tf.train.Example(features=tf.train.Features(feature=create_feature_dict(data, index)))
        writer.write(example.SerializeToString())  # Converts data to serialized string and writes it in the protobuf

    writer.close()  # Close the file after writing

    # Save the feature data dictionary too
    savename = os.path.join(records_file, 'boneageloadict')
    with open(savename, 'r+b') as file_handle:
        pickle._dump(feature_data, file_handle)

    return


def create_feature_dict(data_to_write={}, id=1, restore=False):
    """ Create the features of each image:label pair we want to save to our TFRecord protobuf here instead of inline"""

    if not restore:  # Do this for the storage dictionary first
        feature_dict_write = {}  # initialize an empty dictionary
        feature_dict_write['id'] = _int64_feature(
            int(id))  # id is the unique identifier of the image, make it an integer
        for key, feature in data_to_write.items():  # Loop over the dictionary and append the feature list for each id

            # If this is our Data array, use the tostring() method.
            if key == 'data':
                feature_dict_write[key] = _bytes_feature(feature.tostring())

            else:  # Otherwise convert to a string and encode as bytes to pass on
                features = str(feature)
                feature_dict_write[key] = _bytes_feature(features.encode())

        return feature_dict_write

    else:  # Else do this to create the restoration dictionary
        feature_dict_restore = {'id': tf.FixedLenFeature([], tf.int64)}
        for key, feature in data_to_write.items():
            feature_dict_restore[key] = tf.FixedLenFeature([], tf.string)

        return feature_dict_restore


def load_protobuf(num_epochs, input_name, return_dict=True):
    """ This function loads the previously saved protocol buffer and converts it's entries in to a Tensor for use
        in Training. For now, define the variables and dictionaries here locally and define Peters params[] class later
        Args
            filename_queue: the pointer to the protobuf containing all of our data
        Returns:
            Images and Labels: The tensors with the data"""

    # Outputs strings (filenames) to a queue for an input pipeline can do this in train and pass to this function
    # Will generate a random shuffle of the string tensors each epoch
    filedir = os.path.join(records_file, input_name)  # filenames for the protobuf
    filenames = glob.glob(filedir + '*.tfrecords')
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

    reader = tf.TFRecordReader()  # Instantializes a TFRecordReader which outputs records from a TFRecords file
    _, serialized_example = reader.read(filename_queue)  # Returns the next record (key:value) produced by the reader

    # Tutorial implementation Below --------------------------------------------------------------

    # Restore the feature dictionary to store the variables we will retrieve using the parse
    loadname = os.path.join(records_file, 'boneageloadict')
    with open(loadname, 'rb') as file_handle:
        feature_dict = pickle.load(file_handle)

    # Q? Does defining the value in the feature dict as a type force that type on feed? if so why use decode raw

    # Parses one protocol buffer file into the features dictionary which maps keys to tensors with the data
    features = tf.parse_single_example(serialized_example, features=feature_dict)

    # Change the raw image data to floating point integer tensors stored in image
    # To do: Generalize this

    image = tf.decode_raw(features['data'], tf.float32)  # Set this examples image to a blank tensor with float data

    # Use this to set the size of our image tensor to a 1 dimensional tensor
    img_shape = [256 * 256]
    image.set_shape(img_shape)
    image = tf.reshape(image, shape=[256, 256, 1])

    # Image is now a handle to : "("DecodeRaw:0", shape=(65536,), dtype=float32)"

    # Cast all our data to 32 bit floating point units
    image = tf.cast(image, tf.float32)
    label1 = tf.cast(features['label1'], tf.float32)
    label2 = tf.cast(features['label2'], tf.float32)
    id = tf.cast(features['id'], tf.float32)

    # Return data as a dictionary by default, otherwise return it as just the raw sets
    if not return_dict:
        return image, label1, label2, id
    else:
        final_data = {'image': image, 'label1': label1, 'label2': label2}
        returned_dict = {}
        returned_dict['id'] = id
        for key, feature in final_data.items():
            returned_dict[key] = feature
        return returned_dict


def randomize_batches(image_dict, batch_size):
    """ This function takes our full data tensors and creates shuffled batches of data.
        Args:
            images_dict: Dictionary of tensors with the labels we created
            batch_size: How many examples to load (first dimension of  the matrix created)
        Returns:
            train: a dictionary of label: batch of data with that label """

    min_dq = 16  # Min elements to queue after a dequeue to ensure good mixing
    capacity = min_dq + 3 * batch_size  # max number of elements in the queue
    keys, tensors = zip(*image_dict.items())  # Create zip object

    # This function creates batches by randomly shuffling the input tensors. returns a dict of shuffled tensors
    shuffled = tf.train.shuffle_batch(tensors, batch_size=batch_size,
                                      capacity=capacity, min_after_dequeue=min_dq)

    batch_dict = {}  # Dictionary to store our shuffled examples

    # Recreate the batched data as a dictionary with the new batch size
    for key, shuffle in zip(keys, shuffled): batch_dict[key] = shuffle

    return batch_dict
