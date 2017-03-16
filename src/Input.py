""" Routine for decoding images from the downloaded stack and converting them to usable tensors """

# The functions we will need from python 3.x
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_author_ = 'Simi'

# Libraries we will use in input.py
import os
import tensorflow as tf
import matplotlib.image as mpimg
# from scipy import misc
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
        image = mpimg.imread(filename)
    else:  # Else keep all the color channels
        image = mpimg.imread(filename)

    return image


def read_labels(filename):
    """ Loads a list of serialized labels saved using the pickle module. Returns a dictionary """
    with open(filename, 'rb') as file_holder:
        labels = pickle.load(file_holder)

    return labels


def img_protobuf(images, labels, name, gender='F'):
    """ Combines the images and labels given and saves them to a TFRecords protocol buffer
        Will call this function one time each to save a training, validation and test set.
        Combine the image[index] as an element in the nested dictionary
        Args:
            images: A Dictionary of our source images
            labels: A 2D Dictionary with image id:{x,y,z} pairs """

    # Next we need to store the original image dimensions for when we restore the protobuf binary to a usable form
    rows = 256
    columns = 256
    examples = len(images)

    filenames = os.path.join(records_file, name + '.tfrecords')  # Set the filenames for the protobuf

    # Define the class we will use to write the records to the .tfrecords protobuf. the init opens the file for writing
    writer = tf.python_io.TFRecordWriter(filenames)
    counter = 0  # for testing
    # Loop through each example and append the protobuf with the specified features
    for index, feature in labels.items():
        if index not in images: continue  # Since we deleted some images, some of the labels won't exist

        # Skip the gender not specified by the user
        if labels[index]['Gender'] != gender:
            counter += 1
            print('skipping another, Skipped count: %s' % counter)
            continue

        # Create our dictionary of values to store: Added some dimensions values that may be useful later on
        data = {'data': images[index],
                'label1': labels[index]['Reading1'], 'label2': labels[index]['Reading2'],
                'height': rows, 'width': columns, 'examples': examples}

        example = tf.train.Example(features=tf.train.Features(feature=create_feature_dict(data, index)))
        writer.write(example.SerializeToString())  # Converts example to serialized string and writes it in the protobuf

    writer.close()  # Close the file after writing

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
                feature_dict_write[key] = _bytes_feature(feature.tobytes())  #

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

    # Restore the feature dictionary to store the variables we will retrieve using the parse
    loadname = os.path.join(records_file, 'boneageloadict')
    with open(loadname, 'rb') as file_handle:
        feature_dict = pickle.load(file_handle)

    # Parses one protocol buffer file into the features dictionary which maps keys to tensors with the data
    features = tf.parse_single_example(serialized_example, features=feature_dict)

    # Change the raw image data to 8 bit integers first
    image = tf.decode_raw(features['data'], tf.uint8)  # Set this examples image to a blank tensor with integer data
    image = tf.reshape(image, shape=[512, 512, 1])  # Set the dimensions of the image ( must equal input dims here)

    # Cast all our data to 32 bit floating point units. Cannot convert string to number unless you use that function
    image = tf.cast(image, tf.float32)
    id = tf.cast(features['id'], tf.float32)
    label1 = tf.string_to_number(features['label1'], tf.float32)
    label2 = tf.string_to_number(features['label2'], tf.float32)

    # Apply image pre processing here:
    image = tf.image.random_flip_left_right(image)  # First randomly flip left/right
    image = tf.image.random_flip_up_down(image)  # Up/down flip
    image = tf.image.random_brightness(image, max_delta=0.5)  # Apply random brightness
    image = tf.image.per_image_standardization(image=image)  # Subtract mean and div by variance

    # Resize images
    image = tf.image.resize_images(image, [320, 320])
    image = tf.random_crop(image, [256, 256, 1])  # Random crop the image to a box 80% of the size

    # create float summary image
    tf.summary.image('Normalized Image', tf.reshape(image, shape=[1, 256, 256, 1]), max_outputs=1)

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
