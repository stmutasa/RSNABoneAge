
""" Routine for decoding images from the downloaded stack and converting them to usable tensors """

# The functions we will need from python 3.x
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_author_ = 'simi'

# Libraries we will use in input.py
import os
import tensorflow as tf
#from six.movies import xrange

# Define image sizes and color channels for the input data
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_COLORS = 1

# Define the dimensions of our input data
INPUT_HEIGHT = 400
INPUT_WIDTH = 400
INPUT_DEPTH = 1
INPUT_LABEL_BYTES

# How many classes you will have in the source data
NUM_CLASSES = 30

# Define the sizes of the data set
NUM_TRAINING_EXAMPLES = 2000
NUM_EVAL_EXAMPLES = 500

def read_data(filename_queue):
    """Reads and parses the example data.
    Args:
        filename_queue: a queue of strings with the filenames to read from
    Returns:
        An example object with original dimensions"""

    class ImageRecord(object):
        pass

    result = ImageRecord()
    label
